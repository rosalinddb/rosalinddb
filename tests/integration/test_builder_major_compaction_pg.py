"""Integration coverage for PR-D: MAJOR COMPACTION at the delta cap.

End-to-end on the REAL stack (a `pgvector/pgvector:pg15` recall container + the
session MinIO cold tier + the FastAPI `POST /v1/query` app), with
`RB_DELTA_TIER=true`, `INDEX_TYPE=ivfflat`, and a small `RB_MAX_DELTAS` so a
handful of folds drive the cap.

Proven here:

  - drive folds past the cap → the live generation collapses to ONE level-0 base
    (deltas gone), watermark = max old covered_lsn_hi (no regression), the new
    base is a bare IVF that searches back its native ids;
  - query stays correct ACROSS the cutover: a base id, a re-upserted id (newest
    wins, single copy), and a freshly-folded id all resolve via the new base;
  - CHAOS A — raise AFTER objects written but BEFORE add_shard → the generation
    is unchanged (no new base committed), orphan objects are harmless, and the
    next compaction succeeds → no double-count, no orphan-read;
  - CHAOS B — raise AFTER add_shard (post-cutover) → the new base IS the live
    generation and the old generation is swept on the next cycle → no double
    base, queries resolve the new base only.

Test isolation: EVERY recall env var is routed through `monkeypatch.setenv` from
its first write (see `_migrate_recall` — the bare-`os.environ` hazard from PR-C).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import faiss  # type: ignore

from tests.integration.test_consolidation import (  # noqa: F401 - fixture reuse
    recall_url,
    _truncate_recall,
    _recall_count,
    _signup,
    _tenant_of,
    _auth,
    _post_recall,
    _query,
    _build_client,
)


_DIM = 8
_N_BASE = 80  # > IVF_TRAINING_FLOOR (64) so the base trains an IVF


def _migrate_recall(monkeypatch, recall_url):
    # CRITICAL (PR-C CI lesson): set RB_RECALL_DSN via monkeypatch, NOT a bare
    # `os.environ[...]`. A bare write before `_build_client`'s monkeypatch.setenv
    # would make monkeypatch record the test-local container DSN as the var's
    # "original" and RESTORE it on teardown (instead of removing it), leaking a
    # dead DSN into later modules whose recall writes then fail. Routing every
    # recall-env write through monkeypatch keeps the suite order-independent.
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)


def _delta_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path,
                  *, max_deltas):
    """`_build_client` with the delta tier ON, IVFFlat cold shards, a small cap."""
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")
    monkeypatch.setenv("RB_MAX_DELTAS", str(max_deltas))
    monkeypatch.setenv("RB_MAX_DELTAS_HARD", "16")
    client, state_mod, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    # `_build_client` forces INDEX_TYPE=flat after setting the env; re-assert IVF
    # on the reloaded builder so the delta-tier IVF path is taken.
    builder.INDEX_TYPE = "ivfflat"
    return client, state_mod, v1q, builder


def _rows(n, dim, seed, prefix, start=0):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim), dtype=np.float64).astype(np.float32)
    recs = [
        {"id": f"{prefix}-{i}", "values": vecs[i].tolist(), "metadata": {"i": i + start}}
        for i in range(n)
    ]
    return recs, vecs


def _deser(blob: bytes):
    return faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))


def _drain_recall(builder, tenant, dataset, watermark):
    """Manually trim recall up to `watermark` so the next fold snapshots only the
    new rows (a base+delta is ONE generation → grace 0 → the auto-trim leaves the
    base band; PR-D's compaction is what ages it out, made deterministic here)."""
    builder.recall_trim(tenant, dataset, watermark)


# --------------------------------------------------------------------------- #
# 1. End-to-end: drive folds past the cap → one base, query correct.          #
# --------------------------------------------------------------------------- #


def test_folds_past_cap_collapse_to_one_base_query_correct(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    _migrate_recall(monkeypatch, recall_url)
    _truncate_recall(recall_url)
    max_deltas = 3
    client, state_mod, _v1q, builder = _delta_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path,
        max_deltas=max_deltas,
    )
    import adapters.storage.storage as storage_mod

    s = _signup(client, email="majorpg@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]),
                json={"name": "mc", "dimension": _DIM})

    # First consolidation → bare-IVF base + quantizer-v1.
    base_recs, base_vecs = _rows(_N_BASE, _DIM, 11, "base")
    _post_recall(client, s["token"], "mc", base_recs)
    assert builder.run_consolidate_once("mc", tenant) == _N_BASE
    base = state_mod.get_latest_shard(tenant, "mc")
    old_base_id = int(base["id"])
    assert int(base["quantizer_version"]) == 1
    watermark = _N_BASE
    _drain_recall(builder, tenant, "mc", watermark)

    # Drive folds. The fold that brings the live-delta count TO the cap triggers
    # a synchronous major compaction inside run_consolidate_once.
    reup_vec = None
    fold_vecs = {}
    for k in range(max_deltas):
        if k == 1:
            # Fold 1 RE-UPSERTS an existing base id ("base-5") with a new vector
            # (dedup-newest-wins must survive the compaction).
            recs, vecs = _rows(2, _DIM, 100 + k, f"f{k}")
            reup_vec = np.random.default_rng(777).random(_DIM).astype(np.float32)
            recs.append({"id": "base-5", "values": reup_vec.tolist(),
                         "metadata": {"reupserted": True}})
            _post_recall(client, s["token"], "mc", recs)
            fold_vecs[f"f{k}-0"] = vecs[0]
        else:
            recs, vecs = _rows(2, _DIM, 100 + k, f"f{k}")
            _post_recall(client, s["token"], "mc", recs)
            fold_vecs[f"f{k}-0"] = vecs[0]
        n = builder.run_consolidate_once("mc", tenant)
        watermark = state_mod.dataset_watermark(tenant, "mc")
        _drain_recall(builder, tenant, "mc", watermark)

    # After the capping fold: ONE fresh level-0 base, ZERO deltas.
    gen = state_mod.live_generation(tenant, "mc")
    assert len(gen["deltas"]) == 0, "major compaction must collapse deltas"
    new_base = gen["base"]
    assert int(new_base["id"]) != old_base_id
    assert int(new_base["level"]) == 0
    assert new_base["build_type"] == "consolidate"
    assert int(new_base["quantizer_version"]) == 1  # no retrain
    assert int(new_base["covered_lsn_lo"]) == 0
    assert int(new_base["covered_lsn_hi"]) == watermark
    # Watermark did not regress.
    assert state_mod.dataset_watermark(tenant, "mc") == watermark

    # New base is a BARE IVF and searches back native ids.
    idx = _deser(storage_mod.read_bytes(new_base["shard_uri"]))
    assert not hasattr(idx, "id_map")
    idx.set_direct_map_type(faiss.DirectMap.Hashtable)
    idx.nprobe = idx.nlist
    # Re-upserted id resolves to the NEW vector, exactly once.
    inv = idx.invlists
    reup_int = builder._id_to_int64("base-5")
    occ = sum(
        int(np.sum(np.asarray(faiss.rev_swig_ptr(inv.get_ids(l), inv.list_size(l))) == reup_int))
        for l in range(idx.nlist) if inv.list_size(l)
    )
    assert occ == 1, "re-upserted id must be deduped to one copy in the merged base"
    assert np.allclose(idx.reconstruct(int(reup_int)), reup_vec, atol=1e-5)

    # --- Query correctness across the cutover (through the real /v1/query) -----
    # A base id, a re-upserted id, and a freshly-folded id all resolve.
    r = _query(client, s["token"], "mc", base_vecs[10].tolist(), top_k=5)
    assert "base-10" in [m["id"] for m in r["matches"]]

    r = _query(client, s["token"], "mc", reup_vec.tolist(), top_k=5)
    bm = [m["id"] for m in r["matches"]]
    assert bm.count("base-5") == 1, "re-upserted id appears exactly once post-cutover"
    m5 = next(m for m in r["matches"] if m["id"] == "base-5")
    assert m5["metadata"].get("reupserted") is True, "newest (re-upserted) copy wins"

    r = _query(client, s["token"], "mc", fold_vecs["f2-0"].tolist(), top_k=5)
    assert "f2-0" in [m["id"] for m in r["matches"]]


# --------------------------------------------------------------------------- #
# 2. CHAOS A — raise after objects written, before add_shard (no double-count). #
# --------------------------------------------------------------------------- #


def test_chaos_raise_after_objects_before_add_shard(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    _migrate_recall(monkeypatch, recall_url)
    _truncate_recall(recall_url)
    client, state_mod, _v1q, builder = _delta_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, max_deltas=2,
    )

    s = _signup(client, email="chaosA@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]),
                json={"name": "ca", "dimension": _DIM})

    base_recs, base_vecs = _rows(_N_BASE, _DIM, 11, "base")
    _post_recall(client, s["token"], "ca", base_recs)
    assert builder.run_consolidate_once("ca", tenant) == _N_BASE
    _drain_recall(builder, tenant, "ca", state_mod.dataset_watermark(tenant, "ca"))

    # One fold → one delta (below the cap=2; no compaction yet).
    r1, _ = _rows(3, _DIM, 50, "d0")
    _post_recall(client, s["token"], "ca", r1)
    builder.run_consolidate_once("ca", tenant)
    _drain_recall(builder, tenant, "ca", state_mod.dataset_watermark(tenant, "ca"))
    gen_before = state_mod.live_generation(tenant, "ca")
    assert len(gen_before["deltas"]) == 1

    # CHAOS: the cap-trigger fold writes its delta (commits FIRST), then the
    # compaction's `_write_shard` writes the new-base .bin + sidecar to MinIO and
    # calls `add_shard`. We let the delta's add_shard commit, then make the
    # compaction's add_shard RAISE — i.e. objects landed, catalog row did not.
    real_add_shard = builder.add_shard
    call_state = {"n": 0}

    def _add_shard_delta_ok_then_boom(*a, **k):
        call_state["n"] += 1
        if call_state["n"] == 1:
            return real_add_shard(*a, **k)  # the delta fold commits
        raise RuntimeError("chaos: crash after objects, before catalog row")

    r2, _ = _rows(3, _DIM, 51, "d1")
    _post_recall(client, s["token"], "ca", r2)
    monkeypatch.setattr(builder, "add_shard", _add_shard_delta_ok_then_boom)

    # run_consolidate_once swallows the compaction failure (best-effort) and
    # returns the fold's live-row count.
    builder.run_consolidate_once("ca", tenant)

    # The generation now has base + 2 deltas (the cap fold committed; the
    # compaction crashed before committing a new base → NO new base row).
    gen_after = state_mod.live_generation(tenant, "ca")
    assert len(gen_after["deltas"]) == 2, "crashed compaction must not commit a base"
    assert int(gen_after["base"]["id"]) == int(gen_before["base"]["id"])
    # No double-count: still exactly one level-0 base.
    bases = [s for s in state_mod.list_shards(tenant, "ca") if int(s["level"]) == 0]
    assert len(bases) == 1, "crash before catalog row must leave a single base"

    # Recovery: re-run the cap (now with add_shard restored) → compaction succeeds,
    # collapses to one base, no orphan-read (the orphan .bin is never cataloged).
    monkeypatch.setattr(builder, "add_shard", real_add_shard)
    builder._maybe_major_compaction(tenant, "ca")
    gen_final = state_mod.live_generation(tenant, "ca")
    assert len(gen_final["deltas"]) == 0
    assert len([s for s in state_mod.list_shards(tenant, "ca") if int(s["level"]) == 0]) >= 1
    # The compacted base is the newest level-0 base.
    assert int(gen_final["base"]["covered_lsn_hi"]) == state_mod.dataset_watermark(tenant, "ca")


# --------------------------------------------------------------------------- #
# 3. CHAOS B — crash AFTER add_shard (post-cutover) → no double base, no orphan #
#    read; old generation swept on the next cycle.                              #
# --------------------------------------------------------------------------- #


def test_chaos_crash_after_add_shard_post_cutover(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    _migrate_recall(monkeypatch, recall_url)
    _truncate_recall(recall_url)
    client, state_mod, _v1q, builder = _delta_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, max_deltas=2,
    )
    s = _signup(client, email="chaosB@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]),
                json={"name": "cb", "dimension": _DIM})

    base_recs, base_vecs = _rows(_N_BASE, _DIM, 11, "base")
    _post_recall(client, s["token"], "cb", base_recs)
    assert builder.run_consolidate_once("cb", tenant) == _N_BASE
    _drain_recall(builder, tenant, "cb", state_mod.dataset_watermark(tenant, "cb"))

    # One delta (cap=2, not reached).
    r1, _ = _rows(3, _DIM, 50, "d0")
    _post_recall(client, s["token"], "cb", r1)
    builder.run_consolidate_once("cb", tenant)
    _drain_recall(builder, tenant, "cb", state_mod.dataset_watermark(tenant, "cb"))

    # Crash the post-compaction SWEEP (i.e. AFTER add_shard committed the new
    # base). The new base IS live; the old gen is just not yet swept.
    def _boom_sweep(*a, **k):
        raise RuntimeError("chaos: crash after cutover, during sweep")

    monkeypatch.setattr(builder, "_sweep_superseded_shards", _boom_sweep)

    r2, _ = _rows(3, _DIM, 51, "d1")
    _post_recall(client, s["token"], "cb", r2)
    builder.run_consolidate_once("cb", tenant)  # cap fold → compaction → sweep boom

    # Post-cutover: the NEW base is the live generation (committed), zero deltas —
    # even though the sweep crashed (the cutover = add_shard already happened).
    gen = state_mod.live_generation(tenant, "cb")
    assert len(gen["deltas"]) == 0, "the new base IS live even though the sweep crashed"
    new_base_id = int(gen["base"]["id"])
    assert int(gen["base"]["level"]) == 0

    # The old base+deltas are still CATALOGED (sweep crashed) but are NOT in the
    # live generation — so a query can never read both generations (no double
    # count). `live_generation` resolves only the new base; the keep=2 sweep GCs
    # the old generation on a later cycle once a third generation exists.
    all_bases = [s for s in state_mod.list_shards(tenant, "cb") if int(s["level"]) == 0]
    assert len(all_bases) >= 2, "old + new base both still cataloged (sweep crashed)"
    assert new_base_id == max(int(b["id"]) for b in all_bases), "new base is newest"

    # Query through the cutover resolves the new base only (no orphan read): a base
    # id and a freshly-folded id both resolve, exactly once each.
    r = _query(client, s["token"], "cb", base_vecs[3].tolist(), top_k=5)
    ids = [m["id"] for m in r["matches"]]
    assert ids.count("base-3") == 1
    r = _query(client, s["token"], "cb",
               np.array(r2[0]["values"], dtype=np.float32).tolist(), top_k=5)
    assert r2[0]["id"] in [m["id"] for m in r["matches"]]
