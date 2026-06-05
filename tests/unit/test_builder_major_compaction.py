"""Unit tests for PR-D: the index builder's MAJOR COMPACTION at the delta cap.

When a live generation accumulates `RB_MAX_DELTAS` (default 8) deltas a major
compaction folds the base + all its deltas into ONE new `level=0` bare-IVF base
under the SAME frozen quantizer (no retrain), bounding read fan-out. All behind
`RB_DELTA_TIER` (default OFF) — flag-off byte-identical to today.

Covered here (memory:// only):

  - cap trigger fires at RB_MAX_DELTAS: N folds → one base + N deltas, the
    (N+1)th drives compaction → one level-0 base, deltas gone from the live gen;
  - dedup re-upsert: newest vector wins, single copy (no duplicate native ids);
  - tombstone dropped: a tombstoned-as-final id is absent from the merged base;
  - delete-then-reinsert kept: tombstoned in an early delta, re-inserted live in
    a later delta → present with the reinsert's vector;
  - ids round-trip (search/reconstruct) after compaction;
  - watermark = max old covered_lsn_hi (no regression);
  - recall@k parity: top-k of the pre-compaction base+delta union == top-k of the
    post-compaction single base;
  - flag-off unchanged: no compaction ever runs, deltas not produced;
  - hard ceiling backstop forces a compaction past RB_MAX_DELTAS_HARD.

The recall snapshot is monkeypatched (as in test_builder_delta_fold.py) so each
fold's rows are deterministic; the major compaction reads only the committed
shards + quantizer, never recall.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import faiss  # type: ignore


_DIM = 16
_N_COLD = 96  # > IVF_TRAINING_FLOOR (64) so the base trains an IVF


def _reload_stack(monkeypatch, *, delta_on: bool, max_deltas: int = 4,
                  max_deltas_hard: int = 16):
    """Reload builder/state/storage bound to memory://, IVFFlat, recall ON.

    `max_deltas` is set small (4) so the cap fires after a handful of folds
    without seeding hundreds of rows.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://dummy/recall")
    monkeypatch.setenv("RB_MAX_DELTAS", str(max_deltas))
    monkeypatch.setenv("RB_MAX_DELTAS_HARD", str(max_deltas_hard))
    if delta_on:
        monkeypatch.setenv("RB_DELTA_TIER", "true")
    else:
        monkeypatch.delenv("RB_DELTA_TIER", raising=False)

    import adapters.storage.storage as storage_mod
    importlib.reload(storage_mod)
    storage_mod._MEM_OBJECTS.clear()

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0
    state_mod._MEM_DATASETS.clear()

    import services.index_builder.run as builder_mod
    importlib.reload(builder_mod)
    return builder_mod, state_mod, storage_mod


@pytest.fixture
def delta_on(monkeypatch):
    return _reload_stack(monkeypatch, delta_on=True)


@pytest.fixture
def delta_off(monkeypatch):
    return _reload_stack(monkeypatch, delta_on=False)


def _vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float64).astype(np.float32)


def _recall_row(rid, values, lsn, deleted=False, metadata=None):
    return {
        "id": rid,
        "values": values,
        "metadata": metadata or {"src": "recall"},
        "lsn": lsn,
        "deleted": deleted,
    }


def _patch_snapshot(monkeypatch, builder_mod, max_lsn, rows):
    monkeypatch.setattr(
        builder_mod, "recall_snapshot_for_consolidation", lambda t, d: (max_lsn, rows)
    )
    monkeypatch.setattr(builder_mod, "recall_trim", lambda t, d, g: 0)


def _deser(blob: bytes):
    return faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))


def _seed_base(builder_mod, state_mod, monkeypatch, ids, vectors, *, max_lsn,
               tenant="t1", dataset="ds"):
    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, _DIM)
    rows = [
        _recall_row(rid, vectors[i].tolist(), i + 1, metadata={"v": 0})
        for i, rid in enumerate(ids)
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn, rows)
    n = builder_mod.run_consolidate_once(dataset, tenant)
    assert n == len(ids)
    return state_mod.get_latest_shard(tenant, dataset)


def _fold(builder_mod, monkeypatch, rows, max_lsn, tenant="t1", dataset="ds"):
    _patch_snapshot(monkeypatch, builder_mod, max_lsn, rows)
    return builder_mod.run_consolidate_once(dataset, tenant)


# --------------------------------------------------------------------------- #
# 1. Cap trigger: folds past RB_MAX_DELTAS → one base, deltas gone.           #
# --------------------------------------------------------------------------- #


def test_cap_trigger_folds_to_one_base(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    max_deltas = builder_mod._max_deltas()
    assert max_deltas == 4

    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)
    old_base_id = int(base["id"])

    # Drive max_deltas-1 folds (no compaction yet): the generation reaches
    # max_deltas-1 deltas; on the FINAL allowed fold the count hits the cap.
    # Each fold's rows sit at the TOP of a fresh LSN band (strictly ABOVE the
    # prior generation frontier) with max_lsn == the band max — mirroring a real
    # recall snapshot, where a fold writes only rows with lsn > the frontier.
    lsn = 200
    for k in range(max_deltas - 1):
        lsn += 50
        new = _vectors(3, _DIM, seed=100 + k)
        rows = [_recall_row(f"d{k}-{j}", new[j].tolist(), lsn - 2 + j) for j in range(3)]
        assert _fold(builder_mod, monkeypatch, rows, max_lsn=lsn) == 3
        gen = state_mod.live_generation("t1", "ds")
        assert len(gen["deltas"]) == k + 1, "no compaction before the cap"
        assert int(gen["base"]["id"]) == old_base_id

    # The fold that brings the live-delta count TO the cap triggers compaction.
    lsn += 50
    new = _vectors(3, _DIM, seed=999)
    rows = [_recall_row(f"cap-{j}", new[j].tolist(), lsn - 2 + j) for j in range(3)]
    assert _fold(builder_mod, monkeypatch, rows, max_lsn=lsn) == 3

    gen = state_mod.live_generation("t1", "ds")
    # After compaction: one fresh level-0 base, ZERO deltas.
    assert len(gen["deltas"]) == 0, "compaction must collapse deltas to zero"
    new_base = gen["base"]
    assert int(new_base["id"]) != old_base_id, "compaction must produce a NEW base"
    assert int(new_base["level"]) == 0
    assert new_base["build_type"] == "consolidate"
    assert new_base["parent_shard_id"] in (None, 0)
    assert int(new_base["quantizer_version"]) == 1  # same frozen quantizer (no retrain)
    assert int(new_base["covered_lsn_lo"]) == 0

    # Watermark = max covered_lsn_hi over the old generation (no regression).
    assert int(new_base["covered_lsn_hi"]) == lsn
    assert int(new_base["consolidated_lsn"]) == lsn

    # New base is a BARE IVF (mergeable shape) and searches back its native ids.
    idx = _deser(storage_mod.read_bytes(new_base["shard_uri"]))
    assert not hasattr(idx, "id_map")
    idx.set_direct_map_type(faiss.DirectMap.Hashtable)
    idx.nprobe = idx.nlist
    target = builder_mod._id_to_int64("cap-1")
    _, found = idx.search(np.array([new[1]], dtype=np.float32), 1)
    assert int(found.ravel()[0]) == target
    # An original base id also still round-trips.
    target0 = builder_mod._id_to_int64("id-7")
    _, found0 = idx.search(np.array([vecs[7]], dtype=np.float32), 1)
    assert int(found0.ravel()[0]) == target0


# --------------------------------------------------------------------------- #
# 2. Dedup re-upsert: newest vector wins, single copy.                        #
# --------------------------------------------------------------------------- #


def _occurrences(idx, int_id: int) -> int:
    inv = idx.invlists
    cnt = 0
    for lst in range(int(idx.nlist)):
        sz = int(inv.list_size(lst))
        if sz:
            arr = np.asarray(faiss.rev_swig_ptr(inv.get_ids(lst), sz))
            cnt += int(np.sum(arr == int_id))
    return cnt


def test_major_compaction_dedup_reupsert_newest_wins(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    # Fold 1 re-upserts id-3 with a NEW vector v1.
    v1 = _vectors(1, _DIM, seed=71)[0]
    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("id-3", v1.tolist(), 200)], max_lsn=210) == 1
    # Fold 2 re-upserts id-3 AGAIN with a NEWER vector v2 (this should win).
    v2 = _vectors(1, _DIM, seed=72)[0]
    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("id-3", v2.tolist(), 220)], max_lsn=230) == 1

    new_uri = builder_mod._major_compaction("t1", "ds")
    assert new_uri is not None
    gen = state_mod.live_generation("t1", "ds")
    assert len(gen["deltas"]) == 0
    idx = _deser(storage_mod.read_bytes(gen["base"]["shard_uri"]))
    idx.set_direct_map_type(faiss.DirectMap.Hashtable)

    int_id3 = builder_mod._id_to_int64("id-3")
    # Exactly ONE copy of id-3 (no duplicate native ids).
    assert _occurrences(idx, int_id3) == 1
    # And it is the NEWEST vector (v2), not v1 or the base's original.
    rec = idx.reconstruct(int(int_id3))
    assert np.allclose(rec, v2, atol=1e-5), "newest re-upsert vector must win"
    assert not np.allclose(rec, v1, atol=1e-5)
    assert not np.allclose(rec, vecs[3], atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Tombstone dropped.                                                        #
# --------------------------------------------------------------------------- #


def test_major_compaction_drops_tombstoned_id(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    # A fold that deletes a cold id (id-9) + adds a live row.
    live = _vectors(1, _DIM, seed=81)[0]
    rows = [
        _recall_row("live-a", live.tolist(), 200),
        _recall_row("id-9", vecs[9].tolist(), 201, deleted=True),
    ]
    assert _fold(builder_mod, monkeypatch, rows, max_lsn=210) == 1

    new_uri = builder_mod._major_compaction("t1", "ds")
    assert new_uri is not None
    gen = state_mod.live_generation("t1", "ds")
    idx = _deser(storage_mod.read_bytes(gen["base"]["shard_uri"]))
    idx.set_direct_map_type(faiss.DirectMap.Hashtable)

    int_id9 = builder_mod._id_to_int64("id-9")
    assert _occurrences(idx, int_id9) == 0, "tombstoned id must be physically purged"
    with pytest.raises(Exception):
        idx.reconstruct(int(int_id9))
    # The surviving live row and other base ids remain.
    assert _occurrences(idx, builder_mod._id_to_int64("live-a")) == 1
    assert _occurrences(idx, builder_mod._id_to_int64("id-0")) == 1
    # ntotal = 96 base - 1 deleted + 1 new live = 96.
    assert int(idx.ntotal) == _N_COLD


# --------------------------------------------------------------------------- #
# 4. Delete-then-reinsert kept.                                                #
# --------------------------------------------------------------------------- #


def test_major_compaction_delete_then_reinsert_kept(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    # Fold 1: tombstone id-4 (in the base).
    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("id-4", vecs[4].tolist(), 200, deleted=True)],
                 max_lsn=210) == 0
    # Fold 2: re-insert id-4 LIVE with a fresh vector vr (delete-then-reinsert).
    vr = _vectors(1, _DIM, seed=91)[0]
    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("id-4", vr.tolist(), 220)], max_lsn=230) == 1

    new_uri = builder_mod._major_compaction("t1", "ds")
    assert new_uri is not None
    gen = state_mod.live_generation("t1", "ds")
    idx = _deser(storage_mod.read_bytes(gen["base"]["shard_uri"]))
    idx.set_direct_map_type(faiss.DirectMap.Hashtable)

    int_id4 = builder_mod._id_to_int64("id-4")
    assert _occurrences(idx, int_id4) == 1, "reinserted id must be present exactly once"
    rec = idx.reconstruct(int(int_id4))
    assert np.allclose(rec, vr, atol=1e-5), "the REINSERT vector must win over base"
    assert not np.allclose(rec, vecs[4], atol=1e-5)


# --------------------------------------------------------------------------- #
# 5. Watermark = max old covered_lsn_hi (no regression).                       #
# --------------------------------------------------------------------------- #


def test_major_compaction_watermark_no_regression(delta_on, monkeypatch):
    builder_mod, state_mod, _ = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("a", _vectors(1, _DIM, 9)[0].tolist(), 300)],
                 max_lsn=305) == 1
    assert _fold(builder_mod, monkeypatch,
                 [_recall_row("b", _vectors(1, _DIM, 10)[0].tolist(), 400)],
                 max_lsn=412) == 1
    gen = state_mod.live_generation("t1", "ds")
    max_hi = max(int(s.get("covered_lsn_hi", 0) or 0)
                 for s in [gen["base"], *gen["deltas"]])
    assert max_hi == 412

    builder_mod._major_compaction("t1", "ds")
    new_base = state_mod.live_generation("t1", "ds")["base"]
    assert int(new_base["covered_lsn_hi"]) == 412
    assert int(new_base["consolidated_lsn"]) == 412
    assert int(new_base["covered_lsn_lo"]) == 0
    # The dataset watermark did not regress.
    assert state_mod.dataset_watermark("t1", "ds") == 412


# --------------------------------------------------------------------------- #
# 6. recall@k parity: union before == single base after.                       #
# --------------------------------------------------------------------------- #


def _search_union(shards, storage_mod, query, k):
    """Top-k over a base+delta union: search each shard, merge by exact L2."""
    hits = []
    for s in shards:
        idx = _deser(storage_mod.read_bytes(s["shard_uri"]))
        idx.set_direct_map_type(faiss.DirectMap.Hashtable)
        idx.nprobe = idx.nlist
        D, I = idx.search(query, k)
        for d, i in zip(D.ravel().tolist(), I.ravel().tolist()):
            if i != -1:
                hits.append((d, i))
    hits.sort(key=lambda x: x[0])
    # dedup by id keeping the smallest distance (newest-vector copies share id)
    seen = {}
    for d, i in hits:
        if i not in seen:
            seen[i] = d
    out = sorted(seen.items(), key=lambda x: x[1])[:k]
    return [i for i, _ in out]


def test_major_compaction_recall_parity(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    # Three folds of distinct new rows (no overlap) so the union is base + 3.
    fold_vecs = {}
    lsn = 200
    for k in range(3):
        lsn += 50  # rows at the top of a fresh band, strictly above the frontier
        fv = _vectors(4, _DIM, seed=300 + k)
        rows = [_recall_row(f"f{k}-{j}", fv[j].tolist(), lsn - 3 + j) for j in range(4)]
        for j in range(4):
            fold_vecs[f"f{k}-{j}"] = fv[j]
        assert _fold(builder_mod, monkeypatch, rows, max_lsn=lsn) == 4

    gen_before = state_mod.live_generation("t1", "ds")
    shards_before = [gen_before["base"], *gen_before["deltas"]]
    assert len(gen_before["deltas"]) == 3

    # Query vectors: a base id, and one from each fold.
    queries = np.array([vecs[10], fold_vecs["f0-1"], fold_vecs["f2-3"]],
                       dtype=np.float32)
    k = 5
    before = [_search_union(shards_before, storage_mod, queries[q:q + 1], k)
              for q in range(queries.shape[0])]

    builder_mod._major_compaction("t1", "ds")
    gen_after = state_mod.live_generation("t1", "ds")
    assert len(gen_after["deltas"]) == 0
    after = [_search_union([gen_after["base"]], storage_mod, queries[q:q + 1], k)
             for q in range(queries.shape[0])]

    # Same top-k ids (and the exact-distance ranking under nprobe=nlist) before
    # and after — the shared frozen quantizer makes the union lossless vs the
    # single merged base (P0-C).
    assert before == after


# --------------------------------------------------------------------------- #
# 7. Flag OFF: no compaction ever runs.                                        #
# --------------------------------------------------------------------------- #


def test_flag_off_never_compacts(delta_off, monkeypatch):
    builder_mod, state_mod, _ = delta_off
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    # Many folds (well past the cap) — flag off keeps the legacy base-rewrite
    # fold; there are no deltas and `_maybe_major_compaction` is a no-op.
    lsn = 200
    for k in range(6):
        new = _vectors(3, _DIM, seed=400 + k)
        rows = [_recall_row(f"n{k}-{j}", new[j].tolist(), lsn + j) for j in range(3)]
        lsn += 50
        builder_mod.run_consolidate_once("ds", "t1")
    gen = state_mod.live_generation("t1", "ds")
    # Every flag-off fold writes a level-0 base (union rewrite), never a delta.
    assert len(gen["deltas"]) == 0
    assert int(gen["base"]["level"]) == 0
    assert gen["base"]["build_type"] == "consolidate"
    # `_maybe_major_compaction` short-circuits with the flag off.
    builder_mod._maybe_major_compaction("t1", "ds")  # must not raise / change anything
    assert len(state_mod.live_generation("t1", "ds")["deltas"]) == 0


# --------------------------------------------------------------------------- #
# 8. Hard ceiling backstop forces a compaction past RB_MAX_DELTAS_HARD.        #
# --------------------------------------------------------------------------- #


def test_hard_ceiling_backstop_forces_compaction(monkeypatch):
    # Trigger high (never reached by the per-fold cap) but hard ceiling low (3),
    # so the cap-at-trigger path is disabled and ONLY the backstop can fire.
    builder_mod, state_mod, storage_mod = _reload_stack(
        monkeypatch, delta_on=True, max_deltas=999, max_deltas_hard=3
    )
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)

    lsn = 200
    # First two folds: 1 then 2 deltas (< hard ceiling 3) — no compaction.
    # Rows at the top of a fresh band, strictly above the prior frontier.
    for k in range(2):
        lsn += 50
        new = _vectors(2, _DIM, seed=500 + k)
        rows = [_recall_row(f"h{k}-{j}", new[j].tolist(), lsn - 1 + j) for j in range(2)]
        assert _fold(builder_mod, monkeypatch, rows, max_lsn=lsn) == 2
        assert len(state_mod.live_generation("t1", "ds")["deltas"]) == k + 1

    # Third fold brings the live-delta count to 3 == hard ceiling → backstop
    # forces a major compaction even though the trigger (999) is not reached.
    lsn += 50
    new = _vectors(2, _DIM, seed=599)
    rows = [_recall_row(f"h2-{j}", new[j].tolist(), lsn - 1 + j) for j in range(2)]
    assert _fold(builder_mod, monkeypatch, rows, max_lsn=lsn) == 2
    gen = state_mod.live_generation("t1", "ds")
    assert len(gen["deltas"]) == 0, "hard-ceiling backstop must force a compaction"
    assert int(gen["base"]["level"]) == 0
    assert int(gen["base"]["covered_lsn_hi"]) == lsn


# --------------------------------------------------------------------------- #
# 9. Regression (bench-found): a fold writes ONLY rows above the frontier.     #
# --------------------------------------------------------------------------- #


def test_fold_skips_already_folded_rows_when_recall_untrimmed(delta_on, monkeypatch):
    """A fold must write only NEW rows (lsn > generation frontier) into the delta.

    Bench-found bug: until the first MAJOR compaction creates a 2nd generation,
    `grace_watermark` is 0 so `recall_trim` is a no-op, so the recall snapshot
    keeps returning rows that were ALREADY folded into the base/prior deltas.
    Without a frontier filter every fold re-folds the whole snapshot — the delta
    bloats to O(recall) and stamps a degenerate `[frontier+1, N]` band over rows
    with lsn <= frontier. This asserts the fold folds ONLY the band above the
    frontier even when the snapshot still contains the already-folded base rows.
    """
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_base(builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD)
    # The base rows (lsn 1.._N_COLD) — still present in recall (untrimmed).
    base_rows = [_recall_row(rid, vecs[i].tolist(), i + 1) for i, rid in enumerate(ids)]

    # 5 brand-new rows above the frontier; the UNTRIMMED snapshot returns the base
    # rows AGAIN plus the new ones (max_lsn = the new max).
    nv = _vectors(5, _DIM, seed=77)
    new_rows = [_recall_row(f"new-{j}", nv[j].tolist(), _N_COLD + 1 + j) for j in range(5)]
    n = _fold(builder_mod, monkeypatch, base_rows + new_rows, max_lsn=_N_COLD + 5)

    assert n == 5, "fold must process only the NEW rows above the frontier, not re-fold the base"
    gen = state_mod.live_generation("t1", "ds")
    assert len(gen["deltas"]) == 1
    delta = gen["deltas"][0]
    assert int(delta["vector_count"]) == 5, "delta must hold ONLY the 5 new rows (not re-fold the base)"
    assert int(delta["covered_lsn_lo"]) == _N_COLD + 1
    assert int(delta["covered_lsn_hi"]) == _N_COLD + 5
    assert int(delta["covered_lsn_lo"]) <= int(delta["covered_lsn_hi"]), "band must not be degenerate"
    didx = _deser(storage_mod.read_bytes(delta["shard_uri"]))
    assert int(didx.ntotal) == 5
