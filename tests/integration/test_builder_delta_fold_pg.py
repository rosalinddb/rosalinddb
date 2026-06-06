"""Integration coverage for PR-B: the delta-tier recall fold on the REAL stack.

Runs against a real `pgvector/pgvector:pg15` recall container + the session MinIO
cold tier (reusing the recall/pg fixtures + helpers from `test_consolidation.py`),
with `RB_DELTA_TIER=true` and `INDEX_TYPE=ivfflat` so the builder takes the
delta-tier path end-to-end.

This asserts the BUILDER ARTIFACTS directly (the query read-union is PR-C and is
deliberately NOT exercised here):

  - first IVF consolidation writes a `level=0` BARE-IVF base + a `quantizer-v1`
    object + `quantizer_version=1` (covered band `[0, N]`);
  - a subsequent fold writes a `level=1` `consolidate-delta` (NOT a base rewrite)
    with the right `parent_shard_id`/`quantizer_version`/`covered_lsn_*`, the base
    `.bin` byte-unchanged, and the delta `.bin` searching back its native int64s;
  - recall trims after the second fold (the grace watermark drains the folded
    band).

The recall-store helpers + the pgvector container fixture are imported from the
existing consolidation suite so this test reuses the exact same harness.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import faiss  # type: ignore

from tests.integration.test_consolidation import (  # noqa: F401 - fixture reuse
    recall_url,
    _migrate_recall,
    _truncate_recall,
    _recall_count,
    _signup,
    _tenant_of,
    _auth,
    _post_recall,
    _build_client,
)


_DIM = 8
_N_BASE = 80  # > IVF_TRAINING_FLOOR (64) so the base trains an IVF


def _delta_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    """`_build_client` with the delta tier ON + IVFFlat cold shards."""
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")
    client, state_mod, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    # `_build_client` forces INDEX_TYPE=flat after setting the env; re-assert IVF
    # on the reloaded builder module so the delta-tier IVF path is taken.
    builder.INDEX_TYPE = "ivfflat"
    return client, state_mod, v1q, builder


def _deterministic_recall_rows(n, dim, seed, prefix):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim), dtype=np.float64).astype(np.float32)
    return [
        {"id": f"{prefix}-{i}", "values": vecs[i].tolist(), "metadata": {"i": i}}
        for i in range(n)
    ], vecs


def _deser(blob: bytes):
    return faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))


def test_delta_fold_writes_delta_not_base_rewrite(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, _v1q, builder = _delta_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path
    )
    import adapters.storage.storage as storage_mod

    s = _signup(client, email="delta-pg@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]),
                json={"name": "dl", "dimension": _DIM})

    # --- first consolidation: a bare-IVF base + quantizer-v1 ----------------
    base_recs, base_vecs = _deterministic_recall_rows(_N_BASE, _DIM, 11, "base")
    _post_recall(client, s["token"], "dl", base_recs)
    n = builder.run_consolidate_once("dl", tenant)
    assert n == _N_BASE

    base = state_mod.get_latest_shard(tenant, "dl")
    assert base["build_type"] == "consolidate"
    assert base["index_type"] == "ivfflat"
    assert int(base["level"]) == 0
    assert int(base["quantizer_version"]) == 1
    assert int(base["covered_lsn_lo"]) == 0
    assert int(base["covered_lsn_hi"]) == _N_BASE

    # The base .bin is a BARE IVF (no IDMap2) — the mergeable shape.
    base_blob = storage_mod.read_bytes(base["shard_uri"])
    base_idx = _deser(base_blob)
    assert not hasattr(base_idx, "id_map")
    assert faiss.try_extract_index_ivf(base_idx) is not None

    # The quantizer object exists, trained-but-empty.
    q_uri = builder._quantizer_uri(tenant, "dl", 1)
    q_idx = _deser(storage_mod.read_bytes(q_uri))
    assert q_idx.is_trained and q_idx.ntotal == 0

    base_bytes_before = bytes(base_blob)

    # Drain the base band from recall. Under PR-B a base+delta is ONE generation,
    # so `grace_watermark` is 0 and the auto-trim leaves the base rows in recall
    # (PR-D's major compaction is what ages a generation out and drains them).
    # We trim the base band manually here so the next fold's snapshot is JUST the
    # new rows — this is exactly the band PR-D will reclaim, made deterministic.
    builder.recall_trim(tenant, "dl", _N_BASE)

    # --- second consolidation: a cheap delta, NOT a base rewrite ------------
    delta_recs, delta_vecs = _deterministic_recall_rows(6, _DIM, 22, "delta")
    _post_recall(client, s["token"], "dl", delta_recs)
    n2 = builder.run_consolidate_once("dl", tenant)
    assert n2 == 6

    gen = state_mod.live_generation(tenant, "dl")
    assert int(gen["base"]["id"]) == int(base["id"]), "base must NOT have been rewritten"
    assert len(gen["deltas"]) == 1
    delta = gen["deltas"][0]
    assert delta["build_type"] == "consolidate-delta"
    assert int(delta["level"]) == 1
    assert int(delta["parent_shard_id"]) == int(base["id"])
    assert int(delta["quantizer_version"]) == 1
    assert int(delta["covered_lsn_lo"]) == _N_BASE + 1
    assert int(delta["covered_lsn_hi"]) == _N_BASE + 6
    assert int(delta["consolidated_lsn"]) == _N_BASE + 6

    # The base .bin is byte-unchanged.
    assert storage_mod.read_bytes(base["shard_uri"]) == base_bytes_before

    # The delta .bin searches and returns the native int64 ids.
    didx = _deser(storage_mod.read_bytes(delta["shard_uri"]))
    assert not hasattr(didx, "id_map")
    didx.nprobe = didx.nlist
    target = builder._id_to_int64("delta-3")
    _, found = didx.search(np.array([delta_vecs[3]], dtype=np.float32), 1)
    assert int(found.ravel()[0]) == target


def test_delta_fold_grace_is_generation_aware(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Recall trim follows the GENERATION grace, not list position (I4).

    A base + its deltas are ONE generation, so until PR-D's major compaction
    produces a SECOND generation `grace_watermark` is 0 → recall is NOT trimmed
    (matching the legacy first-consolidation behaviour). This proves the fold
    keeps the watermark/grace machinery generation-correct under the flag: the
    recall rows the deltas fold remain available (their cold copies live in the
    deltas) and are NOT prematurely dropped by a list-position `[keep:]` trim that
    would otherwise have aged the base out from under its live deltas.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, _v1q, builder = _delta_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path
    )
    s = _signup(client, email="delta-trim@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]),
                json={"name": "dt", "dimension": _DIM})

    base_recs, _ = _deterministic_recall_rows(_N_BASE, _DIM, 11, "base")
    _post_recall(client, s["token"], "dt", base_recs)
    assert builder.run_consolidate_once("dt", tenant) == _N_BASE
    # First consolidation: one generation → grace 0 → nothing trimmed.
    assert _recall_count(recall_url, tenant, "dt") == _N_BASE
    assert state_mod.grace_watermark(tenant, "dt", keep=2) == 0

    extra_recs, _ = _deterministic_recall_rows(5, _DIM, 22, "x")
    _post_recall(client, s["token"], "dt", extra_recs)
    n2 = builder.run_consolidate_once("dt", tenant)
    # Grace 0 means the base band is NOT trimmed, so the snapshot STILL contains
    # the already-folded base rows (lsn <= the base frontier) alongside the 5 new
    # ones. A delta fold must write ONLY the rows above the frontier — the base +
    # prior deltas already cover the rest — so the fold processes exactly the 5
    # new rows (O(new), not O(recall)). (A frontier filter, not a re-fold of the
    # whole partition — the bench-found minor-compaction invariant.)
    assert n2 == 5

    # STILL one generation (the new shard is a delta on the same base), so grace
    # stays 0 and the live base is never swept out from under its delta. The
    # watermark advances to the fold's max LSN; deltas drain to one base only at
    # PR-D's major compaction.
    gen = state_mod.live_generation(tenant, "dt")
    assert len(gen["deltas"]) == 1
    # The delta holds ONLY the 5 new rows (the cheap O(new) fold), not a re-fold
    # of the whole live partition, and its band is contiguous (not degenerate).
    delta = gen["deltas"][0]
    assert int(delta["vector_count"]) == 5
    assert int(delta["covered_lsn_lo"]) <= int(delta["covered_lsn_hi"])
    assert state_mod.grace_watermark(tenant, "dt", keep=2) == 0
    assert state_mod.dataset_watermark(tenant, "dt") == _N_BASE + 5
    # The base + its delta both survive the sweep (no live shard GC'd) — the P0
    # regression the generation-aware sweep prevents.
    assert state_mod.superseded_shards(tenant, "dt", keep=2) == []
    assert state_mod.get_latest_shard(tenant, "dt") is not None
