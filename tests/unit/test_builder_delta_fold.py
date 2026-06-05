"""Unit tests for PR-B: the index builder's cheap delta-shard recall fold.

The delta tier is gated behind `RB_DELTA_TIER` (default OFF). These tests cover:

  - FLAG OFF: every consolidate path is byte-identical to today (the existing
    consolidate suite already proves the unchanged behaviour; here we assert the
    flag-off branch still writes a single `build_type='consolidate'` base shard
    and writes NO quantizer object).
  - FLAG ON, first build (no prior shard): a `level=0` BARE-IVF base
    (`build_ivfflat_native`, native ids in the inverted lists, NO IndexIDMap2) +
    a `quantizer-v1.index` object + `quantizer_version=1`, `covered_lsn_lo=0`,
    `covered_lsn_hi=N`, `build_type='consolidate'`.
  - FLAG ON, subsequent fold: a `level=1` `build_type='consolidate-delta'` shard
    with `parent_shard_id=base.id`, `quantizer_version=G`, `covered_lsn_lo` =
    (max covered_lsn_hi of the generation)+1, `covered_lsn_hi=N`; the base `.bin`
    is BYTE-UNCHANGED after the fold; the delta `.bin` searches and returns the
    native int64 ids.
  - `_build_delta_blob`: clone-not-retrain (adding to a clone leaves the
    quantizer/centroids untouched), and the blob searches back its native ids.
  - Tombstone fold with a prior base: the delta carries `tombstone_int_ids` for
    EVERY deleted id (no cold-membership probe) and advances the watermark.
  - Hardening: a delete is recorded even when the shard sidecar is unreadable
    (the fold no longer depends on sidecar reads for tombstone detection).
  - Tombstone-only fold with NO prior shard: returns None (unchanged).
  - Sub-IVF-floor first build under the flag: falls back to flat, level=0, no
    quantizer object.

memory:// only. The QUERY read-path is PR-C and deliberately untested here — we
assert the BUILDER ARTIFACTS directly (catalog columns + the .bin bytes).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

import faiss  # type: ignore


_DIM = 16
_N_COLD = 96  # > IVF_TRAINING_FLOOR (64) so the base trains an IVF


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _reload_stack(monkeypatch, *, delta_on: bool):
    """Reload builder/state/storage bound to memory://, IVFFlat, recall ON."""
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://dummy/recall")
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


def _seed_cold_via_consolidate(builder_mod, state_mod, monkeypatch, ids, vectors,
                               *, max_lsn, tenant="t1", dataset="ds"):
    """Build the FIRST cold shard through the consolidate path (so the delta-tier
    flag decides base shape). Returns the committed base shard row."""
    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, _DIM)
    rows = [
        {"id": rid, "values": vectors[i].tolist(),
         "metadata": {"src": "cold", "v": 0}, "lsn": i + 1, "deleted": False}
        for i, rid in enumerate(ids)
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn, rows)
    n = builder_mod.run_consolidate_once(dataset, tenant)
    assert n == len(ids)
    shard = state_mod.get_latest_shard(tenant, dataset)
    assert shard is not None
    return shard


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


# --------------------------------------------------------------------------- #
# 0. The flag helper.                                                          #
# --------------------------------------------------------------------------- #


def test_delta_tier_flag_parsing(monkeypatch):
    builder_mod, _, _ = _reload_stack(monkeypatch, delta_on=False)
    assert builder_mod._delta_tier_enabled() is False
    for truthy in ("1", "true", "yes", "on", "TRUE", "On"):
        monkeypatch.setenv("RB_DELTA_TIER", truthy)
        assert builder_mod._delta_tier_enabled() is True, truthy
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("RB_DELTA_TIER", falsy)
        assert builder_mod._delta_tier_enabled() is False, falsy


# --------------------------------------------------------------------------- #
# 1. _build_delta_blob: clone-not-retrain + native id search.                  #
# --------------------------------------------------------------------------- #


def test_build_delta_blob_clone_not_retrain(delta_on):
    builder_mod, _, _ = delta_on
    dim, nlist = _DIM, 8
    rng = np.random.default_rng(3)
    quantizer = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
    quantizer.train(rng.random((400, dim)).astype(np.float32))
    quantizer.reset()  # trained-but-empty

    def centroids(ivf):
        qz = faiss.downcast_index(ivf.quantizer)
        return faiss.vector_to_array(qz.codes).view(np.float32).reshape(-1, dim).copy()

    before = centroids(quantizer)

    ids = np.array([10**12 + i for i in range(30)], dtype=np.int64)
    vecs = rng.random((30, dim)).astype(np.float32)
    metas = [{"i": int(i)} for i in range(30)]

    blob, sidecar_blob, index_type_str, n = builder_mod._build_delta_blob(
        quantizer, ids, vecs, metas
    )
    # The source quantizer is untouched (no retrain, no add).
    assert quantizer.ntotal == 0
    assert np.array_equal(before, centroids(quantizer))
    assert index_type_str == "ivfflat"
    assert n == 30

    # The blob is a bare IVF (no IDMap2) that searches back the NATIVE ids.
    idx = _deser(blob)
    assert not hasattr(idx, "id_map"), "delta blob must be a BARE IVF, not IDMap2"
    assert faiss.try_extract_index_ivf(idx) is not None
    idx.nprobe = nlist
    _, found = idx.search(vecs[:5], 1)
    assert found.ravel().tolist() == ids[:5].tolist()

    # Sidecar maps str(int64) -> {id, metadata} for every row.
    sidecar = json.loads(sidecar_blob.decode("utf-8"))
    assert len(sidecar) == 30
    key0 = str(int(ids[0]))
    assert key0 in sidecar
    assert "id" in sidecar[key0] and "metadata" in sidecar[key0]


def test_build_ivfflat_native_is_bare_ivf(delta_on):
    builder_mod, _, _ = delta_on
    vectors = _vectors(_N_COLD, _DIM, seed=1)
    ids = np.array([builder_mod._id_to_int64(f"id-{i}") for i in range(_N_COLD)],
                   dtype=np.int64)
    blob = builder_mod.build_ivfflat_native(vectors, ids)
    idx = _deser(blob)
    assert not hasattr(idx, "id_map"), "native base must NOT be IndexIDMap2-wrapped"
    assert faiss.try_extract_index_ivf(idx) is not None
    assert idx.ntotal == _N_COLD
    idx.nprobe = idx.nlist
    _, found = idx.search(vectors[:4], 1)
    assert found.ravel().tolist() == ids[:4].tolist()


# --------------------------------------------------------------------------- #
# 2. FLAG OFF: consolidate path unchanged (no quantizer object, IDMap2 base).  #
# --------------------------------------------------------------------------- #


def test_flag_off_first_build_is_legacy_idmap2(delta_off, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_off
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    shard = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    assert shard["build_type"] == "consolidate"
    assert shard["index_type"] == "ivfflat"
    # Legacy defaults: level 0, no quantizer version, no parent.
    assert int(shard.get("level", 0) or 0) == 0
    assert int(shard.get("quantizer_version", 0) or 0) == 0
    assert shard.get("parent_shard_id") in (None, 0)
    # Legacy base is an IDMap2-wrapped IVF.
    idx = _deser(storage_mod.read_bytes(shard["shard_uri"]))
    assert hasattr(idx, "id_map"), "flag-off base must keep the legacy IDMap2 shape"
    # No quantizer object was written.
    q_uri = builder_mod._quantizer_uri("t1", "ds", 1)
    assert q_uri not in storage_mod._MEM_OBJECTS


def test_flag_off_subsequent_fold_writes_consolidate_base(delta_off, monkeypatch):
    builder_mod, state_mod, _ = delta_off
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    # A second fold (append-only) still produces a consolidate base (level 0).
    new = _vectors(10, _DIM, seed=6)
    rows = [_recall_row(f"new-{k}", new[k].tolist(), 200 + k) for k in range(10)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=300, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 10
    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["build_type"] == "consolidate"
    assert int(shard.get("level", 0) or 0) == 0
    assert int(shard.get("consolidated_lsn", 0)) == 300


# --------------------------------------------------------------------------- #
# 3. FLAG ON, first build: bare-IVF base + quantizer-v1 object.                #
# --------------------------------------------------------------------------- #


def test_flag_on_first_build_writes_bare_base_and_quantizer(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    shard = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    assert shard["build_type"] == "consolidate"
    assert shard["index_type"] == "ivfflat"
    assert int(shard["level"]) == 0
    assert int(shard["quantizer_version"]) == 1
    assert int(shard["covered_lsn_lo"]) == 0
    assert int(shard["covered_lsn_hi"]) == _N_COLD
    assert int(shard["consolidated_lsn"]) == _N_COLD

    # The base .bin is a BARE IVF (native ids) — the mergeable shape PR-D needs.
    idx = _deser(storage_mod.read_bytes(shard["shard_uri"]))
    assert not hasattr(idx, "id_map"), "flag-on base must be a bare IVF (native ids)"
    assert faiss.try_extract_index_ivf(idx) is not None
    idx.nprobe = idx.nlist
    target_int = builder_mod._id_to_int64("id-7")
    qv = np.array([vecs[7]], dtype=np.float32)
    _, found = idx.search(qv, 1)
    assert int(found.ravel()[0]) == target_int

    # The quantizer-v1 object exists and is a trained-but-empty IVF.
    q_uri = builder_mod._quantizer_uri("t1", "ds", 1)
    assert q_uri in storage_mod._MEM_OBJECTS
    q_idx = _deser(storage_mod.read_bytes(q_uri))
    assert q_idx.is_trained
    assert q_idx.ntotal == 0
    assert faiss.try_extract_index_ivf(q_idx) is not None


# --------------------------------------------------------------------------- #
# 4. FLAG ON, subsequent fold: a consolidate-delta layered on the base.        #
# --------------------------------------------------------------------------- #


def test_flag_on_fold_writes_delta_base_unchanged(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    base_bytes_before = bytes(storage_mod.read_bytes(base["shard_uri"]))

    # FOLD: brand-new ids (no overlap), lsn band 100..104.
    new = _vectors(5, _DIM, seed=9)
    new_ids = [f"new-{k}" for k in range(5)]
    rows = [_recall_row(new_ids[k], new[k].tolist(), 100 + k, metadata={"v": k})
            for k in range(5)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=104, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 5

    gen = state_mod.live_generation("t1", "ds")
    assert gen is not None
    assert gen["base"]["id"] == base["id"]
    assert len(gen["deltas"]) == 1
    delta = gen["deltas"][0]

    assert delta["build_type"] == "consolidate-delta"
    assert int(delta["level"]) == 1
    assert int(delta["parent_shard_id"]) == int(base["id"])
    assert int(delta["quantizer_version"]) == 1
    assert int(delta["covered_lsn_lo"]) == _N_COLD + 1  # base.hi + 1
    assert int(delta["covered_lsn_hi"]) == 104
    assert int(delta["consolidated_lsn"]) == 104

    # The base .bin is BYTE-UNCHANGED after the fold (no rewrite, no retrain).
    assert storage_mod.read_bytes(base["shard_uri"]) == base_bytes_before

    # The delta .bin searches and returns the NATIVE int64 ids.
    didx = _deser(storage_mod.read_bytes(delta["shard_uri"]))
    assert not hasattr(didx, "id_map"), "delta must be a bare IVF (native ids)"
    didx.nprobe = didx.nlist
    target_int = builder_mod._id_to_int64("new-2")
    _, found = didx.search(np.array([new[2]], dtype=np.float32), 1)
    assert int(found.ravel()[0]) == target_int

    # The delta shares the base's quantizer version — no new quantizer object.
    assert builder_mod._quantizer_uri("t1", "ds", 2) not in storage_mod._MEM_OBJECTS


def test_flag_on_two_folds_lsn_bands_contiguous(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    # FOLD 1: band (96, 110].
    r1 = [_recall_row(f"a-{k}", _vectors(3, _DIM, 9)[k].tolist(), 100 + k)
          for k in range(3)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=110, rows=r1)
    assert builder_mod.run_consolidate_once("ds", "t1") == 3
    # FOLD 2: band (110, 120].
    r2 = [_recall_row(f"b-{k}", _vectors(3, _DIM, 12)[k].tolist(), 115 + k)
          for k in range(3)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=120, rows=r2)
    assert builder_mod.run_consolidate_once("ds", "t1") == 3

    gen = state_mod.live_generation("t1", "ds")
    assert int(gen["base"]["id"]) == int(base["id"])
    assert len(gen["deltas"]) == 2
    d1, d2 = gen["deltas"]  # ordered by covered_lsn_lo
    assert (int(d1["covered_lsn_lo"]), int(d1["covered_lsn_hi"])) == (_N_COLD + 1, 110)
    assert (int(d2["covered_lsn_lo"]), int(d2["covered_lsn_hi"])) == (111, 120)
    # Contiguous frontier from the base.
    assert int(d2["covered_lsn_lo"]) == int(d1["covered_lsn_hi"]) + 1


# --------------------------------------------------------------------------- #
# 5. Tombstone folds.                                                          #
# --------------------------------------------------------------------------- #


def test_flag_on_tombstone_fold_carries_int_ids_and_advances_watermark(
    delta_on, monkeypatch
):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    base_bytes_before = bytes(storage_mod.read_bytes(base["shard_uri"]))

    # FOLD: one live new id + a tombstone for a cold id (id-3, in the base)
    # AND a tombstone for an id never in cold (ghost). EVERY deleted id is now
    # carried UNCONDITIONALLY — no cold-membership probe. The ghost tombstone is
    # harmless (suppresses/purges nothing at query time) but MUST NOT be dropped.
    live = _vectors(1, _DIM, seed=8)
    rows = [
        _recall_row("live-1", live[0].tolist(), 130),
        _recall_row("id-3", vecs[3].tolist(), 131, deleted=True),
        _recall_row("ghost", vecs[0].tolist(), 132, deleted=True),
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=140, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 1  # one live row folded (tombstones are not folds)

    gen = state_mod.live_generation("t1", "ds")
    delta = gen["deltas"][0]
    assert delta["build_type"] == "consolidate-delta"
    assert int(delta["covered_lsn_hi"]) == 140
    # BOTH deleted ids' int64s are carried (cold id-3 AND recall-only ghost).
    tombs = [int(x) for x in (delta.get("tombstone_int_ids") or [])]
    assert builder_mod._id_to_int64("id-3") in tombs
    assert builder_mod._id_to_int64("ghost") in tombs
    # Base untouched.
    assert storage_mod.read_bytes(base["shard_uri"]) == base_bytes_before


def test_flag_on_delete_recorded_even_when_sidecar_unreadable(delta_on, monkeypatch):
    """Hardening (adversarial review): a delete must NEVER be silently dropped.

    The fold no longer probes cold-tier membership via shard sidecars when
    deciding which tombstones to carry — it records EVERY deleted id's int64
    unconditionally. To prove the delete can't be dropped by a sidecar failure we
    monkeypatch `read_shard_sidecar` to RAISE on every call; the cold id's
    tombstone must still be carried and the watermark must still advance.
    """
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    base_bytes_before = bytes(storage_mod.read_bytes(base["shard_uri"]))

    # Make ANY sidecar read explode. Under the OLD cold-membership logic this
    # would swallow the error, treat cold as empty, and DROP the delete.
    def _boom(*_a, **_k):
        raise RuntimeError("sidecar read failed")

    monkeypatch.setattr(builder_mod, "read_shard_sidecar", _boom)

    # Tombstone-only fold for a cold id (id-7, present in the base).
    rows = [_recall_row("id-7", vecs[7].tolist(), 150, deleted=True)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=160, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 0  # no live rows folded

    gen = state_mod.live_generation("t1", "ds")
    assert len(gen["deltas"]) == 1
    delta = gen["deltas"][0]
    assert delta["build_type"] == "consolidate-delta"
    # The delete was recorded despite the unreadable sidecar — NOT dropped.
    tombs = [int(x) for x in (delta.get("tombstone_int_ids") or [])]
    assert builder_mod._id_to_int64("id-7") in tombs
    # The watermark advanced so the deleted recall row drains.
    assert int(delta["covered_lsn_hi"]) == 160
    assert int(delta["consolidated_lsn"]) == 160
    assert int(delta["vector_count"]) == 0
    # Base untouched.
    assert storage_mod.read_bytes(base["shard_uri"]) == base_bytes_before


def test_flag_on_tombstone_only_with_prior_base_writes_zero_vector_delta(
    delta_on, monkeypatch
):
    builder_mod, state_mod, storage_mod = delta_on
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    base_bytes_before = bytes(storage_mod.read_bytes(base["shard_uri"]))

    # Tombstone-only fold (no live rows) with a prior base: must write a
    # zero-vector delta carrying the tombstone and advancing the watermark.
    rows = [_recall_row("id-5", vecs[5].tolist(), 150, deleted=True)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=160, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 0  # zero live rows folded

    gen = state_mod.live_generation("t1", "ds")
    assert len(gen["deltas"]) == 1
    delta = gen["deltas"][0]
    assert delta["build_type"] == "consolidate-delta"
    assert int(delta["covered_lsn_hi"]) == 160
    assert int(delta["consolidated_lsn"]) == 160
    assert int(delta["vector_count"]) == 0
    tombs = [int(x) for x in (delta.get("tombstone_int_ids") or [])]
    assert builder_mod._id_to_int64("id-5") in tombs
    # Base untouched.
    assert storage_mod.read_bytes(base["shard_uri"]) == base_bytes_before


def test_flag_on_tombstone_only_no_prior_shard_returns_none(delta_on, monkeypatch):
    builder_mod, state_mod, _ = delta_on
    state_mod.create_dataset("t1", "ds", _DIM)
    rows = [_recall_row("nope", _vectors(1, _DIM, 1)[0].tolist(), 10, deleted=True)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=10, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 0
    assert state_mod.get_latest_shard("t1", "ds") is None  # nothing written


# --------------------------------------------------------------------------- #
# 6. Sub-IVF-floor first build under the flag -> flat fallback, no quantizer.  #
# --------------------------------------------------------------------------- #


def test_flag_flipped_on_after_legacy_base_uses_legacy_fold(delta_off, monkeypatch):
    """A legacy IDMap2 base (built flag-OFF, no quantizer object) must keep the
    legacy union-rebuild fold even after the flag flips ON — never try to load a
    quantizer that was never written (which would crash the consolidation)."""
    builder_mod, state_mod, storage_mod = delta_off
    ids = [f"id-{i}" for i in range(_N_COLD)]
    vecs = _vectors(_N_COLD, _DIM, seed=5)
    base = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=_N_COLD
    )
    # Legacy base: IDMap2, quantizer_version 0, no quantizer object.
    assert hasattr(_deser(storage_mod.read_bytes(base["shard_uri"])), "id_map")
    assert int(base.get("quantizer_version", 0) or 0) == 0

    # Flip the flag ON and fold again — must take the legacy path (no crash, no
    # delta row), since there is no quantizer-v0 object to load.
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    assert builder_mod._delta_tier_enabled() is True
    new = _vectors(10, _DIM, seed=6)
    rows = [_recall_row(f"new-{k}", new[k].tolist(), 200 + k) for k in range(10)]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=300, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 10
    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["build_type"] == "consolidate"  # legacy base rewrite, not a delta
    assert int(shard.get("level", 0) or 0) == 0
    # No quantizer object was written (legacy path never saves one).
    assert builder_mod._quantizer_uri("t1", "ds", 1) not in storage_mod._MEM_OBJECTS


def test_flag_on_subfloor_first_build_is_flat_no_quantizer(delta_on, monkeypatch):
    builder_mod, state_mod, storage_mod = delta_on
    # Fewer than IVF_TRAINING_FLOOR (64) rows -> flat fallback.
    ids = [f"id-{i}" for i in range(10)]
    vecs = _vectors(10, _DIM, seed=2)
    shard = _seed_cold_via_consolidate(
        builder_mod, state_mod, monkeypatch, ids, vecs, max_lsn=10
    )
    assert shard["index_type"] == "flat"
    assert int(shard.get("level", 0) or 0) == 0
    assert int(shard.get("quantizer_version", 0) or 0) == 0
    assert builder_mod._quantizer_uri("t1", "ds", 1) not in storage_mod._MEM_OBJECTS
