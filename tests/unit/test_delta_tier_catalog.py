"""Unit tests for the delta-tier control-plane surface (Phase 1, PR-A).

Covers the additive shard_catalog columns and the generation-aware catalog
helpers that the delta-shard LSM relies on:

  - `add_shard` round-trips the six new columns (quantizer_version,
    parent_shard_id, level, covered_lsn_lo/hi, tombstone_int_ids).
  - `live_generation` groups a base (level=0) with its deltas (level=1,
    parent_shard_id==base.id, same quantizer_version) and ignores foreign rows.
  - `superseded_shards` is rewritten to LIVENESS-BY-GENERATION: a live delta is
    never swept even though it sorts to the head by created_at — the P0 the
    stress phase proved (the old `list_shards[keep:]` would GC the live base).
    For a base-only dataset (no deltas) it MUST behave exactly as before, so the
    existing `test_shard_catalog.py` assertions still hold.
  - `dataset_watermark` = MAX(consolidated_lsn) over the live generation.
  - `grace_watermark` = the oldest-still-live generation's frontier, never a
    sibling delta's lsn.

All hermetic on the `memory://` state adapter.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def state():
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0
    return state_mod


def _base(state, tenant, dataset, *, qv=1, lsn=0, lo=0, hi=0, vc=100):
    return state.add_shard(
        tenant, dataset, f"memory://idx/{dataset}/base-{qv}.bin",
        checksum=f"b{qv}", vector_count=vc, index_type="ivfflat",
        build_type="consolidate", consolidated_lsn=lsn,
        quantizer_version=qv, parent_shard_id=None, level=0,
        covered_lsn_lo=lo, covered_lsn_hi=hi,
    )


def _delta(state, tenant, dataset, base_id, *, qv=1, lsn, lo, hi,
           tombstones=None, vc=5):
    return state.add_shard(
        tenant, dataset, f"memory://idx/{dataset}/delta-{lsn}.bin",
        checksum=f"d{lsn}", vector_count=vc, index_type="ivfflat",
        build_type="consolidate-delta", consolidated_lsn=lsn,
        quantizer_version=qv, parent_shard_id=base_id, level=1,
        covered_lsn_lo=lo, covered_lsn_hi=hi,
        tombstone_int_ids=tombstones or [],
    )


# ---- new columns round-trip ------------------------------------------------

def test_add_shard_round_trips_delta_columns(state):
    bid = _base(state, "t", "ds", qv=3, lsn=100, lo=0, hi=100)
    did = _delta(state, "t", "ds", bid, qv=3, lsn=140, lo=101, hi=140,
                 tombstones=[111, 222])
    rows = {s["id"]: s for s in state.list_shards("t", "ds")}
    b, d = rows[bid], rows[did]
    assert (b["quantizer_version"], b["level"], b["parent_shard_id"]) == (3, 0, None)
    assert (b["covered_lsn_lo"], b["covered_lsn_hi"]) == (0, 100)
    assert (d["quantizer_version"], d["level"], d["parent_shard_id"]) == (3, 1, bid)
    assert (d["covered_lsn_lo"], d["covered_lsn_hi"]) == (101, 140)
    assert list(d["tombstone_int_ids"]) == [111, 222]
    assert d["build_type"] == "consolidate-delta"


def test_add_shard_delta_columns_default_safely(state):
    """Legacy callers that pass none of the new kwargs get a level-0 base row."""
    state.add_shard("t", "ds", "memory://idx/ds/s.bin", checksum="c",
                    vector_count=1, index_type="flat")
    s = state.get_latest_shard("t", "ds")
    assert s["level"] == 0 and s["parent_shard_id"] is None
    assert s["quantizer_version"] == 0
    assert s["covered_lsn_lo"] == 0 and s["covered_lsn_hi"] == 0
    assert list(s["tombstone_int_ids"]) == []


# ---- live_generation -------------------------------------------------------

def test_live_generation_groups_base_and_deltas(state):
    bid = _base(state, "t", "ds", qv=1, lsn=100, hi=100)
    d1 = _delta(state, "t", "ds", bid, qv=1, lsn=120, lo=101, hi=120)
    d2 = _delta(state, "t", "ds", bid, qv=1, lsn=140, lo=121, hi=140)
    gen = state.live_generation("t", "ds")
    assert gen["base"]["id"] == bid
    # deltas ordered by covered_lsn_lo (oldest band first)
    assert [d["id"] for d in gen["deltas"]] == [d1, d2]


def test_live_generation_excludes_foreign_and_old(state):
    old = _base(state, "t", "ds", qv=1, lsn=50, hi=50)
    _delta(state, "t", "ds", old, qv=1, lsn=60, lo=51, hi=60)  # delta on OLD base
    new = _base(state, "t", "ds", qv=2, lsn=100, hi=100)       # new generation
    dn = _delta(state, "t", "ds", new, qv=2, lsn=130, lo=101, hi=130)
    gen = state.live_generation("t", "ds")
    assert gen["base"]["id"] == new
    assert [d["id"] for d in gen["deltas"]] == [dn]  # only the new base's delta


def test_live_generation_none_when_empty(state):
    assert state.live_generation("t", "empty") is None


# ---- superseded_shards: the P0 fix -----------------------------------------

def test_superseded_never_sweeps_live_base_or_deltas(state):
    """1 base + 8 deltas: ALL survive. (Old list_shards[2:] would GC base+6 deltas.)"""
    bid = _base(state, "t", "ds", qv=1, lsn=100, hi=100)
    dids = []
    lo = 101
    for i in range(8):
        hi = lo + 9
        dids.append(_delta(state, "t", "ds", bid, qv=1, lsn=hi, lo=lo, hi=hi))
        lo = hi + 1
    stale = state.superseded_shards("t", "ds")
    assert stale == [], "live base + its 8 deltas must never be superseded"
    live_ids = {s["id"] for s in state.list_shards("t", "ds")}
    assert live_ids == {bid, *dids}


def test_superseded_sweeps_prior_generations_beyond_keep(state):
    """keep=2 generations: gen0 (current) + gen1 (grace) live; gen2 swept."""
    g0_old = _base(state, "t", "ds", qv=1, lsn=50, hi=50)        # gen2 (oldest)
    _delta(state, "t", "ds", g0_old, qv=1, lsn=60, lo=51, hi=60)
    g1 = _base(state, "t", "ds", qv=2, lsn=100, hi=100)          # gen1 (grace)
    g1d = _delta(state, "t", "ds", g1, qv=2, lsn=110, lo=101, hi=110)
    g2 = _base(state, "t", "ds", qv=3, lsn=200, hi=200)          # gen0 (current)
    g2d = _delta(state, "t", "ds", g2, qv=3, lsn=210, lo=201, hi=210)
    stale_ids = {s["id"] for s in state.superseded_shards("t", "ds")}
    # oldest generation (g0_old + its delta) is beyond keep=2 -> swept
    assert g0_old in stale_ids
    # current + grace generations survive
    assert stale_ids.isdisjoint({g1, g1d, g2, g2d})


def test_superseded_base_only_matches_legacy_behavior(state):
    """No deltas -> generation logic must degenerate to list_shards[keep:]."""
    ids = [
        state.add_shard("t", "ds", f"memory://idx/ds/s{i}.bin", checksum=f"c{i}",
                        vector_count=10, index_type="flat")
        for i in range(5)
    ]  # oldest-first
    stale = {s["id"] for s in state.superseded_shards("t", "ds")}
    assert stale == set(ids[:3])  # newest two kept, identical to legacy
    stale1 = {s["id"] for s in state.superseded_shards("t", "ds", keep=1)}
    assert stale1 == set(ids[:4])


# ---- watermarks ------------------------------------------------------------

def test_dataset_watermark_is_max_over_live_generation(state):
    bid = _base(state, "t", "ds", qv=1, lsn=100, hi=100)
    _delta(state, "t", "ds", bid, qv=1, lsn=140, lo=101, hi=140)
    _delta(state, "t", "ds", bid, qv=1, lsn=170, lo=141, hi=170)
    assert state.dataset_watermark("t", "ds") == 170


def test_dataset_watermark_zero_when_empty(state):
    assert state.dataset_watermark("t", "empty") == 0


def test_grace_watermark_is_oldest_live_generation_frontier(state):
    # gen1 (grace): base@100 + delta covering up to 130 -> frontier 130
    g1 = _base(state, "t", "ds", qv=1, lsn=100, hi=100)
    _delta(state, "t", "ds", g1, qv=1, lsn=130, lo=101, hi=130)
    # gen0 (current): base@200 + delta up to 240
    g0 = _base(state, "t", "ds", qv=2, lsn=200, hi=200)
    _delta(state, "t", "ds", g0, qv=2, lsn=240, lo=201, hi=240)
    # keep=2 -> oldest live generation is gen1, frontier = 130 (NOT a current-gen lsn)
    assert state.grace_watermark("t", "ds") == 130


def test_grace_watermark_zero_with_single_generation(state):
    bid = _base(state, "t", "ds", qv=1, lsn=100, hi=100)
    _delta(state, "t", "ds", bid, qv=1, lsn=140, lo=101, hi=140)
    # only one generation exists -> nothing has aged into the grace window
    assert state.grace_watermark("t", "ds") == 0
