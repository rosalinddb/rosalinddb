"""Unit tests for shard-catalog sweep helpers (rough-edges — item 1).

`superseded_shards` / `delete_shards` back the index builder's superseded-shard
sweeper. The sweeper retains the newest shard plus one previous (a grace buffer
for an in-flight query) and deletes the rest. These tests run on the
`memory://` state adapter so they stay hermetic.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def state():
    """Fresh in-memory state module with an empty shard catalog."""
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0
    return state_mod


def _add(state, tenant, dataset, n):
    """Add `n` shards for a dataset, returning their ids oldest-first."""
    ids = []
    for i in range(n):
        ids.append(
            state.add_shard(
                tenant, dataset, f"memory://idx/{dataset}/shard-{i}.bin",
                checksum=f"c{i}", vector_count=10, index_type="flat",
                indexed_landing_uris=[f"memory://landing/{dataset}/p{i}.parquet"],
            )
        )
    return ids


def test_superseded_keeps_newest_two_by_default(state):
    ids = _add(state, "t1", "ds", 5)  # oldest-first
    stale = state.superseded_shards("t1", "ds")
    stale_ids = {s["id"] for s in stale}
    # Newest two (the last two added) are retained; the older three are stale.
    assert stale_ids == set(ids[:3])


def test_superseded_empty_when_two_or_fewer(state):
    _add(state, "t1", "ds", 2)
    assert state.superseded_shards("t1", "ds") == []
    _add(state, "t1", "solo", 1)
    assert state.superseded_shards("t1", "solo") == []


def test_delete_shards_removes_only_named_rows(state):
    ids = _add(state, "t1", "ds", 4)
    removed = state.delete_shards("t1", "ds", ids[:2])
    assert removed == 2
    remaining = {s["id"] for s in state.list_shards("t1", "ds")}
    assert remaining == set(ids[2:])


def test_delete_shards_is_tenant_dataset_scoped(state):
    a_ids = _add(state, "tA", "ds", 2)
    b_ids = _add(state, "tB", "ds", 2)
    # Passing tenant tA's ids under tenant tB deletes nothing.
    assert state.delete_shards("tB", "ds", a_ids) == 0
    assert len(state.list_shards("tA", "ds")) == 2
    assert len(state.list_shards("tB", "ds")) == 2
    # Correct scope removes them.
    assert state.delete_shards("tA", "ds", a_ids) == 2
    assert state.list_shards("tA", "ds") == []
    assert len(state.list_shards("tB", "ds")) == 2


def test_keep_one_leaves_only_newest(state):
    ids = _add(state, "t1", "ds", 3)
    stale = state.superseded_shards("t1", "ds", keep=1)
    assert {s["id"] for s in stale} == set(ids[:2])


def test_add_shard_defaults_consolidated_lsn_to_zero(state):
    """Every non-consolidate build leaves the recall watermark at 0 (default-off)."""
    state.add_shard(
        "t1", "ds", "memory://idx/ds/s.bin", checksum="c", vector_count=1,
        index_type="flat",
    )
    shard = state.get_latest_shard("t1", "ds")
    assert shard["consolidated_lsn"] == 0
    assert shard["build_type"] == "full"


def test_add_shard_round_trips_consolidated_lsn(state):
    """A consolidation stamps the watermark + build_type='consolidate' on the row."""
    state.add_shard(
        "t1", "ds", "memory://idx/ds/c.bin", checksum="c", vector_count=3,
        index_type="flat", build_type="consolidate", consolidated_lsn=42,
    )
    shard = state.get_latest_shard("t1", "ds")
    assert shard["consolidated_lsn"] == 42
    assert shard["build_type"] == "consolidate"
