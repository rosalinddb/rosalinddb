"""Unit tests for the index builder's DELETE_VECTORS handler.

The handler loads the newest shard's FAISS index + sidecar, removes one
hashed id, rewrites the sidecar without it, and writes a NEW superseded shard
via the same build/catalog/sweep tail `run_once` uses. These tests run end to
end against a real FAISS index but use the `memory://` storage + state
adapters so they stay hermetic (no MinIO/Postgres).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def builder(monkeypatch):
    """Reloaded builder + state + storage bound to memory:// prefixes."""
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny fixtures use the flat path

    import adapters.storage.storage as storage_mod
    importlib.reload(storage_mod)
    storage_mod._MEM_OBJECTS.clear()

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0
    for attr in ("_MEM_DATASETS",):
        getattr(state_mod, attr).clear()

    import services.index_builder.run as builder_mod
    importlib.reload(builder_mod)
    return builder_mod, state_mod, storage_mod


def _seed_shard(builder_mod, ids, dim=4, tenant="t1", dataset="ds"):
    """Build a real flat shard for `ids` via the builder's full-build tail.

    Writes a landing parquet under the prefix the builder scans, then runs
    `run_once` so the shard + sidecar are produced exactly as production
    would. Returns the newest shard row.
    """
    import adapters.state.state as state_mod
    from adapters.landing.parquet_writer import write_parquet

    # A dataset_catalog row must exist so the builder's status flips land
    # somewhere (run_delete_once calls update_dataset_status).
    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, dim)
    vectors = np.random.rand(len(ids), dim).astype(np.float32)
    records = [
        {"id": rid, "values": vectors[i].tolist(), "metadata": {"n": i}}
        for i, rid in enumerate(ids)
    ]
    landing_prefix = builder_mod._landing_prefix(dataset, tenant)
    write_parquet(f"{landing_prefix}/uploads", records)
    builder_mod.run_once(dataset, tenant)
    return state_mod.get_latest_shard(tenant, dataset)


def test_delete_removes_id_from_sidecar_and_supersedes(builder):
    builder_mod, state_mod, _ = builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar

    ids = ["a", "b", "c", "d"]
    before = _seed_shard(builder_mod, ids)
    assert before is not None
    before_shard_uri = before["shard_uri"]

    # Delete "b".
    added = builder_mod.run_delete_once("ds", "t1", "b")
    assert added is not None

    latest = state_mod.get_latest_shard("t1", "ds")
    assert latest is not None
    # A NEW shard superseded the old one (different uri).
    assert latest["shard_uri"] != before_shard_uri
    assert latest["vector_count"] == 3

    # The new sidecar no longer carries "b" but keeps the rest.
    sidecar = read_shard_sidecar(latest["shard_uri"])
    surviving = {entry["id"] for entry in sidecar.values()}
    assert surviving == {"a", "c", "d"}
    assert str(id_to_int64("b")) not in sidecar

    # The FAISS index dropped the hashed id too.
    import faiss

    from adapters.storage.storage import read_bytes

    index = faiss.deserialize_index(
        np.frombuffer(read_bytes(latest["shard_uri"]), dtype=np.uint8)
    )
    live = set(faiss.vector_to_array(index.id_map).tolist())
    assert id_to_int64("b") not in live
    assert id_to_int64("a") in live


def test_delete_missing_id_is_noop(builder):
    builder_mod, state_mod, _ = builder
    before = _seed_shard(builder_mod, ["a", "b"])
    n_shards_before = len(state_mod.list_shards("t1", "ds"))

    result = builder_mod.run_delete_once("ds", "t1", "does-not-exist")
    # A delete of an absent id must be a clean no-op — no new shard.
    assert result == 0
    assert len(state_mod.list_shards("t1", "ds")) == n_shards_before
    assert state_mod.get_latest_shard("t1", "ds")["shard_uri"] == before["shard_uri"]


def test_delete_no_shard_is_noop(builder):
    builder_mod, state_mod, _ = builder
    # Dataset with no shard at all.
    result = builder_mod.run_delete_once("ds", "t1", "anything")
    assert result == 0
    assert state_mod.get_latest_shard("t1", "ds") is None


def test_delete_no_shard_leaves_empty_status(builder):
    """A no-shard delete must leave a never-ingested dataset's status `empty`.

    Regression for the status-integrity bug: the builder used to force
    `indexed` in the no-shard branch, so deleting on an `empty` dataset
    reported `status=indexed, row_count=0`, masking the true state.
    """
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)
    assert state_mod.get_dataset("t1", "ds")["status"] == "empty"

    result = builder_mod.run_delete_once("ds", "t1", "anything")
    assert result == 0
    assert state_mod.get_latest_shard("t1", "ds") is None
    # Status untouched — NOT rewritten to `indexed`.
    assert state_mod.get_dataset("t1", "ds")["status"] == "empty"


def test_delete_no_shard_leaves_error_status(builder):
    """A no-shard delete must not clobber an `error` dataset's status."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)
    state_mod.update_dataset_status("t1", "ds", "error", error_message="boom")

    result = builder_mod.run_delete_once("ds", "t1", "anything")
    assert result == 0
    row = state_mod.get_dataset("t1", "ds")
    assert row["status"] == "error"
    assert row["error_message"] == "boom"


def test_delete_labels_build_type_delete(builder):
    """A delete-driven rebuild stamps `build_type='delete'` on the new shard.

    Distinct from an ingest's `incremental` so deletes are not miscounted as
    ingests in `build_type`-keyed metrics / the shard_catalog column.
    """
    builder_mod, state_mod, _ = builder
    _seed_shard(builder_mod, ["a", "b", "c"])
    builder_mod.run_delete_once("ds", "t1", "b")
    latest = state_mod.get_latest_shard("t1", "ds")
    assert latest["build_type"] == "delete"


def test_handle_delete_vectors_message(builder):
    builder_mod, state_mod, _ = builder
    from adapters.landing.parquet_reader import read_shard_sidecar
    from adapters.queue.queue import Message

    _seed_shard(builder_mod, ["a", "b", "c"])
    msg = Message(
        {"dataset": "ds", "tenant": "t1", "id": "b", "job_id": "job_x"},
        topic="DELETE_VECTORS", msg_id="m1", raw="{}",
    )
    done = builder_mod._handle_delete_vectors(msg)
    assert done is True

    latest = state_mod.get_latest_shard("t1", "ds")
    surviving = {e["id"] for e in read_shard_sidecar(latest["shard_uri"]).values()}
    assert surviving == {"a", "c"}
    # Dataset flips back to indexed once the delete build lands.
    ds = state_mod.get_dataset("t1", "ds")
    assert ds["status"] == "indexed"


def test_delete_last_vector_leaves_empty_shard(builder):
    builder_mod, state_mod, _ = builder
    from adapters.landing.parquet_reader import read_shard_sidecar

    _seed_shard(builder_mod, ["solo"])
    builder_mod.run_delete_once("ds", "t1", "solo")
    latest = state_mod.get_latest_shard("t1", "ds")
    assert latest is not None
    assert latest["vector_count"] == 0
    assert read_shard_sidecar(latest["shard_uri"]) == {}
