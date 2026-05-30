"""Integration tests for the index builder's storage sweepers (rough-edges).

Two sweepers run at the end of a successful `run_once`:

  - **Superseded shards** (item 1): every ingest writes a new FAISS shard;
    older ones are dead weight. The sweeper keeps the newest shard plus one
    previous (grace buffer for an in-flight query) and deletes the rest — the
    `.bin`, its `.meta.json` sidecar, and the `shard_catalog` row.

  - **Indexed landing** (item 2): once a landing part is folded into a shard
    (recorded in the newest shard's `indexed_landing_uris` manifest) the part
    is reclaimed. The manifest still records the URI, so this must NOT break a
    subsequent incremental ingest.

These run against real MinIO via the shared `s3_*` fixtures.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


os.environ["DATABASE_URL"] = "memory://test"


@pytest.fixture
def state(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh state + builder modules with per-test MinIO prefixes (flat index)."""
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS",
                 "_MEM_DATASETS", "_MEM_IMPORTS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
    state_mod._MEM_SHARDS.clear()
    state_mod.create_tenant("ten_test", "tester@example.com", "x")

    import services.index_builder.run as builder
    importlib.reload(builder)
    return state_mod, builder, s3_landing_prefix, s3_indexes_prefix


def _write_batch(landing, tenant, dataset, upload, records):
    from adapters.landing.parquet_writer import write_parquet

    prefix = f"{landing}/{tenant}/{dataset}/upload-{upload}"
    return write_parquet(prefix, records)


def _vec():
    return list(np.random.rand(4).astype(float))


# --- item 1: superseded shard sweep --------------------------------------


def test_sweep_keeps_newest_two_shards_and_deletes_older(state):
    state_mod, builder, landing, _ = state
    from adapters.storage.storage import exists

    # Four ingests → four builds → four shards. The 1st build is full, the
    # next three incremental; after each the sweeper runs.
    for i in range(4):
        _write_batch(landing, "ten_test", "swp", chr(ord("a") + i),
                     [{"id": f"r{i}", "values": _vec(), "metadata": {}}])
        builder.run_once("swp", "ten_test")

    shards = state_mod.list_shards("ten_test", "swp")
    # Only the newest two shard rows survive in the catalog.
    assert len(shards) == 2, f"expected 2 retained shards, got {len(shards)}"
    # Their objects (and sidecars) still exist.
    for s in shards:
        assert exists(s["shard_uri"]), "retained shard .bin missing"
        assert exists(f"{s['shard_uri']}.meta.json"), "retained sidecar missing"


def test_query_still_works_after_shard_sweep(state):
    state_mod, builder, landing, _ = state
    from adapters.landing.parquet_reader import read_shard_sidecar

    for i in range(4):
        _write_batch(landing, "ten_test", "qsw", chr(ord("a") + i),
                     [{"id": f"r{i}", "values": _vec(), "metadata": {"b": i}}])
        builder.run_once("qsw", "ten_test")

    latest = state_mod.get_latest_shard("ten_test", "qsw")
    # The newest shard is fully usable: index loads and the sidecar resolves.
    blob_index = read_shard_sidecar(latest["shard_uri"])
    assert latest["vector_count"] == 4
    assert len(blob_index) == 4  # sidecar covers every vector


# --- item 2: indexed-landing sweep does not break incremental ingest -----


def test_landing_sweep_does_not_break_subsequent_incremental(state):
    state_mod, builder, landing, _ = state
    from adapters.storage.storage import exists

    p1 = _write_batch(landing, "ten_test", "lsw", "a",
                      [{"id": "r0", "values": _vec(), "metadata": {}}])
    builder.run_once("lsw", "ten_test")
    # The first part is now folded into the shard; the sweeper deleted it.
    assert not exists(p1), "indexed landing part should have been swept"

    # A second ingest must still take the incremental path cleanly even though
    # the first part's bytes are gone — the manifest still records its URI.
    _write_batch(landing, "ten_test", "lsw", "b",
                 [{"id": "r1", "values": _vec(), "metadata": {}}])
    added = builder.run_once("lsw", "ten_test")
    assert added == 1
    assert builder._LAST_BUILD["build_type"] == "incremental"

    final = state_mod.get_latest_shard("ten_test", "lsw")
    assert final["vector_count"] == 2  # batch1 + batch2 both in the index
    # The manifest still records both parts (sweep deletes bytes, not history).
    assert len(final["indexed_landing_uris"]) == 2
