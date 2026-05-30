"""Tests for the builder-reads-landing path.

Covers:
  - parquet_writer + parquet_reader roundtrip (ids/vectors/metadata)
  - empty landing returns empty arrays and builder skips without crashing
  - builder consumes DATASET_READY, reads landing, builds a real FAISS shard
  - shard's vectors match the input (loaded via faiss.read_index)
  - builder updates dataset status to `indexed` and sets `last_indexed_at`
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

import faiss  # type: ignore
import numpy as np
import pytest


os.environ["DATABASE_URL"] = "memory://test"


@pytest.fixture
def state(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh state module + per-test MinIO landing/index prefixes.

    Landing and indexes live in real MinIO (object-storage-first); only the
    FAISS shard cache (`CACHE_DIR`) is a local tmp dir, since FAISS reads an
    index from a filesystem path.
    """
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny test fixtures use flat

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()
    # Seed a tenant so dataset rows reference something coherent in the
    # in-memory adapter; FK enforcement only matters in postgres mode.
    state_mod.create_tenant("ten_test", "tester@example.com", "x")

    # Reload builder/validator modules so module-level env reads pick up
    # the patched MinIO prefixes.
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    return state_mod, s3_landing_prefix, s3_indexes_prefix


# --- parquet_writer + parquet_reader roundtrip ---------------------------


def test_parquet_roundtrip_preserves_ids_vectors_metadata(s3_landing_prefix):
    from adapters.landing.parquet_writer import write_parquet
    from adapters.landing.parquet_reader import read_landing_vectors

    prefix = f"{s3_landing_prefix}/rt"
    records = [
        {"id": "a", "values": [1.0, 2.0, 3.0, 4.0], "metadata": {"k": "1"}},
        {"id": "b", "values": [5.0, 6.0, 7.0, 8.0], "metadata": {"k": "2"}},
    ]
    write_parquet(prefix, records)
    ids, vectors, metas = read_landing_vectors(prefix)
    assert ids == ["a", "b"]
    assert vectors.shape == (2, 4)
    np.testing.assert_allclose(vectors[0], [1, 2, 3, 4])
    np.testing.assert_allclose(vectors[1], [5, 6, 7, 8])
    assert metas[0]["k"] == "1"
    assert metas[1]["k"] == "2"


def test_parquet_roundtrip_multiple_files(s3_landing_prefix):
    """Multiple parquet files under the prefix should all be read and concat."""
    from adapters.landing.parquet_writer import write_parquet
    from adapters.landing.parquet_reader import read_landing_vectors

    base = f"{s3_landing_prefix}/multi"
    # Write into two distinct sub-prefixes so the writer's fixed
    # `part-0001.parquet` filename does not collide.
    write_parquet(f"{base}/u1", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"_": "1"}},
    ])
    write_parquet(f"{base}/u2", [
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"_": "1"}},
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"_": "1"}},
    ])
    ids, vectors, _ = read_landing_vectors(base)
    assert sorted(ids) == ["a", "b", "c"]
    assert vectors.shape == (3, 4)


def test_empty_landing_returns_empty_arrays(s3_landing_prefix):
    from adapters.landing.parquet_reader import read_landing_vectors

    # Nothing written under this prefix -> empty arrays.
    ids, vectors, metas = read_landing_vectors(f"{s3_landing_prefix}/empty")
    assert ids == []
    assert vectors.shape == (0, 0) or vectors.size == 0
    assert metas == []


def test_missing_landing_directory_returns_empty(s3_landing_prefix):
    from adapters.landing.parquet_reader import read_landing_vectors

    ids, vectors, metas = read_landing_vectors(f"{s3_landing_prefix}/does-not-exist")
    assert ids == []
    assert vectors.size == 0
    assert metas == []


# --- index builder reads landing -----------------------------------------


def test_builder_skips_empty_landing(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "empty-ds", 4)
    import services.index_builder.run as builder
    n = builder.run_once("empty-ds", "ten_test")
    assert n == 0
    # No shard should have been added
    shards = state_mod.list_shards("ten_test", "empty-ds")
    assert shards == []
    # Status remains `empty` (builder must not stomp it)
    ds = state_mod.get_dataset("ten_test", "empty-ds")
    assert ds["status"] == "empty"


def _read_shard_index(shard_uri):
    """Download a MinIO-resident FAISS shard and load it via faiss."""
    import tempfile

    from adapters.storage.storage import read_bytes

    blob = read_bytes(shard_uri)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    return faiss.read_index(path)


def test_builder_reads_landing_and_writes_shard(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ds", 4)
    # Populate landing parquet directly (skip validator for unit isolation).
    from adapters.landing.parquet_writer import write_parquet
    from adapters.storage.storage import read_bytes
    records = [
        {"id": f"r{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"_": "1"}}
        for i in range(5)
    ]
    landing_prefix = f"{landing}/ten_test/ds/upload-1"
    write_parquet(landing_prefix, records)

    import services.index_builder.run as builder
    n = builder.run_once("ds", "ten_test")
    assert n == 5
    shards = state_mod.list_shards("ten_test", "ds")
    assert len(shards) == 1
    shard = shards[0]
    assert shard["vector_count"] == 5
    # Shard object should physically exist in MinIO.
    assert read_bytes(shard["shard_uri"])


def test_shard_vectors_round_trip_back_to_input(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "round", 4)
    from adapters.landing.parquet_writer import write_parquet
    records = [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"_": "1"}},
        {"id": "y", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"_": "1"}},
        {"id": "z", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"_": "1"}},
    ]
    write_parquet(f"{landing}/ten_test/round/up", records)

    import services.index_builder.run as builder
    builder.run_once("round", "ten_test")
    shards = state_mod.list_shards("ten_test", "round")
    assert len(shards) == 1
    index = _read_shard_index(shards[0]["shard_uri"])
    assert index.ntotal == 3
    # Query the index with one of the inputs and expect a near-zero distance
    # to itself; this is the "vectors round-trip back to input" assertion.
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    distances, ids = index.search(q, 1)
    # The closest neighbor must be the same vector (distance ~ 0).
    assert distances[0][0] < 1e-3, distances


def test_builder_updates_dataset_status_to_indexed(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "stat", 4)
    from adapters.landing.parquet_writer import write_parquet
    write_parquet(f"{landing}/ten_test/stat/up", [
        {"id": "a", "values": [1, 2, 3, 4], "metadata": {"_": "1"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("stat", "ten_test")
    ds = state_mod.get_dataset("ten_test", "stat")
    assert ds["status"] == "indexed", ds
    assert ds["last_indexed_at"], ds


@pytest.mark.parametrize("n_vectors", [20, 40])
def test_first_ingest_below_ivf_floor_builds_flat(state, monkeypatch, n_vectors):
    """Recall touch-up: a tiny first ingest (below the IVF training floor)
    falls back to the exact `flat` index.

    IVFFlat has a single training step — k-means on the `nlist` coarse-
    quantizer centroids — so the index-type gate only needs IVF's floor
    (`>= 64` rows, `nlist >= 4`). Below it the gate builds an exact
    `IndexFlatL2`, which is queryable immediately.
    """
    state_mod, landing, indexes = state
    # Force the IVFFlat regime so the gate — not the test fixture's `flat`
    # env override — is what decides; for a tiny batch it must still pick
    # `flat`.
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    importlib.reload(__import__("services.index_builder.run", fromlist=["run"]))
    import services.index_builder.run as builder

    ds_name = f"ivffloor{n_vectors}"
    state_mod.create_dataset("ten_test", ds_name, 8)
    from adapters.landing.parquet_writer import write_parquet
    records = [
        {"id": f"r{i}", "values": [float(i)] + [0.0] * 7, "metadata": {"_": "1"}}
        for i in range(n_vectors)
    ]
    write_parquet(f"{landing}/ten_test/{ds_name}/up", records)

    n = builder.run_once(ds_name, "ten_test")
    assert n == n_vectors
    ds = state_mod.get_dataset("ten_test", ds_name)
    assert ds["status"] == "indexed", ds  # NOT error
    shards = state_mod.list_shards("ten_test", ds_name)
    assert len(shards) == 1
    # Below the IVF floor (64) the gate must fall back to the exact flat index.
    assert shards[0]["index_type"] == "flat", shards[0]
    assert shards[0]["vector_count"] == n_vectors
    # The shard is queryable: a near-neighbour search returns a real hit.
    index = _read_shard_index(shards[0]["shard_uri"])
    assert index.ntotal == n_vectors
    q = np.array([[5.0] + [0.0] * 7], dtype=np.float32)
    distances, ids = index.search(q, 1)
    assert ids[0][0] != -1


def test_first_ingest_above_ivf_floor_builds_ivfflat(state, monkeypatch):
    """Recall touch-up: a non-tiny first ingest builds an **IVFFlat** index
    (IVF coarse quantizer + raw, uncompressed vectors) — never IVF+PQ.

    IVFFlat ranks on exact L2 distances, so recall is no longer PQ-ceilinged;
    it is still an IVF index, so the query path's `nprobe` knob keeps working.
    """
    state_mod, landing, indexes = state
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    importlib.reload(__import__("services.index_builder.run", fromlist=["run"]))
    import services.index_builder.run as builder

    ds_name = "ivfflatds"
    state_mod.create_dataset("ten_test", ds_name, 8)
    from adapters.landing.parquet_writer import write_parquet
    rng = np.random.default_rng(0)
    records = [
        {"id": f"r{i}", "values": rng.random(8).tolist(), "metadata": {"_": "1"}}
        for i in range(300)
    ]
    write_parquet(f"{landing}/ten_test/{ds_name}/up", records)

    n = builder.run_once(ds_name, "ten_test")
    assert n == 300
    ds = state_mod.get_dataset("ten_test", ds_name)
    assert ds["status"] == "indexed", ds
    shards = state_mod.list_shards("ten_test", ds_name)
    assert len(shards) == 1
    assert shards[0]["index_type"] == "ivfflat", shards[0]
    # The serialized shard is a real IVFFlat: an IVF index storing raw vectors.
    index = _read_shard_index(shards[0]["shard_uri"])
    concrete = faiss.downcast_index(faiss.extract_index_ivf(index))  # raises if not IVF
    assert isinstance(concrete, faiss.IndexIVFFlat)
    assert not isinstance(concrete, faiss.IndexIVFPQ)
    assert index.ntotal == 300
    # And it is queryable.
    q = np.array([records[5]["values"]], dtype=np.float32)
    distances, ids = index.search(q, 1)
    assert ids[0][0] != -1


def test_validator_writes_parquet_builder_reads_it(state):
    """End-to-end on the in-process queues: validator produces, builder consumes."""
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "e2e", 4)
    # Stage a JSONL source object in MinIO that the validator will read.
    from adapters.storage.storage import write_bytes
    src_uri = f"{landing}/source.jsonl"
    body = "".join(
        '{"id":"r%d","values":[%.1f, 0.0, 0.0, 0.0]}\n' % (i, float(i))
        for i in range(6)
    )
    write_bytes(src_uri, body.encode("utf-8"))

    import services.validator_worker.run as validator
    import services.index_builder.run as builder
    validator.process_uri("e2e", "ten_test", src_uri, "jsonl")
    n = builder.run_once("e2e", "ten_test")
    assert n == 6
    shards = state_mod.list_shards("ten_test", "e2e")
    assert len(shards) == 1
    ds = state_mod.get_dataset("ten_test", "e2e")
    assert ds["status"] == "indexed"
    assert ds["row_count"] == 6
