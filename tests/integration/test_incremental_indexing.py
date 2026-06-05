"""Tests for incremental indexing.

The index builder must `index.add()` only the *new* landing parquet parts onto
the existing FAISS shard rather than rebuilding the whole shard from all
landing data ("rebuild amplification"). These tests prove:

  - first ingest for a dataset still does a full build (unchanged behaviour);
  - a subsequent ingest takes the incremental path: it does NOT retrain and
    does NOT re-read the first batch's landing parts;
  - the final index contains batch1 + batch2 vectors and a query spans both;
  - metadata from both batches lands in the sidecar;
  - empty new batch is skipped cleanly;
  - a dimension mismatch in a later batch fails the build (status=error);
  - a duplicate DATASET_READY for the same batch does not double-index.

`build_type` is asserted via the shard catalog (`indexed_landing_uris` manifest
+ `build_type` column) and via `builder._LAST_BUILD` instrumentation — the
incremental path is taken iff `_LAST_BUILD["build_type"] == "incremental"`.
"""
from __future__ import annotations

import importlib
import os

import faiss  # type: ignore
import numpy as np
import pytest


os.environ["DATABASE_URL"] = "memory://test"


@pytest.fixture
def state(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh state module + per-test MinIO landing/index prefixes, flat index."""
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()
    state_mod.create_tenant("ten_test", "tester@example.com", "x")

    import services.index_builder.run as builder
    importlib.reload(builder)
    return state_mod, s3_landing_prefix, s3_indexes_prefix


def _write_batch(landing: str, tenant: str, dataset: str, upload: str, records):
    """Write one upload's parquet into its own sub-prefix (mirrors validator)."""
    from adapters.landing.parquet_writer import write_parquet

    prefix = f"{landing}/{tenant}/{dataset}/upload-{upload}"
    return write_parquet(prefix, records)


def _read_shard_index(shard_uri):
    """Download a MinIO-resident FAISS shard and load it via faiss."""
    import tempfile

    import faiss  # type: ignore

    from adapters.storage.storage import read_bytes

    blob = read_bytes(shard_uri)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    return faiss.read_index(path)


# --- first ingest: full build unchanged ----------------------------------


def test_first_ingest_is_full_build(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ds1", 4)
    _write_batch(landing, "ten_test", "ds1", "a", [
        {"id": f"a{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"b": "1"}}
        for i in range(5)
    ])
    import services.index_builder.run as builder
    n = builder.run_once("ds1", "ten_test")
    assert n == 5
    assert builder._LAST_BUILD["build_type"] == "full"

    shard = state_mod.get_latest_shard("ten_test", "ds1")
    assert shard["vector_count"] == 5
    assert shard["build_type"] == "full"
    # The first batch's landing part is recorded in the manifest.
    assert len(shard["indexed_landing_uris"]) == 1


# --- subsequent ingest: incremental, no retrain, no re-read ---------------


def test_second_ingest_is_incremental_and_skips_first_batch(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ds2", 4)

    _write_batch(landing, "ten_test", "ds2", "a", [
        {"id": f"a{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"batch": "1"}}
        for i in range(5)
    ])
    import services.index_builder.run as builder
    builder.run_once("ds2", "ten_test")
    first_shard = state_mod.get_latest_shard("ten_test", "ds2")
    first_uris = set(first_shard["indexed_landing_uris"])

    # Second batch in a new upload sub-prefix.
    _write_batch(landing, "ten_test", "ds2", "b", [
        {"id": f"b{i}", "values": [0.0, float(i), 0.0, 0.0], "metadata": {"batch": "2"}}
        for i in range(4)
    ])
    builder.run_once("ds2", "ten_test")

    # (b) The second build took the incremental path and only read batch 2.
    assert builder._LAST_BUILD["build_type"] == "incremental"
    assert builder._LAST_BUILD["vectors_added"] == 4
    assert builder._LAST_BUILD["parts_read"] == 1  # only the new part
    assert set(builder._LAST_BUILD["parts_read_uris"]).isdisjoint(first_uris)

    final_shard = state_mod.get_latest_shard("ten_test", "ds2")
    assert final_shard["build_type"] == "incremental"
    # (a) The final index contains batch1 + batch2.
    assert final_shard["vector_count"] == 9
    # The manifest is the union of both batches' parts.
    assert first_uris.issubset(set(final_shard["indexed_landing_uris"]))
    assert len(final_shard["indexed_landing_uris"]) == 2

    index = _read_shard_index(final_shard["shard_uri"])
    assert index.ntotal == 9

    # (c) A query returns hits spanning both batches.
    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final_shard["shard_uri"])
    # (d) Metadata from both batches is in the sidecar.
    batches = {entry["metadata"].get("batch") for entry in sidecar.values()}
    assert batches == {"1", "2"}

    # Query near a batch-1 vector and near a batch-2 vector; each nearest hit
    # must come from the right batch.
    q1 = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    _d, ids1 = index.search(q1, 1)
    assert sidecar[str(int(ids1[0][0]))]["metadata"]["batch"] == "1"
    q2 = np.array([[0.0, 3.0, 0.0, 0.0]], dtype=np.float32)
    _d, ids2 = index.search(q2, 1)
    assert sidecar[str(int(ids2[0][0]))]["metadata"]["batch"] == "2"


def test_incremental_does_not_retrain_quantizer(state):
    """An incremental build must reuse the trained IVFFlat quantizer unchanged.

    IVFFlat's coarse quantizer is trained once on the first ingest; a
    subsequent `index.add()` appends raw vectors into the existing cells
    without re-running k-means. This test verifies the touch-up did not
    regress that — `add()` onto a trained IVFFlat works exactly as it did for
    IVF+PQ.
    """
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ivf", 8)
    # Enough rows to clear the IVF training floor (64) so the first ingest
    # builds an IVFFlat index rather than the tiny-dataset flat fallback.
    rng = np.random.default_rng(0)
    batch1 = [
        {"id": f"a{i}", "values": rng.random(8).tolist(), "metadata": {"b": "1"}}
        for i in range(400)
    ]
    _write_batch(landing, "ten_test", "ivf", "a", batch1)
    os.environ["INDEX_TYPE"] = "ivfflat"
    import services.index_builder.run as builder
    importlib.reload(builder)
    builder.run_once("ivf", "ten_test")
    shard1 = state_mod.get_latest_shard("ten_test", "ivf")
    assert shard1["index_type"] == "ivfflat", shard1
    idx1 = _read_shard_index(shard1["shard_uri"])
    # Reconstruct the quantizer centroids of the trained IVF index.
    inner1 = faiss.extract_index_ivf(idx1)
    centroids1 = inner1.quantizer.reconstruct_n(0, inner1.nlist).copy()

    batch2 = [
        {"id": f"b{i}", "values": rng.random(8).tolist(), "metadata": {"b": "2"}}
        for i in range(20)
    ]
    _write_batch(landing, "ten_test", "ivf", "b", batch2)
    builder.run_once("ivf", "ten_test")
    assert builder._LAST_BUILD["build_type"] == "incremental"

    shard2 = state_mod.get_latest_shard("ten_test", "ivf")
    idx2 = _read_shard_index(shard2["shard_uri"])
    inner2 = faiss.extract_index_ivf(idx2)
    centroids2 = inner2.quantizer.reconstruct_n(0, inner2.nlist).copy()
    # Quantizer centroids are byte-identical → no retraining happened.
    np.testing.assert_array_equal(centroids1, centroids2)
    assert idx2.ntotal == 420
    os.environ["INDEX_TYPE"] = "flat"


def test_query_path_loads_newest_shard_after_incremental(state, tmp_path, monkeypatch):
    """query_api's hot path must serve the post-incremental shard, not the old one."""
    state_mod, landing, indexes = state
    # FAISS shard cache is a local dir (the `state` fixture already sets one;
    # this keeps the override explicit and filesystem-resident).
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "qp-cache"))
    state_mod.create_dataset("ten_test", "qp", 4)
    _write_batch(landing, "ten_test", "qp", "a", [
        {"id": "a0", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"batch": "1"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("qp", "ten_test")
    _write_batch(landing, "ten_test", "qp", "b", [
        {"id": "b0", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"batch": "2"}},
    ])
    builder.run_once("qp", "ten_test")

    import importlib
    import services.query_api.v1_query as vq
    importlib.reload(vq)
    # A query near the batch-2 vector must surface b0 — only possible if the
    # hot path loaded the newest (incremental) shard.
    matches, _mode = vq._consolidated_search("ten_test", "qp", [0.0, 1.0, 0.0, 0.0], top_k=2)
    ids = {m["id"] for m in matches}
    assert "b0" in ids
    # And the batch-1 vector is still searchable from the same shard.
    matches1, _ = vq._consolidated_search("ten_test", "qp", [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert "a0" in {m["id"] for m in matches1}


# --- edge cases ----------------------------------------------------------


def test_empty_new_batch_is_skipped_cleanly(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "empti", 4)
    _write_batch(landing, "ten_test", "empti", "a", [
        {"id": "a0", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"b": "1"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("empti", "ten_test")
    shards_before = state_mod.list_shards("ten_test", "empti")
    assert len(shards_before) == 1

    # Re-trigger with no new landing parts → nothing to do.
    n = builder.run_once("empti", "ten_test")
    assert n == 0
    assert builder._LAST_BUILD["build_type"] == "noop"
    shards_after = state_mod.list_shards("ten_test", "empti")
    assert len(shards_after) == 1  # no new shard written


def test_dimension_mismatch_in_later_batch_sets_error(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "dim", 4)
    _write_batch(landing, "ten_test", "dim", "a", [
        {"id": "a0", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"b": "1"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("dim", "ten_test")

    # Second batch with a 6-dim vector — mismatches the existing 4-dim index.
    _write_batch(landing, "ten_test", "dim", "b", [
        {"id": "b0", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "metadata": {"b": "2"}},
    ])
    n = builder.run_once("dim", "ten_test")
    assert n == 0
    ds = state_mod.get_dataset("ten_test", "dim")
    assert ds["status"] == "error"
    assert "dimension" in (ds["error_message"] or "").lower()
    # The bad batch must NOT have produced a shard.
    assert len(state_mod.list_shards("ten_test", "dim")) == 1


def test_two_builds_in_same_millisecond_produce_distinct_shards(state, monkeypatch):
    """Two builds completing in the same millisecond must not collide.

    The shard object URI used to be `shard-<ms>.bin`. Two builds in the same
    millisecond produced the same filename and the second `write_bytes`
    silently overwrote the first shard's `.bin`. The uuid suffix makes the
    filename collision-proof. Here we freeze `time.time()` so both builds see
    the *same* millisecond and assert the two shards are still distinct
    objects (different URIs, both present in storage).
    """
    import time as _time

    from adapters.storage.storage import exists

    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "msc", 4)

    import services.index_builder.run as builder

    # Freeze the clock so both builds compute an identical `int(time*1000)`.
    monkeypatch.setattr(builder.time, "time", lambda: 1_700_000_000.123456)

    _write_batch(landing, "ten_test", "msc", "a", [
        {"id": "a0", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"b": "1"}},
    ])
    builder.run_once("msc", "ten_test")
    shard1 = state_mod.get_latest_shard("ten_test", "msc")

    _write_batch(landing, "ten_test", "msc", "b", [
        {"id": "b0", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"b": "2"}},
    ])
    builder.run_once("msc", "ten_test")
    shard2 = state_mod.get_latest_shard("ten_test", "msc")

    # Distinct catalog rows and distinct object URIs despite the frozen clock.
    assert shard1["id"] != shard2["id"]
    assert shard1["shard_uri"] != shard2["shard_uri"]
    # Both shard objects (and their sidecars) survive — neither was overwritten.
    assert exists(shard1["shard_uri"])
    assert exists(f"{shard1['shard_uri']}.meta.json")
    assert exists(shard2["shard_uri"])
    assert exists(f"{shard2['shard_uri']}.meta.json")
    # The second (incremental) build folded both batches into its shard.
    assert shard2["vector_count"] == 2


def test_duplicate_dataset_ready_does_not_double_index(state):
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "dup", 4)
    _write_batch(landing, "ten_test", "dup", "a", [
        {"id": f"a{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"b": "1"}}
        for i in range(3)
    ])
    import services.index_builder.run as builder
    builder.run_once("dup", "ten_test")

    _write_batch(landing, "ten_test", "dup", "b", [
        {"id": f"b{i}", "values": [0.0, float(i), 0.0, 0.0], "metadata": {"b": "2"}}
        for i in range(3)
    ])
    # First DATASET_READY for batch b → indexes it incrementally.
    builder.run_once("dup", "ten_test")
    shard_after_b = state_mod.get_latest_shard("ten_test", "dup")
    assert shard_after_b["vector_count"] == 6

    # Duplicate DATASET_READY for the *same* batch b → manifest is authoritative,
    # no part is new, so no double-counting and no new shard.
    n = builder.run_once("dup", "ten_test")
    assert n == 0
    assert builder._LAST_BUILD["build_type"] == "noop"
    final = state_mod.get_latest_shard("ten_test", "dup")
    assert final["vector_count"] == 6  # still 6, not 9
    assert final["id"] == shard_after_b["id"]  # no new shard row


# --- upsert / dedup (fix/ingest-upsert-dedup) ----------------------------
#
# `POST /v1/datasets/{name}/vectors` is documented as an upsert: re-sending a
# record with an existing id must OVERWRITE it (last-write-wins), not append a
# duplicate. The incremental build path must therefore dedup within each batch
# and, when an incoming batch's ids overlap the existing shard, remove the old
# copies before adding the new ones.


def test_incremental_upsert_replaces_not_duplicates(state):
    """The exact user repro: ingest 6, then ingest 6 re-used ids + 4 new.

    Final row count must be 10 (last-write-wins), NOT 16.
    """
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ups", 4)

    _write_batch(landing, "ten_test", "ups", "a", [
        {"id": f"r{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"v": "1"}}
        for i in range(6)
    ])
    import services.index_builder.run as builder
    builder.run_once("ups", "ten_test")
    first = state_mod.get_latest_shard("ten_test", "ups")
    assert first["vector_count"] == 6

    # Second batch: r0..r5 re-used (overlap) + r6..r9 new.
    _write_batch(landing, "ten_test", "ups", "b", [
        {"id": f"r{i}", "values": [float(i), 1.0, 0.0, 0.0], "metadata": {"v": "2"}}
        for i in range(10)
    ])
    builder.run_once("ups", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "ups")
    assert builder._LAST_BUILD["build_type"] == "incremental"
    # 10, not 16 — the 6 re-sent ids replaced their old copies.
    assert final["vector_count"] == 10
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 10


def test_incremental_upsert_overwrites_vector_value(state):
    """Re-sending an existing id with a different vector replaces the value.

    A query for the NEW vector returns it; the OLD vector is gone (not still
    present as a stale near-duplicate).
    """
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "upsv", 4)

    _write_batch(landing, "ten_test", "upsv", "a", [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "old"}},
        {"id": "y", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"v": "y"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("upsv", "ten_test")

    # Re-send id "x" with a wholly different vector.
    _write_batch(landing, "ten_test", "upsv", "b", [
        {"id": "x", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"v": "new"}},
    ])
    builder.run_once("upsv", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "upsv")
    assert final["vector_count"] == 2  # x (updated) + y, no duplicate x
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 2

    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final["shard_uri"])

    # A query at the NEW vector position returns x with distance ~0.
    q_new = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    d_new, ids_new = index.search(q_new, 1)
    assert d_new[0][0] < 1e-3
    assert sidecar[str(int(ids_new[0][0]))]["id"] == "x"

    # The OLD vector position no longer has an exact (distance ~0) hit — the
    # stale copy was removed.
    q_old = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    d_old, _ids_old = index.search(q_old, 2)
    assert min(float(v) for v in d_old[0]) > 1e-3


def test_within_batch_dup_keeps_last(state):
    """A single ingest containing the same id twice yields one row, last wins."""
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "wb", 4)

    # First ingest establishes a shard so the second ingest is incremental.
    _write_batch(landing, "ten_test", "wb", "a", [
        {"id": "seed", "values": [9.0, 9.0, 9.0, 9.0], "metadata": {"v": "seed"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("wb", "ten_test")

    # Second ingest: id "d" appears twice with different vectors.
    _write_batch(landing, "ten_test", "wb", "b", [
        {"id": "d", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "first"}},
        {"id": "d", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"v": "last"}},
    ])
    builder.run_once("wb", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "wb")
    assert final["vector_count"] == 2  # seed + one "d"
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 2

    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final["shard_uri"])
    # The surviving "d" is the LAST occurrence.
    q = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    d, ids = index.search(q, 1)
    assert d[0][0] < 1e-3
    hit = sidecar[str(int(ids[0][0]))]
    assert hit["id"] == "d"
    assert hit["metadata"]["v"] == "last"


def test_first_ingest_within_batch_dup_keeps_last(state):
    """The first-ingest (full build) path also dedupes within the batch."""
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "fwb", 4)

    _write_batch(landing, "ten_test", "fwb", "a", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "1"}},
        {"id": "dup", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"v": "first"}},
        {"id": "dup", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"v": "last"}},
        {"id": "b", "values": [0.0, 0.0, 0.0, 1.0], "metadata": {"v": "1"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("fwb", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "fwb")
    assert final["build_type"] == "full"
    assert final["vector_count"] == 3  # a + dup + b, not 4
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 3

    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final["shard_uri"])
    q = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    d, ids = index.search(q, 1)
    assert d[0][0] < 1e-3
    assert sidecar[str(int(ids[0][0]))]["metadata"]["v"] == "last"


def test_upsert_keeps_sidecar_metadata_in_sync(state):
    """After an upsert the sidecar entry reflects the NEW metadata."""
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "upsm", 4)

    _write_batch(landing, "ten_test", "upsm", "a", [
        {"id": "m", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "old"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("upsm", "ten_test")

    _write_batch(landing, "ten_test", "upsm", "b", [
        {"id": "m", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"v": "new"}},
    ])
    builder.run_once("upsm", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "upsm")
    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final["shard_uri"])
    key = str(builder._id_to_int64("m"))
    assert sidecar[key]["id"] == "m"
    assert sidecar[key]["metadata"]["v"] == "new"
    # The sidecar and the index agree on the row count.
    assert len(sidecar) == 1
    assert _read_shard_index(final["shard_uri"]).ntotal == 1


def test_pure_append_no_overlap_unaffected(state):
    """The common append-only case (no overlapping ids) still works."""
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "app", 4)

    _write_batch(landing, "ten_test", "app", "a", [
        {"id": f"a{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"v": "1"}}
        for i in range(5)
    ])
    import services.index_builder.run as builder
    builder.run_once("app", "ten_test")

    _write_batch(landing, "ten_test", "app", "b", [
        {"id": f"b{i}", "values": [0.0, float(i), 0.0, 0.0], "metadata": {"v": "2"}}
        for i in range(4)
    ])
    builder.run_once("app", "ten_test")

    final = state_mod.get_latest_shard("ten_test", "app")
    assert builder._LAST_BUILD["build_type"] == "incremental"
    assert final["vector_count"] == 9
    assert _read_shard_index(final["shard_uri"]).ntotal == 9


def test_incremental_upsert_on_ivfflat_index(state):
    """Upsert must also work when the existing shard is an IVFFlat index.

    `IndexIVF.remove_ids` needs an initialised direct map; this exercises the
    `make_direct_map()` path that the flat-index tests above do not.
    """
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "ivfups", 8)
    rng = np.random.default_rng(1)
    batch1 = [
        {"id": f"r{i}", "values": rng.random(8).tolist(), "metadata": {"v": "1"}}
        for i in range(400)
    ]
    _write_batch(landing, "ten_test", "ivfups", "a", batch1)
    os.environ["INDEX_TYPE"] = "ivfflat"
    import services.index_builder.run as builder
    importlib.reload(builder)
    builder.run_once("ivfups", "ten_test")
    shard1 = state_mod.get_latest_shard("ten_test", "ivfups")
    assert shard1["index_type"] == "ivfflat", shard1

    # Re-send r0..r199 (overlap) + r400..r449 new → 450 rows, not 650.
    # r0 is re-sent with a CHANGED, distinctive vector value so the test can
    # query for value correctness after the remove+add+serialize round-trip
    # (count-only assertions would miss a stale-DirectMap upsert bug).
    old_r0 = batch1[0]["values"]
    new_r0 = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    batch2 = [
        {"id": f"r{i}", "values": rng.random(8).tolist(), "metadata": {"v": "2"}}
        for i in list(range(200)) + list(range(400, 450))
    ]
    batch2[0] = {"id": "r0", "values": new_r0, "metadata": {"v": "2"}}
    _write_batch(landing, "ten_test", "ivfups", "b", batch2)
    builder.run_once("ivfups", "ten_test")
    os.environ["INDEX_TYPE"] = "flat"

    assert builder._LAST_BUILD["build_type"] == "incremental"
    final = state_mod.get_latest_shard("ten_test", "ivfups")
    assert final["vector_count"] == 450
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 450

    # IVFFlat needs nprobe high enough to scan the relevant cells.
    index.nprobe = 32
    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(final["shard_uri"])

    # The NEW r0 vector is found with distance ~0 and resolves to id "r0".
    q_new = np.array([new_r0], dtype=np.float32)
    d_new, ids_new = index.search(q_new, 1)
    assert d_new[0][0] < 1e-3, d_new
    assert sidecar[str(int(ids_new[0][0]))]["id"] == "r0"

    # The OLD r0 vector is gone — no ~0 hit at its old position. (random
    # vectors in [0,1)^8 are not within 1e-3 of each other, so any ~0 hit
    # would mean the stale copy survived the upsert.)
    q_old = np.array([old_r0], dtype=np.float32)
    d_old, _ids_old = index.search(q_old, 5)
    assert min(float(v) for v in d_old[0]) > 1e-3, d_old


def test_incremental_upsert_gate_survives_unreadable_sidecar(state, monkeypatch):
    """The upsert overlap gate must NOT depend on the metadata sidecar.

    `read_shard_sidecar` swallows every error and returns `{}` on a transient
    read failure. If the overlap gate derived the existing ids from the
    sidecar, a `{}` result would empty the overlap, skip `remove_ids`, and let
    `_add_to_index` silently APPEND a duplicate of a re-sent id.

    The gate now derives existing ids from the FAISS index's `id_map` (the
    authoritative id store), so it stays correct even with an unusable
    sidecar. This test forces `read_shard_sidecar` to behave as if every read
    failed, then re-sends an existing id with a changed value and asserts no
    duplicate results — it FAILS against the sidecar-based gate.
    """
    state_mod, landing, indexes = state
    state_mod.create_dataset("ten_test", "sidefail", 4)

    _write_batch(landing, "ten_test", "sidefail", "a", [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "old"}},
        {"id": "y", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"v": "y"}},
    ])
    import services.index_builder.run as builder
    builder.run_once("sidefail", "ten_test")
    assert state_mod.get_latest_shard("ten_test", "sidefail")["vector_count"] == 2

    # Simulate a transient sidecar read failure: every read returns {}.
    monkeypatch.setattr(builder, "read_shard_sidecar", lambda _uri: {})

    # Re-send id "x" with a changed vector — a true upsert must replace it.
    _write_batch(landing, "ten_test", "sidefail", "b", [
        {"id": "x", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"v": "new"}},
    ])
    builder.run_once("sidefail", "ten_test")

    assert builder._LAST_BUILD["build_type"] == "incremental"
    final = state_mod.get_latest_shard("ten_test", "sidefail")
    # 2, not 3 — the re-sent "x" replaced its old copy despite the dead sidecar.
    assert final["vector_count"] == 2
    index = _read_shard_index(final["shard_uri"])
    assert index.ntotal == 2

    # The NEW "x" vector is present (distance ~0); the OLD one is gone.
    q_new = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    d_new, _ = index.search(q_new, 1)
    assert d_new[0][0] < 1e-3
    q_old = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    d_old, _ = index.search(q_old, 2)
    assert min(float(v) for v in d_old[0]) > 1e-3
