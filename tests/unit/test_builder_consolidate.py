"""Unit tests for the index builder's CONSOLIDATE handler (recall→consolidated flush).

These exercise the consolidation OPERATION end-to-end against a real FAISS index
and the `memory://` storage + state adapters, with the RECALL side mocked: the
recall snapshot / trim helpers (`recall_snapshot_for_consolidation`,
`recall_trim`, `recall_idle_partitions`) are monkeypatched to inject synthetic
recall rows so the build → commit-catalog → grace-bounded-trim machinery can be
driven without a live pgvector. The real recall round-trips are covered by the
integration suite (testcontainers pgvector).

Properties proven here:
  - a consolidation folds LIVE recall rows into a new shard, stamps
    `consolidated_lsn = N` and `build_type='consolidate'`, and advances the
    watermark monotonically;
  - tombstones are applied (the deleted id is removed from cold, not carried);
  - the grace-bounded trim deletes only up to the 2nd-newest shard's watermark
    (I4), never the newest;
  - build → commit → trim ordering: the trim runs AFTER the catalog commit (I2);
  - flag OFF: the consolidate handler is a clean no-op (opens no recall conn);
  - the supersede sweep still keeps the newest 2 shards.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def builder(monkeypatch):
    """Reloaded builder + state + storage bound to memory:// prefixes, recall ON.

    The recall tier is gated on `recall_enabled()` (RB_RECALL + RB_RECALL_DSN).
    We set both so `run_consolidate_once` reaches the real machinery; the recall
    STORE itself is never connected to because the snapshot/trim helpers are
    monkeypatched per test. The DSN value is a dummy — no socket is opened.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny fixtures use the flat path
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://dummy/recall")

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


def _seed_cold_shard(builder_mod, ids, dim=4, tenant="t1", dataset="ds"):
    """Build a real flat cold shard for `ids` via the builder's full-build tail."""
    import adapters.state.state as state_mod
    from adapters.landing.parquet_writer import write_parquet

    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, dim)
    vectors = np.eye(len(ids), dim, dtype=np.float32)
    records = [
        {"id": rid, "values": vectors[i].tolist(), "metadata": {"src": "cold"}}
        for i, rid in enumerate(ids)
    ]
    landing_prefix = builder_mod._landing_prefix(dataset, tenant)
    write_parquet(f"{landing_prefix}/uploads", records)
    builder_mod.run_once(dataset, tenant)
    return state_mod.get_latest_shard(tenant, dataset)


def _recall_row(rid, values, lsn, deleted=False, metadata=None):
    return {
        "id": rid,
        "values": values,
        "metadata": metadata or {"src": "recall"},
        "lsn": lsn,
        "deleted": deleted,
    }


def _patch_recall(monkeypatch, builder_mod, *, snapshot, trim_record):
    """Monkeypatch the builder's recall helpers with synthetic data.

    `snapshot` is `(max_lsn, rows)`. `trim_record` is a list the trim appends
    its `(grace_watermark, rows_remaining)` to, so a test can assert the trim
    ran (and with which watermark) AFTER the commit.
    """
    monkeypatch.setattr(
        builder_mod, "recall_snapshot_for_consolidation",
        lambda t, d: snapshot,
    )

    def _trim(t, d, grace):
        trim_record.append(grace)
        return 0

    monkeypatch.setattr(builder_mod, "recall_trim", _trim)


# --- fold live rows + watermark + build_type ------------------------------


def test_consolidate_folds_live_rows_and_stamps_watermark(builder, monkeypatch):
    """A from-scratch consolidation builds a shard from live recall rows and
    stamps consolidated_lsn=N + build_type='consolidate'."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)

    rows = [
        _recall_row("r1", [1.0, 0, 0, 0], 1),
        _recall_row("r2", [0, 1.0, 0, 0], 2),
        _recall_row("r3", [0, 0, 1.0, 0], 3),
    ]
    trims: list = []
    _patch_recall(monkeypatch, builder_mod, snapshot=(3, rows), trim_record=trims)

    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 3, "three live rows folded"

    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard is not None
    assert shard["build_type"] == "consolidate"
    assert shard["consolidated_lsn"] == 3, "watermark = max lsn N"
    assert shard["vector_count"] == 3
    # First consolidation: no 2nd-newest shard, so the grace trim watermark is 0
    # (trims nothing — its shard is the only one).
    assert trims == [0], trims

    # The folded ids are searchable in the shard's sidecar.
    from adapters.landing.parquet_reader import read_shard_sidecar
    sidecar = read_shard_sidecar(shard["shard_uri"])
    ids = {v["id"] for v in sidecar.values()}
    assert ids == {"r1", "r2", "r3"}


def test_consolidate_empty_partition_is_noop(builder, monkeypatch):
    """An empty recall partition (max_lsn=0) folds nothing and writes no shard."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)
    trims: list = []
    _patch_recall(monkeypatch, builder_mod, snapshot=(0, []), trim_record=trims)

    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 0
    assert state_mod.get_latest_shard("t1", "ds") is None, "no shard written"
    assert trims == [], "no trim on an empty partition"


# --- tombstones applied ---------------------------------------------------


def test_consolidate_applies_tombstones(builder, monkeypatch):
    """A deleted=true recall id is removed from cold and not carried forward."""
    builder_mod, state_mod, _ = builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar

    # Cold shard holds doomed + survivor.
    _seed_cold_shard(builder_mod, ["doomed", "survivor"])
    cold = state_mod.get_latest_shard("t1", "ds")
    cold_ids = {v["id"] for v in read_shard_sidecar(cold["shard_uri"]).values()}
    assert cold_ids == {"doomed", "survivor"}

    # Recall snapshot: a tombstone for `doomed` + a fresh live `new1`.
    rows = [
        _recall_row("doomed", [9.0, 0, 0, 0], 5, deleted=True),
        _recall_row("new1", [0, 0, 0, 1.0], 6),
    ]
    trims: list = []
    _patch_recall(monkeypatch, builder_mod, snapshot=(6, rows), trim_record=trims)

    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 1, "one live row folded (the tombstone is not a fold)"

    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["build_type"] == "consolidate"
    assert shard["consolidated_lsn"] == 6
    final_ids = {v["id"] for v in read_shard_sidecar(shard["shard_uri"]).values()}
    assert "doomed" not in final_ids, "tombstone removed the cold id"
    assert final_ids == {"survivor", "new1"}

    # The deleted int64 is also gone from the FAISS index itself.
    import faiss
    from adapters.storage.storage import read_bytes
    index = faiss.deserialize_index(
        np.frombuffer(read_bytes(shard["shard_uri"]), dtype=np.uint8)
    )
    live_int_ids = set(faiss.vector_to_array(index.id_map).tolist())
    assert id_to_int64("doomed") not in live_int_ids


# --- monotonic watermark + grace-bounded trim (I4) ------------------------


def test_grace_bounded_trim_uses_second_newest_watermark(builder, monkeypatch):
    """The trim watermark is the 2nd-newest shard's consolidated_lsn (I4), never
    the newest just-committed shard's."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)

    # Consolidation #1 → shard A, watermark 2.
    trims1: list = []
    _patch_recall(
        monkeypatch, builder_mod,
        snapshot=(2, [_recall_row("a", [1.0, 0, 0, 0], 1),
                      _recall_row("b", [0, 1.0, 0, 0], 2)]),
        trim_record=trims1,
    )
    builder_mod.run_consolidate_once("ds", "t1")
    assert state_mod.get_latest_shard("t1", "ds")["consolidated_lsn"] == 2
    assert trims1 == [0], "first consolidation: only one shard, grace=0"

    # Consolidation #2 → shard B, watermark 5. Now A is the 2nd-newest, so the
    # grace-bounded trim watermark is A's consolidated_lsn (2), NOT B's (5).
    trims2: list = []
    _patch_recall(
        monkeypatch, builder_mod,
        snapshot=(5, [_recall_row("c", [0, 0, 1.0, 0], 4),
                      _recall_row("d", [0, 0, 0, 1.0], 5)]),
        trim_record=trims2,
    )
    builder_mod.run_consolidate_once("ds", "t1")

    newest = state_mod.get_latest_shard("t1", "ds")
    assert newest["consolidated_lsn"] == 5, "newest watermark monotonically advanced"
    assert trims2 == [2], (
        "grace trim must use the 2nd-newest shard's watermark (2), not the "
        f"newest (5): {trims2}"
    )
    # Monotonic: watermarks across shards are non-decreasing newest-first.
    shards = state_mod.list_shards("t1", "ds")
    watermarks = [s["consolidated_lsn"] for s in shards]
    assert watermarks == sorted(watermarks, reverse=True), watermarks


def test_consolidate_keeps_newest_two_shards(builder, monkeypatch):
    """The supersede sweep still keeps the newest 2 shards after consolidation."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)

    for i in range(4):
        trims: list = []
        lsn = (i + 1) * 2
        _patch_recall(
            monkeypatch, builder_mod,
            snapshot=(lsn, [_recall_row(f"r{i}", [float(i), 0, 0, 0], lsn)]),
            trim_record=trims,
        )
        builder_mod.run_consolidate_once("ds", "t1")

    shards = state_mod.list_shards("t1", "ds")
    assert len(shards) == 2, f"sweep keeps newest 2 shards: {len(shards)}"


# --- I2 ordering: trim runs AFTER commit ----------------------------------


def test_trim_runs_after_catalog_commit(builder, monkeypatch):
    """I2: the recall trim must run STRICTLY AFTER the catalog row is committed.

    We record the order of `add_shard` (catalog commit) vs `recall_trim`. The
    trim must observe the new shard already in the catalog.
    """
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)
    # Seed a first shard so the second consolidation has a real grace trim.
    _patch_recall(
        monkeypatch, builder_mod,
        snapshot=(2, [_recall_row("a", [1.0, 0, 0, 0], 2)]),
        trim_record=[],
    )
    builder_mod.run_consolidate_once("ds", "t1")

    order: list = []
    real_add_shard = builder_mod.add_shard

    def _spy_add_shard(*a, **k):
        order.append("commit")
        return real_add_shard(*a, **k)

    def _spy_trim(t, d, grace):
        # At trim time the new shard must already be in the catalog (committed).
        order.append("trim")
        shards = state_mod.list_shards(t, d)
        assert any(s["build_type"] == "consolidate" and s["consolidated_lsn"] == 5
                   for s in shards), "trim ran before the new shard was committed"
        return 0

    monkeypatch.setattr(builder_mod, "add_shard", _spy_add_shard)
    monkeypatch.setattr(builder_mod, "recall_trim", _spy_trim)
    monkeypatch.setattr(
        builder_mod, "recall_snapshot_for_consolidation",
        lambda t, d: (5, [_recall_row("b", [0, 1.0, 0, 0], 5)]),
    )

    builder_mod.run_consolidate_once("ds", "t1")
    assert order == ["commit", "trim"], f"build→commit→trim ordering violated: {order}"


# --- flag OFF: no-op ------------------------------------------------------


def test_consolidate_flag_off_is_noop(builder, monkeypatch):
    """Flag OFF: run_consolidate_once is a clean no-op and never reads recall."""
    builder_mod, state_mod, _ = builder
    monkeypatch.delenv("RB_RECALL", raising=False)  # master switch off
    state_mod.create_dataset("t1", "ds", 4)

    def _boom(*a, **k):
        raise AssertionError("recall must not be read with the flag off")

    monkeypatch.setattr(builder_mod, "recall_snapshot_for_consolidation", _boom)

    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 0
    assert state_mod.get_latest_shard("t1", "ds") is None


# --- handler ack/nack contract --------------------------------------------


def test_handle_consolidate_acks_on_terminal(builder, monkeypatch):
    """_handle_consolidate returns True (ack) on a terminal outcome."""
    builder_mod, state_mod, _ = builder
    state_mod.create_dataset("t1", "ds", 4)
    monkeypatch.setattr(
        builder_mod, "recall_snapshot_for_consolidation",
        lambda t, d: (1, [_recall_row("r1", [1.0, 0, 0, 0], 1)]),
    )
    monkeypatch.setattr(builder_mod, "recall_trim", lambda t, d, g: 0)

    done = builder_mod._handle_consolidate({"dataset": "ds", "tenant": "t1"})
    assert done is True


def test_handle_consolidate_nacks_on_skip(builder, monkeypatch):
    """_handle_consolidate returns False (nack/redeliver) on a lock skip."""
    builder_mod, state_mod, _ = builder
    monkeypatch.setattr(
        builder_mod, "run_consolidate_once",
        lambda d, t: builder_mod.BUILD_SKIPPED,
    )
    done = builder_mod._handle_consolidate({"dataset": "ds", "tenant": "t1"})
    assert done is False
