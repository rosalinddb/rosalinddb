"""Regression tests for the index builder's INCREMENTAL consolidation fold on
IVFFlat shards — the FAISS `remove_ids` abort (#18).

Background (see bench-lab/analysis/a3-rootcause.md): the incremental consolidation
path used to call `faiss.IndexIDMap2.remove_ids()` on an IVFFlat-backed shard to
drop the stale copies of re-upserted ids before re-adding the new ones. In FAISS
1.8.0 that removal corrupts the `IndexIDMap2` id_map bookkeeping; a *subsequent*
overlapping fold trips a C++ assertion

    Faiss assertion 'j == index->ntotal' failed ... IndexIDMap.cpp:181

which `abort()`s the whole process (SIGABRT / exit 134). It is NOT a Python
`RuntimeError`, so the builder's `try/except RuntimeError` cannot catch it — the
builder dies and the consolidation watermark freezes forever.

The fix: on the incremental path, when the overlap set is non-empty on an IVF
shard, DO NOT call `remove_ids`. Instead REBUILD the shard as a union via the
crash-free from-scratch path — reconstruct the surviving (non-replaced,
non-tombstoned) vectors from the loaded IVFFlat (lossless: raw float32) and
concatenate them with the new live vectors, then `build_ivfflat` a fresh shard.

These tests:
  - `test_incremental_overlap_fold_aborts_before_fix_subprocess` PROVES the crash:
    it drives the exact two-fold overlapping consolidation in a SUBPROCESS (because
    a C++ abort would otherwise kill the pytest runner) and, when run against the
    OLD code, asserts exit 134. After the fix it asserts a clean exit 0. The body
    is version-aware: it only insists on exit-134-before-fix on FAISS 1.8.0 where
    the abort is present.
  - `test_incremental_overlap_fold_produces_correct_union` runs the same fold
    IN-PROCESS (post-fix it no longer aborts) and asserts a correct union: updated
    ids carry their NEW vectors, removed/tombstoned ids are gone, no duplicate ids,
    vector_count correct, sidecar consistent, and a search returns the updated row.
  - `test_incremental_nonoverlap_fold_still_appends` keeps the cheap append-only
    incremental path working (no overlap → no rebuild).
  - `test_from_scratch_ivf_fold_unchanged` keeps the from-scratch IVF path working.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

import faiss  # type: ignore


# Enough rows to actually train an IVF (IVF_TRAINING_FLOOR=64, nlist>=4) but tiny.
_DIM = 16
_N_COLD = 96


@pytest.fixture
def ivf_builder(monkeypatch):
    """Reloaded builder/state/storage bound to memory://, recall ON, IVFFlat.

    Unlike the flat fixture in test_builder_consolidate.py this sets
    INDEX_TYPE=ivfflat so the seeded cold shard and every fold use the IVF path —
    the one that hits the `remove_ids` abort.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    # Keep the IVF training floor low so a tiny fixture still trains an IVF.
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")
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


def _deterministic_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float64).astype(np.float32)


def _seed_cold_ivf_shard(builder_mod, state_mod, ids, vectors, tenant="t1", dataset="ds"):
    """Build a real IVFFlat cold shard for `ids`/`vectors` via the full-build tail."""
    from adapters.landing.parquet_writer import write_parquet

    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, _DIM)
    records = [
        {"id": rid, "values": vectors[i].tolist(), "metadata": {"src": "cold", "v": 0}}
        for i, rid in enumerate(ids)
    ]
    landing_prefix = builder_mod._landing_prefix(dataset, tenant)
    write_parquet(f"{landing_prefix}/uploads", records)
    builder_mod.run_once(dataset, tenant)
    shard = state_mod.get_latest_shard(tenant, dataset)
    assert shard is not None
    assert shard["index_type"] == "ivfflat", (
        f"fixture must seed an IVFFlat shard, got {shard['index_type']}"
    )
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


# --------------------------------------------------------------------------- #
# 1. Reproduction: the overlapping incremental fold aborts the process (#18). #
# --------------------------------------------------------------------------- #

# The two-fold overlapping consolidation, written as a standalone script so we can
# run it in a subprocess and observe the C++ abort (exit 134) without it killing
# the pytest runner. Mirrors `_build_consolidated_shard`'s incremental path exactly:
# seed an IVFFlat cold shard, then consolidate twice with overlapping recall ids.
_FOLD_SCRIPT = '''
import importlib, os, sys
import numpy as np

os.environ["DATABASE_URL"] = "memory://test"
os.environ["INDEXES_PREFIX"] = "memory://rosalinddb/indexes"
os.environ["LANDING_PREFIX"] = "memory://rosalinddb/landing"
os.environ["TENANT_PREFIX"] = "true"
os.environ["INDEX_TYPE"] = "ivfflat"
os.environ["IVF_TRAINING_FLOOR"] = "64"
os.environ["RB_RECALL"] = "true"
os.environ["RB_RECALL_DSN"] = "postgresql://dummy/recall"
os.environ["OTEL_SDK_DISABLED"] = "true"

import adapters.storage.storage as storage_mod
importlib.reload(storage_mod); storage_mod._MEM_OBJECTS.clear()
import adapters.state.state as state_mod
importlib.reload(state_mod)
state_mod._MEM_SHARDS.clear(); state_mod._MEM_SHARD_ID = 0; state_mod._MEM_DATASETS.clear()
import services.index_builder.run as b
importlib.reload(b)
from adapters.landing.parquet_writer import write_parquet

DIM, N = 16, 96
rng = np.random.default_rng(11)
base = rng.random((N, DIM), dtype=np.float64).astype(np.float32)
ids = ["id-%d" % i for i in range(N)]

state_mod.create_dataset("t1", "ds", DIM)
recs = [{"id": ids[i], "values": base[i].tolist(), "metadata": {"v": 0}} for i in range(N)]
write_parquet(b._landing_prefix("ds", "t1") + "/uploads", recs)
b.run_once("ds", "t1")
assert state_mod.get_latest_shard("t1", "ds")["index_type"] == "ivfflat", "need IVF"

def fold(lsn, n_overlap, seed):
    rng2 = np.random.default_rng(seed)
    rows = []
    for k in range(n_overlap):
        rows.append({"id": ids[k],
                     "values": rng2.random(DIM, dtype=np.float64).astype(np.float32).tolist(),
                     "metadata": {"v": lsn}, "lsn": lsn, "deleted": False})
    b.recall_snapshot_for_consolidation = lambda t, d: (lsn, rows)
    b.recall_trim = lambda t, d, g: 0
    return b.run_consolidate_once("ds", "t1")

# FOLD 1: overlapping re-upsert of the first 40 ids -> remove+readd corrupts IDMap2.
print("FOLD1", fold(100, 40, 21), flush=True)
# FOLD 2: another overlapping re-upsert -> remove_ids aborts (exit 134) pre-fix.
print("FOLD2", fold(200, 40, 22), flush=True)
print("NO_ABORT_BOTH_FOLDS_OK", flush=True)
'''


def _run_fold_subprocess():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    return subprocess.run(
        [sys.executable, "-c", _FOLD_SCRIPT],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_incremental_overlap_fold_aborts_before_fix_subprocess(ivf_builder):
    """Drive the two-fold overlapping IVF consolidation in a SUBPROCESS.

    The bug is a C++ `abort()` (SIGABRT / exit 134) that would kill the pytest
    runner, so it must be observed out-of-process. On the OLD code this exits 134
    with the `j == index->ntotal` / IndexIDMap.cpp assertion in stderr. After the
    union-rebuild fix it exits 0 and prints `NO_ABORT_BOTH_FOLDS_OK`.

    The assertion is version-aware: the abort only exists in FAISS 1.8.0. On a
    FAISS build where `remove_ids` does not abort, we simply require a clean exit
    (the fix must never regress it).
    """
    proc = _run_fold_subprocess()
    combined = (proc.stdout or "") + (proc.stderr or "")

    faiss_18 = faiss.__version__.startswith("1.8.")
    if proc.returncode == 0:
        # Post-fix (or a FAISS build without the abort): both folds completed.
        assert "NO_ABORT_BOTH_FOLDS_OK" in combined, combined
        return

    # Non-zero: this must be the FAISS abort we are fixing, not some other error.
    assert proc.returncode in (134, -6), (
        f"expected SIGABRT/exit 134 from the FAISS remove_ids assertion, "
        f"got returncode={proc.returncode}\n{combined}"
    )
    assert "IndexIDMap" in combined or "j == index->ntotal" in combined, (
        f"abort was not the expected FAISS remove_ids assertion:\n{combined}"
    )
    if not faiss_18:  # pragma: no cover - documents the version dependency
        pytest.fail(
            "unexpected FAISS abort outside 1.8.0 — investigate: " + combined
        )


# --------------------------------------------------------------------------- #
# 2. After the fix: the overlapping fold produces a correct UNION shard.       #
# --------------------------------------------------------------------------- #


def test_incremental_overlap_fold_produces_correct_union(ivf_builder, monkeypatch):
    """Post-fix: the overlapping incremental IVF fold completes IN-PROCESS and
    yields a correct union (updated vectors, no dup ids, tombstones gone)."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar
    from adapters.storage.storage import read_bytes

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    # FOLD 1: re-upsert ids 0..39 with NEW vectors. This corrupts IDMap2 state on
    # the old code; the union-rebuild sidesteps it.
    new1 = _deterministic_vectors(40, _DIM, seed=21)
    rows1 = [
        _recall_row(ids[k], new1[k].tolist(), 100, metadata={"v": 100})
        for k in range(40)
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=100, rows=rows1)
    n1 = builder_mod.run_consolidate_once("ds", "t1")
    assert n1 == 40

    # FOLD 2: re-upsert ids 0..39 AGAIN with yet newer vectors + tombstone id-95.
    # On the old code FOLD 2's remove_ids aborts; here it must complete.
    new2 = _deterministic_vectors(40, _DIM, seed=22)
    rows2 = [
        _recall_row(ids[k], new2[k].tolist(), 200 + k, metadata={"v": 200})
        for k in range(40)
    ]
    rows2.append(_recall_row(ids[95], base[95].tolist(), 250, deleted=True))
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=300, rows=rows2)
    n2 = builder_mod.run_consolidate_once("ds", "t1")
    assert n2 == 40, "40 live re-upserts folded (the tombstone is not a fold)"

    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["build_type"] == "consolidate"
    assert shard["consolidated_lsn"] == 300
    assert shard["index_type"] == "ivfflat", "union rebuild stays IVFFlat"

    index = faiss.deserialize_index(
        np.frombuffer(read_bytes(shard["shard_uri"]), dtype=np.uint8)
    )
    faiss_int_ids = faiss.vector_to_array(index.id_map).tolist()

    # No duplicate ids in the FAISS index.
    assert len(faiss_int_ids) == len(set(faiss_int_ids)), "duplicate ids in union"

    # Correct count: 96 cold - 1 tombstone = 95 (the 40 re-upserts replace in place).
    expected_ids = {ids[i] for i in range(_N_COLD)} - {ids[95]}
    assert index.ntotal == len(expected_ids) == 95, index.ntotal
    assert shard["vector_count"] == 95

    # Sidecar consistent: maps exactly the surviving ids, tombstone gone.
    sidecar = read_shard_sidecar(shard["shard_uri"])
    sidecar_ids = {v["id"] for v in sidecar.values()}
    assert sidecar_ids == expected_ids
    assert ids[95] not in sidecar_ids
    assert str(id_to_int64(ids[95])) not in sidecar

    # Tombstoned id is gone from the FAISS index too.
    assert id_to_int64(ids[95]) not in set(faiss_int_ids)

    # Updated ids carry their NEW (FOLD 2) vectors — reconstruct & compare.
    faiss.extract_index_ivf(index).make_direct_map()
    for k in range(40):
        recon = index.reconstruct(int(id_to_int64(ids[k])))
        assert np.allclose(recon, new2[k], atol=1e-5), (
            f"id {ids[k]} did not carry its FOLD-2 vector"
        )
    # An untouched survivor still carries its ORIGINAL cold vector.
    recon_survivor = index.reconstruct(int(id_to_int64(ids[50])))
    assert np.allclose(recon_survivor, base[50], atol=1e-5)

    # A search for an updated vector returns that updated id.
    index.nprobe = 16
    D, I = index.search(new2[0].reshape(1, -1), 1)
    assert int(I[0][0]) == id_to_int64(ids[0]), "search did not return the updated id"


# --------------------------------------------------------------------------- #
# 3. Non-overlap incremental path must still work (cheap append, no rebuild).  #
# --------------------------------------------------------------------------- #


def test_incremental_nonoverlap_fold_still_appends(ivf_builder, monkeypatch):
    """A fold whose recall ids are all NEW appends onto the existing shard with
    no rebuild and no removal (the common, crash-free fast path)."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar
    from adapters.storage.storage import read_bytes

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    # Brand-new ids only — no overlap with the cold shard.
    new_ids = [f"new-{i}" for i in range(8)]
    new_vecs = _deterministic_vectors(8, _DIM, seed=31)
    rows = [
        _recall_row(new_ids[i], new_vecs[i].tolist(), 100 + i) for i in range(8)
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=108, rows=rows)
    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == 8

    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["consolidated_lsn"] == 108
    assert shard["vector_count"] == _N_COLD + 8

    index = faiss.deserialize_index(
        np.frombuffer(read_bytes(shard["shard_uri"]), dtype=np.uint8)
    )
    assert index.ntotal == _N_COLD + 8
    sidecar_ids = {v["id"] for v in read_shard_sidecar(shard["shard_uri"]).values()}
    assert sidecar_ids == set(ids) | set(new_ids)


# --------------------------------------------------------------------------- #
# 4. From-scratch IVF fold unchanged.                                         #
# --------------------------------------------------------------------------- #


def test_from_scratch_ivf_fold_unchanged(ivf_builder, monkeypatch):
    """A first consolidation (no prior shard) trains a fresh IVFFlat over the live
    recall rows — the crash-free from-scratch path, unchanged by the fix."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import read_shard_sidecar

    state_mod.create_dataset("t1", "ds", _DIM)
    vecs = _deterministic_vectors(_N_COLD, _DIM, seed=41)
    rows = [
        _recall_row(f"r{i}", vecs[i].tolist(), i + 1) for i in range(_N_COLD)
    ]
    _patch_snapshot(monkeypatch, builder_mod, max_lsn=_N_COLD, rows=rows)

    n = builder_mod.run_consolidate_once("ds", "t1")
    assert n == _N_COLD

    shard = state_mod.get_latest_shard("t1", "ds")
    assert shard["build_type"] == "consolidate"
    assert shard["index_type"] == "ivfflat"
    assert shard["consolidated_lsn"] == _N_COLD
    assert shard["vector_count"] == _N_COLD
    sidecar_ids = {v["id"] for v in read_shard_sidecar(shard["shard_uri"]).values()}
    assert sidecar_ids == {f"r{i}" for i in range(_N_COLD)}
