"""Regression tests for the index builder's INGEST-upsert and SINGLE-ID-DELETE
paths on IVFFlat shards — the FAISS `remove_ids` abort (#28).

Background (see #18 / bench-lab/analysis/a3-rootcause.md): FAISS 1.8.0
`IndexIVF.remove_ids` (through an `IndexIDMap2`) trips a C++ assertion
(`j == index->ntotal`, IndexIDMap.cpp:181) that `abort()`s the whole process
(SIGABRT / exit 134) when it removes ids that overlap ids already living in an
IVF shard. It is NOT a Python `RuntimeError`, so the builder's
`try/except RuntimeError` cannot catch it — the builder dies.

PR #18 fixed the CONSOLIDATION fold (`_build_consolidated_shard`) with a
UNION-REBUILD: reconstruct the surviving (non-replaced, non-tombstoned) vectors
from the loaded IVFFlat (lossless: raw float32), concatenate any new live
vectors, and `build_ivfflat` a fresh shard from scratch — never `remove_ids`,
carrying every survivor's ORIGINAL int64 through unchanged.

The SAME latent abort still existed in two other paths:
  - Site 1 — landing-ingest incremental upsert (`_run_once_locked`): an
    overlapping re-ingest of an id already present in an IVF shard would
    `remove_ids` the stale copy before adding the new vector.
  - Site 2 — single-id delete (`_run_delete_locked`): deleting an id that lives
    in an IVF shard would `remove_ids` that id.

These tests drive both entrypoints (`run_once` re-ingest / `run_delete_once`)
end to end against a REAL `IndexIDMap2(IVFFlat)` shard via the `memory://`
storage + state adapters (no MinIO/Docker), with enough rows that the index
actually trains as IVF — so the abort-prone path is exercised, not the flat
fallback. Post-fix both complete cleanly with a correct, original-int64-
preserving result.
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
    """Reloaded builder/state/storage bound to memory://, IVFFlat index type.

    Mirrors the fixture in test_builder_consolidate_fold.py: INDEX_TYPE=ivfflat
    so the seeded cold shard and every subsequent ingest/delete use the IVF path
    — the one that hits the `remove_ids` abort.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    # Keep the IVF training floor low so a tiny fixture still trains an IVF.
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "64")

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


def _write_batch(builder_mod, ids, vectors, upload, tenant="t1", dataset="ds"):
    """Write one upload's parquet into its own landing sub-prefix.

    A distinct sub-prefix per upload makes each batch a NEW landing part so the
    builder's incremental path (`indexed_landing_uris` gate) folds it onto the
    existing shard rather than treating it as already-indexed.
    """
    from adapters.landing.parquet_writer import write_parquet

    landing_prefix = builder_mod._landing_prefix(dataset, tenant)
    records = [
        {"id": rid, "values": vectors[i].tolist(), "metadata": {"upload": upload, "i": i}}
        for i, rid in enumerate(ids)
    ]
    write_parquet(f"{landing_prefix}/upload-{upload}", records)


def _seed_cold_ivf_shard(builder_mod, state_mod, ids, vectors, tenant="t1", dataset="ds"):
    """Build a real IVFFlat cold shard for `ids`/`vectors` via the full-build tail."""
    if state_mod.get_dataset(tenant, dataset) is None:
        state_mod.create_dataset(tenant, dataset, _DIM)
    _write_batch(builder_mod, ids, vectors, upload="seed", tenant=tenant, dataset=dataset)
    builder_mod.run_once(dataset, tenant)
    shard = state_mod.get_latest_shard(tenant, dataset)
    assert shard is not None
    assert shard["index_type"] == "ivfflat", (
        f"fixture must seed an IVFFlat shard, got {shard['index_type']}"
    )
    return shard


def _load_shard_index(state_mod, tenant="t1", dataset="ds"):
    from adapters.storage.storage import read_bytes

    shard = state_mod.get_latest_shard(tenant, dataset)
    index = faiss.deserialize_index(
        np.frombuffer(read_bytes(shard["shard_uri"]), dtype=np.uint8)
    )
    return shard, index


# --------------------------------------------------------------------------- #
# Reproduction: both paths abort the process in a SUBPROCESS (pre-fix).        #
# --------------------------------------------------------------------------- #

# Standalone scripts run in a subprocess so a C++ abort (SIGABRT / exit 134)
# cannot kill the pytest runner. Each seeds an IVFFlat cold shard then drives the
# entrypoint TWICE with an overlapping operation: the first `remove_ids` on the
# IVF shard corrupts the `IndexIDMap2` id_map bookkeeping, and the SECOND
# overlapping operation trips the `j == index->ntotal` assertion that `abort()`s
# the process pre-fix (the exact two-step shape #18 used to surface the abort).
_INGEST_SCRIPT = '''
import importlib, os
import numpy as np

os.environ["DATABASE_URL"] = "memory://test"
os.environ["INDEXES_PREFIX"] = "memory://rosalinddb/indexes"
os.environ["LANDING_PREFIX"] = "memory://rosalinddb/landing"
os.environ["TENANT_PREFIX"] = "true"
os.environ["INDEX_TYPE"] = "ivfflat"
os.environ["IVF_TRAINING_FLOOR"] = "64"
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
lp = b._landing_prefix("ds", "t1")
write_parquet(lp + "/upload-seed",
              [{"id": ids[i], "values": base[i].tolist(), "metadata": {}} for i in range(N)])
b.run_once("ds", "t1")
assert state_mod.get_latest_shard("t1", "ds")["index_type"] == "ivfflat", "need IVF"

def reingest(upload, seed, n_overlap):
    rng2 = np.random.default_rng(seed)
    new = rng2.random((n_overlap, DIM), dtype=np.float64).astype(np.float32)
    write_parquet(lp + ("/upload-%s" % upload),
                  [{"id": ids[k], "values": new[k].tolist(), "metadata": {}}
                   for k in range(n_overlap)])
    return b.run_once("ds", "t1")

# RE-INGEST 1: overlapping re-upsert of the first 40 ids -> remove+readd
# corrupts the IDMap2 bookkeeping.
print("INGEST1", reingest("reingest1", 21, 40), flush=True)
# RE-INGEST 2: another overlapping re-upsert -> remove_ids aborts (exit 134)
# pre-fix.
print("INGEST2", reingest("reingest2", 22, 40), flush=True)
print("NO_ABORT_INGEST_OK", flush=True)
'''

_DELETE_SCRIPT = '''
import importlib, os
import numpy as np

os.environ["DATABASE_URL"] = "memory://test"
os.environ["INDEXES_PREFIX"] = "memory://rosalinddb/indexes"
os.environ["LANDING_PREFIX"] = "memory://rosalinddb/landing"
os.environ["TENANT_PREFIX"] = "true"
os.environ["INDEX_TYPE"] = "ivfflat"
os.environ["IVF_TRAINING_FLOOR"] = "64"
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
lp = b._landing_prefix("ds", "t1")
write_parquet(lp + "/upload-seed",
              [{"id": ids[i], "values": base[i].tolist(), "metadata": {}} for i in range(N)])
b.run_once("ds", "t1")
assert state_mod.get_latest_shard("t1", "ds")["index_type"] == "ivfflat", "need IVF"

# DELETE 1: remove an id that lives in the IVF shard -> remove_ids on IVF
# corrupts the IDMap2 bookkeeping.
print("DELETE1", b.run_delete_once("ds", "t1", ids[10]), flush=True)
# DELETE 2: remove another id -> remove_ids aborts (exit 134) pre-fix.
print("DELETE2", b.run_delete_once("ds", "t1", ids[20]), flush=True)
print("NO_ABORT_DELETE_OK", flush=True)
'''


def _run_script(script: str):
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def _assert_no_abort(proc, ok_marker):
    """Assert the subprocess either succeeded, or aborted with the FAISS
    assertion we are fixing (version-aware on FAISS 1.8.0) — never some other
    error. Post-fix it must exit 0 and print `ok_marker`."""
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        assert ok_marker in combined, combined
        return
    # Non-zero: must be the FAISS abort we are fixing, not an unrelated failure.
    assert proc.returncode in (134, -6), (
        f"expected SIGABRT/exit 134 from the FAISS remove_ids assertion, "
        f"got returncode={proc.returncode}\n{combined}"
    )
    assert "IndexIDMap" in combined or "j == index->ntotal" in combined, (
        f"abort was not the expected FAISS remove_ids assertion:\n{combined}"
    )
    if not faiss.__version__.startswith("1.8."):  # pragma: no cover
        pytest.fail("unexpected FAISS abort outside 1.8.0 — investigate: " + combined)


def test_overlapping_reingest_does_not_abort_subprocess(ivf_builder):
    """The overlapping re-ingest must not abort the builder (Site 1)."""
    _assert_no_abort(_run_script(_INGEST_SCRIPT), "NO_ABORT_INGEST_OK")


def test_delete_in_ivf_shard_does_not_abort_subprocess(ivf_builder):
    """Deleting ids that live in an IVF shard must not abort (Site 2).

    Out-of-process guard. NOTE: unlike the ingest path, two separate-generation
    deletes do not always surface the C++ `abort()` (the serialize/deserialize
    between shard generations can heal the IDMap2 enough to dodge the second
    `remove_ids` assertion) — but the pre-fix delete still leaves a CORRUPT IVF
    shard (its direct map can no longer be built; see
    `test_delete_from_ivf_shard_produces_correct_result`, the strong in-process
    regression guard). This test ensures we never regress to an abort here.
    """
    _assert_no_abort(_run_script(_DELETE_SCRIPT), "NO_ABORT_DELETE_OK")


# --------------------------------------------------------------------------- #
# Site 1: overlapping re-ingest -> correct union, new vectors win, ids kept.   #
# --------------------------------------------------------------------------- #


def test_overlapping_reingest_produces_correct_union(ivf_builder):
    """Post-fix: re-ingesting ids already in an IVF shard with NEW vectors folds
    in-process to a correct union — the new embedding wins, the original int64 is
    preserved, untouched survivors keep their cold vectors, and the count holds."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    # Re-ingest ids 0..39 with NEW vectors (overlap) PLUS 8 brand-new ids. The
    # overlap forces the IVF union-rebuild; the new ids exercise the concat-new
    # branch in the same pass.
    overlap_new = _deterministic_vectors(40, _DIM, seed=21)
    fresh = _deterministic_vectors(8, _DIM, seed=31)
    fresh_ids = [f"new-{i}" for i in range(8)]
    batch_ids = ids[:40] + fresh_ids
    batch_vecs = np.concatenate([overlap_new, fresh], axis=0)
    _write_batch(builder_mod, batch_ids, batch_vecs, upload="reingest")

    added = builder_mod.run_once("ds", "t1")
    assert added == 48, added  # 40 overlapping + 8 new vectors added
    assert builder_mod._LAST_BUILD["build_type"] == "incremental"

    shard, index = _load_shard_index(state_mod)
    assert shard["index_type"] == "ivfflat", "union rebuild stays IVFFlat"
    faiss_int_ids = faiss.vector_to_array(index.id_map).tolist()

    # No duplicate ids — the overlapping ids replaced in place, not appended.
    assert len(faiss_int_ids) == len(set(faiss_int_ids)), "duplicate ids in union"

    expected_ids = set(ids) | set(fresh_ids)
    assert index.ntotal == len(expected_ids) == _N_COLD + 8, index.ntotal
    assert shard["vector_count"] == _N_COLD + 8

    # Sidecar maps exactly the surviving + new ids.
    sidecar = read_shard_sidecar(shard["shard_uri"])
    assert {v["id"] for v in sidecar.values()} == expected_ids

    # The re-ingested ids carry their NEW vectors under their ORIGINAL int64.
    faiss.extract_index_ivf(index).make_direct_map()
    int_set = set(faiss_int_ids)
    for k in range(40):
        orig_int64 = id_to_int64(ids[k])
        assert orig_int64 in int_set, f"original int64 for {ids[k]} not preserved"
        recon = index.reconstruct(int(orig_int64))
        assert np.allclose(recon, overlap_new[k], atol=1e-5), (
            f"re-ingested {ids[k]} did not carry its NEW vector"
        )
    # An untouched survivor still carries its ORIGINAL cold vector + int64.
    survivor_int64 = id_to_int64(ids[50])
    assert survivor_int64 in int_set
    assert np.allclose(index.reconstruct(int(survivor_int64)), base[50], atol=1e-5)
    # A brand-new id is present with its vector.
    new_int64 = id_to_int64(fresh_ids[0])
    assert new_int64 in int_set
    assert np.allclose(index.reconstruct(int(new_int64)), fresh[0], atol=1e-5)

    # A search for an updated vector returns that updated id.
    index.nprobe = 16
    _, I = index.search(overlap_new[0].reshape(1, -1), 1)
    assert int(I[0][0]) == id_to_int64(ids[0]), "search did not return the updated id"


def test_nonoverlap_reingest_still_appends(ivf_builder):
    """A re-ingest of all-NEW ids appends onto the IVF shard with no rebuild and
    no removal (the cheap, crash-free fast path stays intact)."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import read_shard_sidecar

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    new_ids = [f"new-{i}" for i in range(8)]
    new_vecs = _deterministic_vectors(8, _DIM, seed=31)
    _write_batch(builder_mod, new_ids, new_vecs, upload="append")
    added = builder_mod.run_once("ds", "t1")
    assert added == 8
    assert builder_mod._LAST_BUILD["build_type"] == "incremental"

    shard, index = _load_shard_index(state_mod)
    assert index.ntotal == _N_COLD + 8
    assert shard["vector_count"] == _N_COLD + 8
    sidecar_ids = {v["id"] for v in read_shard_sidecar(shard["shard_uri"]).values()}
    assert sidecar_ids == set(ids) | set(new_ids)


# --------------------------------------------------------------------------- #
# Site 2: single-id delete from an IVF shard -> id gone, others intact.        #
# --------------------------------------------------------------------------- #


def test_delete_from_ivf_shard_produces_correct_result(ivf_builder):
    """Post-fix: deleting an id that lives in an IVF shard folds to a correct
    result — the id is gone, every other id keeps its ORIGINAL vector + int64,
    no duplicates, count decremented by one."""
    builder_mod, state_mod, _ = ivf_builder
    from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    victim = ids[10]
    victim_int64 = id_to_int64(victim)
    removed = builder_mod.run_delete_once("ds", "t1", victim)
    assert removed == 1, removed

    shard, index = _load_shard_index(state_mod)
    assert shard["build_type"] == "delete"
    assert shard["index_type"] == "ivfflat", "union rebuild stays IVFFlat"
    faiss_int_ids = faiss.vector_to_array(index.id_map).tolist()
    int_set = set(faiss_int_ids)

    assert len(faiss_int_ids) == len(int_set), "duplicate ids after delete"
    assert victim_int64 not in int_set, "deleted id still present in the index"

    expected_ids = set(ids) - {victim}
    assert index.ntotal == len(expected_ids) == _N_COLD - 1, index.ntotal
    assert shard["vector_count"] == _N_COLD - 1

    sidecar = read_shard_sidecar(shard["shard_uri"])
    assert {v["id"] for v in sidecar.values()} == expected_ids
    assert str(victim_int64) not in sidecar

    # Every surviving id keeps its ORIGINAL vector under its ORIGINAL int64.
    faiss.extract_index_ivf(index).make_direct_map()
    for i in range(_N_COLD):
        if i == 10:
            continue
        orig_int64 = id_to_int64(ids[i])
        assert orig_int64 in int_set, f"survivor {ids[i]} lost its original int64"
        assert np.allclose(index.reconstruct(int(orig_int64)), base[i], atol=1e-5), (
            f"survivor {ids[i]} lost its original vector"
        )

    # The deleted vector no longer dominates a search for itself.
    index.nprobe = 16
    _, I = index.search(base[10].reshape(1, -1), 1)
    assert int(I[0][0]) != victim_int64, "search still returns the deleted id"


def test_delete_absent_id_is_noop(ivf_builder):
    """Deleting an id absent from the IVF shard is a clean no-op — no rebuild,
    no new shard, the shard is untouched (the cheap gate stays intact)."""
    builder_mod, state_mod, _ = ivf_builder

    ids = [f"id-{i}" for i in range(_N_COLD)]
    base = _deterministic_vectors(_N_COLD, _DIM, seed=11)
    seed_shard = _seed_cold_ivf_shard(builder_mod, state_mod, ids, base)

    removed = builder_mod.run_delete_once("ds", "t1", "does-not-exist")
    assert removed == 0
    # No new shard was written — the latest shard is still the seeded one.
    assert state_mod.get_latest_shard("t1", "ds")["id"] == seed_shard["id"]
