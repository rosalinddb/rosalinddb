"""Integration test: top-K parity between RB_FAISS_MMAP=false and =true.

The contract is bit-identical results for the same shard + same query +
same nprobe regardless of how the index was loaded. This test is the
correctness safety net for the mmap rollout.

The shard is built directly with the FAISS Python API (matches what the
index builder produces — `IndexIDMap2` over `IndexFlatL2`/`IndexIVFFlat`
plus a `{shard_uri}.meta.json` sidecar) and written to the session MinIO
via the storage adapter. We then drive `_hot_search` twice — once with the
flag off, once with it on — flipping the env and reloading `v1_query` in
between so the module-level `_MMAP_ENABLED` capture is honoured.
"""
from __future__ import annotations

import importlib
import json
import os
import uuid

import faiss  # type: ignore
import numpy as np
import pytest


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_flat_shard_on_s3(tenant: str, dataset: str, prefix: str,
                            dim: int = 16, n: int = 64):
    """Build an `IndexIDMap2(IndexFlatL2)` shard, persist it to S3, register it.

    Returns `(shard_id, shard_uri, query_vec)`. Flat L2 is chosen because the
    test asserts bit-identical results — IVF training jitter is not the focus,
    and `IndexFlatL2` is supported by `read_index(..., IO_FLAG_MMAP)` in this
    FAISS build.
    """
    from adapters.storage import storage as storage_mod
    from adapters.state import state as state_mod

    rng = np.random.default_rng(20240523)
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    inner = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(inner)
    index.add_with_ids(vecs, ids)

    shard_uri = f"{prefix}/{tenant}/{dataset}/shard-{uuid.uuid4().hex[:8]}.bin"
    storage_mod.write_bytes(shard_uri, faiss.serialize_index(index).tobytes())
    sidecar = {str(int(i)): {"id": f"r{int(i)}", "metadata": {"row": int(i)}} for i in ids}
    storage_mod.write_bytes(
        f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8")
    )
    shard_id = state_mod.add_shard(
        tenant, dataset, shard_uri, "chk", n, "flat", "full", []
    )
    # Use the first vector as the query so we know there is at least one
    # exact-match neighbour (it self-scores 0.0 with L2).
    return shard_id, shard_uri, vecs[0].tolist()


def _build_ivfflat_shard_on_s3(tenant: str, dataset: str, prefix: str,
                               dim: int = 32, n: int = 512, nlist: int = 8):
    """Build an `IndexIDMap2(IndexIVFFlat)` shard, persist to S3, register it.

    Returns `(shard_id, shard_uri, query_vec)`. Trains the IVF index once
    here; both parity passes then load the SAME serialised bytes off S3,
    so the comparison is "does the load path preserve the index contract"
    — not "does IVF training produce a reproducible centroid set" (which is
    a separate, well-known FAISS question).

    Sized just above the IVF training floor (need >= ~39 * nlist samples
    for FAISS to be happy with the k-means init).
    """
    from adapters.storage import storage as storage_mod
    from adapters.state import state as state_mod

    rng = np.random.default_rng(20240601)
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    ivf.train(vecs)
    inner = faiss.IndexIDMap2(ivf)
    inner.add_with_ids(vecs, ids)

    shard_uri = f"{prefix}/{tenant}/{dataset}/shard-{uuid.uuid4().hex[:8]}.bin"
    storage_mod.write_bytes(shard_uri, faiss.serialize_index(inner).tobytes())
    sidecar = {str(int(i)): {"id": f"r{int(i)}", "metadata": {"row": int(i)}} for i in ids}
    storage_mod.write_bytes(
        f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8")
    )
    shard_id = state_mod.add_shard(
        tenant, dataset, shard_uri, "chk", n, "ivf", "full", []
    )
    return shard_id, shard_uri, vecs[0].tolist()


def _reload_v1_query():
    """Reload v1_query so the module-level _MMAP_ENABLED capture re-runs."""
    import services.query_api.v1_query as v1q
    importlib.reload(v1q)
    return v1q


def _reset_state_and_storage():
    """Drop in-memory state + the per-process shard cache before each parity run.

    The MinIO bucket itself is shared across the session by design (see
    `tests/integration/conftest.py`), so isolation is per-prefix; we clear the
    in-process state-adapter caches plus the FAISS cache so the second pass
    really does a cold load and exercises the mmap path.
    """
    from adapters.state import state as state_mod

    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()


@pytest.fixture
def parity_env(tmp_path, monkeypatch, s3_indexes_prefix):
    """Per-test CACHE_DIR + state reset, with the session MinIO prefix wired in.

    The `minio_env` autouse fixture from `tests/integration/conftest.py`
    already exports S3_* env vars and a unique `RB_TEST_INDEXES_PREFIX`; we
    just need a local cache dir and to clear in-memory state.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CACHE_DIR", str(cache))
    _reset_state_and_storage()
    return {"prefix": s3_indexes_prefix, "cache_dir": str(cache)}


def _run_query_under_flag(flag_value: str, tenant: str, dataset: str,
                          query_vec, top_k: int, monkeypatch):
    """Reload v1_query with RB_FAISS_MMAP=<flag_value> and run `_hot_search`.

    Returns `(matches, mode, mmap_enabled_observed)`. Clearing the cache up
    front ensures both passes go through the cold-load path (the second pass
    would otherwise reuse the already-deserialised non-mmap index from the
    first pass's `_cache_put`).
    """
    if flag_value is None:
        monkeypatch.delenv("RB_FAISS_MMAP", raising=False)
    else:
        monkeypatch.setenv("RB_FAISS_MMAP", flag_value)
    v1q = _reload_v1_query()
    v1q.cache_clear()
    # Bind CACHE_DIR after reload — reload re-reads the env, but the module
    # captured the value via `os.getenv("CACHE_DIR", "/var/cache/shards")`
    # at import; setenv before reload already covered that.
    out = v1q._consolidated_search(tenant, dataset, query_vec, top_k=top_k)
    assert out is not None, "no shard found — fixture wiring is wrong"
    matches, mode = out
    return matches, mode, v1q._MMAP_ENABLED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_top_k_parity_between_mmap_and_read_index(parity_env, monkeypatch):
    """Same shard, same query, same nprobe → identical top-10 results.

    Setup:
      - Real MinIO via testcontainers fixture (matches existing pattern in
        tests/integration/conftest.py)
      - Build one `IndexIDMap2(IndexFlatL2)` shard with a known seeded corpus.
        Flat-L2 is the right choice for a bit-identical parity assertion:
        the same bytes through the same arithmetic produce the same answer,
        with no IVF training jitter to muddy the comparison. The separate
        `test_ivfflat_parity_between_mmap_and_read_index` test below covers
        the production index type.
      - Run a probe query with RB_FAISS_MMAP=false, capture (ids, scores)
      - Reset cache, flip RB_FAISS_MMAP=true, run same query
      - Assert ids match exactly, scores match within float epsilon
    """
    tenant, dataset = "tparity", "dsparity"
    _shard_id, _uri, query_vec = _build_flat_shard_on_s3(
        tenant, dataset, parity_env["prefix"]
    )

    # Pass 1: flag OFF → classic `faiss.read_index(path)` path.
    matches_off, mode_off, mmap_off = _run_query_under_flag(
        "false", tenant, dataset, query_vec, top_k=10, monkeypatch=monkeypatch
    )
    assert mmap_off is False
    assert mode_off in ("hot", "cold")
    assert len(matches_off) > 0

    # Pass 2: flag ON → mmap path with IO_FLAG_MMAP | IO_FLAG_READ_ONLY.
    matches_on, mode_on, mmap_on = _run_query_under_flag(
        "true", tenant, dataset, query_vec, top_k=10, monkeypatch=monkeypatch
    )
    assert mmap_on is True
    assert mode_on in ("hot", "cold")

    # Bit-identical ids in the same order.
    ids_off = [m["id"] for m in matches_off]
    ids_on = [m["id"] for m in matches_on]
    assert ids_on == ids_off, f"id order diverged: off={ids_off} on={ids_on}"

    # Scores match exactly (flat L2 over the same bytes → same arithmetic).
    scores_off = [m["score"] for m in matches_off]
    scores_on = [m["score"] for m in matches_on]
    for a, b in zip(scores_off, scores_on):
        assert a == pytest.approx(b, abs=1e-6, rel=1e-6), (
            f"score divergence: off={scores_off} on={scores_on}"
        )

    # Metadata must round-trip identically — the sidecar load is shared code.
    meta_off = [m["metadata"] for m in matches_off]
    meta_on = [m["metadata"] for m in matches_on]
    assert meta_on == meta_off


def test_ivfflat_parity_between_mmap_and_read_index(parity_env, monkeypatch):
    """Same IVFFlat shard, same query → identical top-K under both load paths.

    The Flat-L2 parity test above proves the call-site wiring; this test
    proves the production index type (IVFFlat) also behaves identically when
    loaded via `IO_FLAG_MMAP | IO_FLAG_READ_ONLY`. Both passes load the SAME
    serialised bytes off S3, so any divergence is in the load path itself,
    not in IVF training.
    """
    tenant, dataset = "tparityivf", "dsparityivf"
    _shard_id, _uri, query_vec = _build_ivfflat_shard_on_s3(
        tenant, dataset, parity_env["prefix"]
    )

    matches_off, _, mmap_off = _run_query_under_flag(
        "false", tenant, dataset, query_vec, top_k=10, monkeypatch=monkeypatch
    )
    assert mmap_off is False
    assert len(matches_off) > 0, "IVF probe returned no matches — check nprobe"

    matches_on, _, mmap_on = _run_query_under_flag(
        "true", tenant, dataset, query_vec, top_k=10, monkeypatch=monkeypatch
    )
    assert mmap_on is True

    # ID order must be bit-identical: same serialised index, same probed
    # cells, same distance math.
    ids_off = [m["id"] for m in matches_off]
    ids_on = [m["id"] for m in matches_on]
    assert ids_on == ids_off, (
        f"IVFFlat mmap path returned a different order: off={ids_off} on={ids_on}"
    )

    scores_off = [m["score"] for m in matches_off]
    scores_on = [m["score"] for m in matches_on]
    for a, b in zip(scores_off, scores_on):
        assert a == pytest.approx(b, abs=1e-6, rel=1e-6), (
            f"IVFFlat score divergence: off={scores_off} on={scores_on}"
        )


def test_mmap_query_returns_same_status_codes(parity_env, monkeypatch):
    """Error envelopes (404 unknown dataset, 400 dim mismatch) are unchanged under mmap.

    Validation runs BEFORE any FAISS load, so the mmap flag should be a no-op
    here — the test pins that by running `execute_v1_query` under both flag
    settings and asserting the same `_err`-shaped JSONResponse comes back.
    """
    from fastapi.responses import JSONResponse

    tenant = "terr"
    # Build a real dataset so the "dim mismatch" path can actually run.
    from adapters.state import state as state_mod
    state_mod.create_tenant(tenant, f"{tenant}@example.com", "pw")
    state_mod.create_dataset(tenant, "good", 4)

    def _statuses_under(flag_value):
        monkeypatch.setenv("RB_FAISS_MMAP", flag_value)
        v1q = _reload_v1_query()
        v1q.cache_clear()
        # Pin that the flag actually reached the module — without this, the
        # test passes even before the production code lands (because validation
        # short-circuits before any FAISS load), which is not the contract.
        # Use the production `_truthy` so the test never drifts from prod
        # semantics (e.g. if `"  true  "` is added to the parametrize set).
        assert v1q._MMAP_ENABLED is v1q._truthy(flag_value)

        # 404 — unknown dataset (cross-tenant / missing collapse here).
        r_404 = v1q.execute_v1_query(
            tenant, {"dataset": "missing", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3}
        )
        assert isinstance(r_404, JSONResponse), type(r_404)

        # 400 — vector length != dataset dimension.
        r_400 = v1q.execute_v1_query(
            tenant, {"dataset": "good", "vector": [0.0, 0.0, 0.0], "top_k": 3}
        )
        assert isinstance(r_400, JSONResponse), type(r_400)

        return r_404.status_code, r_400.status_code

    off = _statuses_under("false")
    on = _statuses_under("true")
    assert on == off, f"status codes diverged: off={off} on={on}"
    assert off == (404, 400)
