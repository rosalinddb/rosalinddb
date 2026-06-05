"""Integration coverage for the delta-tier query read union (PR-C).

End-to-end through the real FastAPI `POST /v1/query` app, with:

  - the **recall tier** on a REAL `pgvector/pgvector:pg15` container
    (`RB_RECALL_DSN`), exactly like `test_query_union.py`;
  - the **consolidated (cold)** tier as a base + live-delta GENERATION whose
    FAISS shards are built DIRECTLY with faiss (bare `IndexIVFFlat` sharing one
    trained-empty quantizer — mirroring compaction-redesign.md §0) and written
    into the session MinIO with their `.meta.json` sidecars, plus catalog rows
    registered via `state.add_shard(... level=, parent_shard_id=, ...)`. This
    keeps PR-C independent of the PR-B index_builder;
  - the control plane on the `memory://` state adapter (recall is gated on
    `RB_RECALL_DSN`, not the control-plane DSN), and `RB_DELTA_TIER=1`.

Properties proven (spec §4.3, §5.1–5.3):

  - **union recall parity**: the base+delta loop-search union == a single
    monolithic index over all the vectors (same ids/ranking).
  - **tombstone suppression**: a delta's `tombstone_int_ids` hides the cold id.
  - **gap clamp**: a stale-cache missing delta clamps the frontier (no dropped
    vectors; recall re-serves the band).
  - **unreadable frontier shard → 503**: a deleted delta `.bin` fails the query.
"""
from __future__ import annotations

import importlib
import json

import faiss  # type: ignore
import numpy as np
import psycopg2
import pytest

from adapters.landing.parquet_reader import id_to_int64

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DIM = 8
NLIST = 4


@pytest.fixture(scope="module")
def recall_url():
    """One pgvector container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the delta-union suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _truncate_recall(dsn: str) -> None:
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.recall_vectors'), "
                "to_regclass('public.recall_lsn_seq')"
            )
            hv, seq = cur.fetchone()
            if hv is not None:
                cur.execute("TRUNCATE recall_vectors")
            if seq is not None:
                cur.execute("TRUNCATE recall_lsn_seq")
    finally:
        conn.close()


def _migrate_recall(recall_url):
    import os
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    os.environ["RB_RECALL_DSN"] = recall_url
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)


def _master_quantizer(train: np.ndarray) -> faiss.IndexIVFFlat:
    q = faiss.IndexFlatL2(DIM)
    m = faiss.IndexIVFFlat(q, DIM, NLIST, faiss.METRIC_L2)
    m.train(train)
    return m


def _bare_ivf_blob(master, ids, vecs) -> bytes:
    idx = faiss.clone_index(master)
    int_ids = np.array([id_to_int64(i) for i in ids], dtype=np.int64)
    idx.add_with_ids(vecs, int_ids)
    blob = faiss.serialize_index(idx)
    return blob.tobytes() if isinstance(blob, np.ndarray) else blob


def _sidecar(ids, metas) -> bytes:
    return json.dumps(
        {str(id_to_int64(i)): {"id": i, "metadata": m or {}} for i, m in zip(ids, metas)}
    ).encode("utf-8")


def _rng(n, seed):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _build_client(monkeypatch, indexes_prefix, tmp_path, recall_url):
    """Reload the pipeline with recall + delta tier ON, MinIO storage, fresh state."""
    import os
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("INDEXES_PREFIX", indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    os.environ["RB_RECALL_DSN"] = recall_url
    state_mod.migrate_recall(force=True)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()
    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app), state_mod, v1_query


def _signup(client, email="delta@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_of(client, signup):
    r = client.get("/auth/me", headers=_auth(signup["token"]))
    return r.json()["tenant"]["id"]


def _build_generation(
    storage, state_mod, tenant, dataset, indexes_prefix, *,
    tombstone_ids=None, contiguous=True,
):
    """Build base + 2 deltas into MinIO; return `(all_ids, all_vecs, all_metas)`."""
    n_base, n_d0, n_d1 = 40, 12, 12
    train = _rng(n_base + n_d0 + n_d1, seed=900)
    master = _master_quantizer(train)

    base_ids = [f"b{i}" for i in range(n_base)]
    base_vecs = _rng(n_base, seed=1)
    base_metas = [{"src": "base"} for _ in base_ids]
    base_uri = f"{indexes_prefix}/{dataset}/base.bin"
    storage.write_bytes(base_uri, _bare_ivf_blob(master, base_ids, base_vecs))
    storage.write_bytes(f"{base_uri}.meta.json", _sidecar(base_ids, base_metas))
    base_id = state_mod.add_shard(
        tenant, dataset, base_uri, checksum="x", vector_count=n_base,
        index_type="ivfflat", build_type="consolidate", consolidated_lsn=100,
        quantizer_version=1, level=0, covered_lsn_lo=0, covered_lsn_hi=100,
    )

    all_ids, all_vecs, all_metas = list(base_ids), [base_vecs], list(base_metas)
    prev_hi = 100
    for d_idx, n in enumerate((n_d0, n_d1)):
        ids = [f"d{d_idx}_{j}" for j in range(n)]
        vecs = _rng(n, seed=10 + d_idx)
        metas = [{"src": f"delta{d_idx}"} for _ in ids]
        lo = prev_hi + 1
        if not contiguous and d_idx == 1:
            lo = prev_hi + 50
        hi = lo + 9
        uri = f"{indexes_prefix}/{dataset}/delta{d_idx}.bin"
        storage.write_bytes(uri, _bare_ivf_blob(master, ids, vecs))
        storage.write_bytes(f"{uri}.meta.json", _sidecar(ids, metas))
        state_mod.add_shard(
            tenant, dataset, uri, checksum="x", vector_count=n,
            index_type="ivfflat", build_type="consolidate-delta",
            consolidated_lsn=hi, quantizer_version=1, level=1,
            parent_shard_id=base_id, covered_lsn_lo=lo, covered_lsn_hi=hi,
            tombstone_int_ids=(tombstone_ids if d_idx == 0 else None),
        )
        all_ids += ids
        all_vecs.append(vecs)
        all_metas += metas
        prev_hi = hi

    return all_ids, np.vstack(all_vecs), all_metas, master


def _monolith(master, ids, vecs, metas):
    mono = faiss.clone_index(master)
    mono.add_with_ids(vecs, np.array([id_to_int64(i) for i in ids], dtype=np.int64))
    sidecar = {str(id_to_int64(i)): {"id": i, "metadata": m or {}} for i, m in zip(ids, metas)}
    return mono, sidecar


# --- union recall parity ----------------------------------------------------


def test_delta_union_recall_parity_vs_monolith(
    monkeypatch, recall_url, s3_indexes_prefix, tmp_path
):
    """The base+delta union over MinIO shards == a single monolithic index."""
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)
    client, state_mod, v1q = _build_client(monkeypatch, s3_indexes_prefix, tmp_path, recall_url)
    s = _signup(client, email="parity@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "p", "dimension": DIM})
    tenant = _tenant_of(client, s)

    from adapters.storage import storage
    all_ids, all_vecs, all_metas, master = _build_generation(
        storage, state_mod, tenant, "p", s3_indexes_prefix
    )
    mono, mono_sidecar = _monolith(master, all_ids, all_vecs, all_metas)

    top_k = 10
    q = all_vecs[0].tolist()
    r = client.post(
        "/v1/query", headers=_auth(s["token"]),
        json={"dataset": "p", "vector": q, "top_k": top_k},
    )
    assert r.status_code == 200, r.text
    union_ids = [m["id"] for m in r.json()["matches"]]

    x = np.array([q], dtype=np.float32)
    sp, _ = v1q._ivf_search_params(mono, None)
    kwargs = {"params": sp} if sp is not None else {}
    d, i = mono.search(x, top_k, **kwargs)
    mono_ids = [m["id"] for m in v1q.map_hits_to_matches(i[0], d[0], mono_sidecar, top_k)]

    assert union_ids == mono_ids, (union_ids, mono_ids)


# --- tombstone suppression --------------------------------------------------


def test_delta_tombstone_suppresses_cold_id_e2e(
    monkeypatch, recall_url, s3_indexes_prefix, tmp_path
):
    """A delta's catalog tombstone hides the cold (base) id from the union."""
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)
    client, state_mod, _v1q = _build_client(monkeypatch, s3_indexes_prefix, tmp_path, recall_url)
    s = _signup(client, email="tomb@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "tb", "dimension": DIM})
    tenant = _tenant_of(client, s)

    from adapters.storage import storage
    all_ids, all_vecs, all_metas, _ = _build_generation(
        storage, state_mod, tenant, "tb", s3_indexes_prefix,
        tombstone_ids=[id_to_int64("b0")],
    )
    r = client.post(
        "/v1/query", headers=_auth(s["token"]),
        json={"dataset": "tb", "vector": all_vecs[0].tolist(), "top_k": 10},
    )
    assert r.status_code == 200, r.text
    ids = [m["id"] for m in r.json()["matches"]]
    assert "b0" not in ids, "the delta cold-vs-cold tombstone must suppress b0"
    assert len(ids) > 0


# --- gap clamp (stale-cache missing delta) ----------------------------------


def test_delta_gap_clamps_frontier(
    monkeypatch, recall_url, s3_indexes_prefix, tmp_path
):
    """A non-contiguous delta band clamps the frontier (recall re-serves the band)."""
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)
    client, state_mod, v1q = _build_client(monkeypatch, s3_indexes_prefix, tmp_path, recall_url)
    s = _signup(client, email="gap@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "g", "dimension": DIM})
    tenant = _tenant_of(client, s)

    from adapters.storage import storage
    _build_generation(
        storage, state_mod, tenant, "g", s3_indexes_prefix, contiguous=False
    )
    out = v1q._resolve_shard(tenant, "g", {})
    # base hi=100, delta0 [101,110], delta1 [160,169] → frontier clamps to 110.
    assert v1q._frontier_watermark(out) == 110

    # The query still succeeds (over-serving from recall is safe).
    r = client.post(
        "/v1/query", headers=_auth(s["token"]),
        json={"dataset": "g", "vector": [0.0] * DIM, "top_k": 5},
    )
    assert r.status_code == 200, r.text


# --- unreadable frontier shard → 503 ----------------------------------------


def test_unreadable_frontier_delta_returns_503(
    monkeypatch, recall_url, s3_indexes_prefix, tmp_path
):
    """A deleted delta `.bin` fails the query (no silent narrowing of the cold set)."""
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)
    client, state_mod, _v1q = _build_client(monkeypatch, s3_indexes_prefix, tmp_path, recall_url)
    s = _signup(client, email="503@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "x", "dimension": DIM})
    tenant = _tenant_of(client, s)

    from adapters.storage import storage
    all_ids, all_vecs, all_metas, _ = _build_generation(
        storage, state_mod, tenant, "x", s3_indexes_prefix
    )
    # Delete the delta0 object from MinIO to make a frontier shard unreadable.
    from adapters.storage.storage import _split_s3, _s3_client
    bucket, key = _split_s3(f"{s3_indexes_prefix}/x/delta0.bin")
    _s3_client().delete_object(Bucket=bucket, Key=key)

    r = client.post(
        "/v1/query", headers=_auth(s["token"]),
        json={"dataset": "x", "vector": all_vecs[0].tolist(), "top_k": 10},
    )
    assert r.status_code == 503, r.text
