"""End-to-end proof of read-your-writes through the mem0 RosalindDB adapter.

Stands up a **recall-enabled** RosalindDB entirely in-process and serves it over
a REAL HTTP socket (uvicorn on a NON-default port — never :8080), so the actual
REST client + mem0 adapter talk to it over the wire:

  - the **recall tier** on a real ``pgvector/pgvector:pg15`` testcontainer
    (``RB_RECALL_DSN``), with the recall migrations applied;
  - the **consolidated (cold)** object store on the session MinIO container;
  - the control plane on the ``memory://`` state adapter (the recall path is
    gated on ``RB_RECALL_DSN``, not the control-plane DSN);
  - ``RB_RECALL=true`` so writes are synchronous + immediately queryable.

It then drives the **mem0 adapter** (which drives the **REST client**) to prove:

  - insert -> immediate search returns the just-written vector (read-your-writes);
  - get returns it; list returns it; delete -> get is None (read-your-deletes)
    and search no longer returns it.

CI safety: gated by ``importorskip("mem0")`` so the core integration CI (no
``mem0ai``) skips it. Requires Docker (testcontainers), like the rest of the
integration suite. Runs on a NON-default port so it never disturbs a stack on
:8080.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import threading
import time

import pytest

pytest.importorskip("mem0")  # optional dep — skip without mem0ai installed

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# Put the adapter dir on sys.path so the REAL client + adapter import.
_ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "integrations",
    "mem0",
)
if _ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _ADAPTER_DIR)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def recall_url():
    """One pgvector container for this module; yields a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(f"testcontainers required. Import error: {_IMPORT_ERROR}")
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def recall_server(recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path, monkeypatch):
    """Boot a recall-enabled RosalindDB over a real socket; yield its base URL.

    Mirrors the in-process harness used by the recall integration tests
    (``test_crud_union.py``) but serves the combined app (source_registry +
    the in-process v1_query search router) under uvicorn on a free port so the
    real REST client can reach it.
    """
    import uvicorn

    # --- env BEFORE importing the app modules (they read config at import) ---
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("DATABASE_URL", "memory://e2e")
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    # OSS no-auth single `default` tenant — keeps the demo path token-free.
    monkeypatch.setenv("RB_REQUIRE_AUTH", "false")

    # Drain any queues left from earlier tests in this process.
    from adapters.queue.queue import consume as _consume

    for _topic in (
        "VALIDATE_DATASET", "DATASET_READY", "DELETE_VECTORS", "CONSOLIDATE",
        "RUN_EPHEMERAL_QUERY", "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    # Reload state + apply the recall migrations against the pgvector container.
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()
    state_mod.migrate_recall(force=True)

    # Reload the app graph so each module picks up the env above.
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

    # Mount the in-process search router (read-your-writes union runs here).
    main_mod.app.include_router(v1_query.router)

    port = _free_port()
    config = uvicorn.Config(main_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    # Wait for the server to accept connections.
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover
            pytest.fail("uvicorn did not start in time")
        time.sleep(0.05)

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_read_your_writes_through_mem0_adapter(recall_server):
    """insert -> immediate search/get/list returns it; delete -> gone."""
    from rosalinddb import RosalindDB

    dim = 4
    store = RosalindDB(
        collection_name="agentmem",
        embedding_model_dims=dim,
        base_url=recall_server,
        token=None,  # OSS no-auth default tenant
    )

    # Collection exists.
    assert "agentmem" in store.list_cols()

    # insert two synthetic memories through the adapter -> NDJSON recall upsert.
    store.insert(
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        payloads=[
            {"user_id": "u1", "data": "allergic to peanuts"},
            {"user_id": "u1", "data": "prefers tea"},
        ],
        ids=["fact-1", "fact-2"],
    )

    # READ-YOUR-WRITES: an immediate search returns the just-written vector.
    hits = store.search(
        query="avoid?", vectors=[1.0, 0.0, 0.0, 0.0], top_k=5, filters={"user_id": "u1"}
    )
    by_id = {h.id: h for h in hits}
    assert "fact-1" in by_id, "read-your-writes: insert must be immediately searchable"
    assert "fact-2" in by_id
    # Nearest match (the query equals fact-1's vector) has the top similarity.
    assert by_id["fact-1"].score >= by_id["fact-2"].score
    assert by_id["fact-1"].score == pytest.approx(1.0)  # exact match -> 1/(1+0)
    assert all(0.0 < h.score <= 1.0 for h in hits)
    assert by_id["fact-1"].payload["data"] == "allergic to peanuts"

    # get returns it.
    got = store.get("fact-1")
    assert got is not None and got.id == "fact-1"
    assert got.payload["data"] == "allergic to peanuts"

    # list returns both (double-wrapped, filter passthrough).
    rows = store.list(filters={"user_id": "u1"})[0]
    assert {r.id for r in rows} == {"fact-1", "fact-2"}

    # delete -> read-your-deletes: get is None and search no longer returns it.
    store.delete("fact-1")
    assert store.get("fact-1") is None
    hits_after = store.search(
        query="avoid?", vectors=[1.0, 0.0, 0.0, 0.0], top_k=5, filters={"user_id": "u1"}
    )
    ids_after = {h.id for h in hits_after}
    assert "fact-1" not in ids_after, "read-your-deletes: deleted id must vanish from search"
    assert "fact-2" in ids_after
    rows_after = store.list(filters={"user_id": "u1"})[0]
    assert "fact-1" not in {r.id for r in rows_after}

    # update = re-upsert (last-write-wins) is immediately visible.
    store.update("fact-2", vector=[0.0, 1.0, 0.0, 0.0], payload={"user_id": "u1", "data": "prefers green tea"})
    assert store.get("fact-2").payload["data"] == "prefers green tea"

    # cleanup.
    store.reset()
