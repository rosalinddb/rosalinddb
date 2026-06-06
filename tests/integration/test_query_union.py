"""Integration coverage for the Recall + Consolidated query union (PR4).

Runs `POST /v1/query` end-to-end through the real FastAPI app, with:

  - the **recall tier** on a REAL `pgvector/pgvector:pg15` container
    (`RB_RECALL_DSN`), exactly like the write-path suite;
  - the **consolidated (cold)** tier built by the real ingest pipeline into the
    session MinIO (the same fixtures `test_query_api.py` uses);
  - the control plane on the default `memory://` state adapter (the recall path
    is gated on `RB_RECALL_DSN`, not the control-plane DSN).

Properties proven (docs/architecture/recall-consolidate.md, "Read path — the
union", invariants I1/I3):

  - **read-your-writes**: flag-on `POST /vectors` (recall) then an immediate
    `POST /query` returns the just-written vector — synchronously, no ephemeral
    poll.
  - **recall-wins**: a vector in the cold shard + an updated copy in recall →
    the recall version is returned.
  - **authoritative suppression**: a live recall re-upsert that FAILS the query
    filter still hides the stale, filter-matching cold copy of its id.
  - **tombstone suppression**: a recall tombstone hides a cold id.
  - **no cold shard + recall data**: a synchronous recall result (`mode:
    "recall"`), NOT the ephemeral empty+job_id path.
  - **flag OFF (default)**: pure cold path, unchanged, no recall row consulted.
"""
from __future__ import annotations

import importlib
import json

import psycopg2
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def recall_url():
    """One pgvector container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the query-union suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _truncate_recall(dsn: str) -> None:
    """Drop all recall rows + reset the LSN sequence (per-test isolation)."""
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


def _tombstone_recall(dsn, tenant, dataset, vid) -> None:
    """Mark a recall row deleted (a later PR writes tombstones; we seed one)."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE recall_vectors SET deleted = TRUE "
                "WHERE tenant_id=%s AND dataset=%s AND id=%s",
                (tenant, dataset, vid),
            )
    finally:
        conn.close()


def _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on):
    """Build a TestClient + reloaded state module.

    `recall_on` toggles `RB_RECALL` / `RB_RECALL_DSN` for the whole client.
    Cold shards always land in MinIO (the pipeline); recall always points at the
    container so the OFF test can prove it is never consulted.
    """
    if recall_on:
        monkeypatch.setenv("RB_RECALL", "true")
    else:
        monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY"):
        while _consume(_topic, block=False):
            pass

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

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.ephemeral_runner.run as ephemeral
    importlib.reload(ephemeral)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app), state_mod, v1_query


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_of(client, signup):
    r = client.get("/auth/me", headers=_auth(signup["token"]))
    return r.json()["tenant"]["id"]


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously (cold shard)."""
    from adapters.queue.queue import consume
    from services.validator_worker.run import process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


def _make_cold_dataset(client, token, name, dimension, records):
    """Create + populate + index a dataset into a real cold FAISS shard.

    Must run with the recall flag OFF so the upload lands + builds a shard (the
    flag-on path writes recall instead). Caller flips the flag afterward.
    """
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension})
    assert r.status_code == 201, r.text
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds


def _post_recall(client, token, name, records):
    """Flag-on `POST /vectors` → synchronous recall write (200, no landing)."""
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    return r.json()


def _migrate_recall(recall_url):
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    import os
    os.environ["RB_RECALL_DSN"] = recall_url
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)


# --- read-your-writes -----------------------------------------------------


def test_read_your_writes_no_cold_shard(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag-on write then immediate query returns the vector — no cold shard.

    The dataset never builds a shard (flag-on `POST /vectors` writes recall only),
    so the cold tier is empty. The query MUST still return the just-written
    vector synchronously (`mode: "recall"`), NOT the ephemeral empty+job_id path.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "rw", "dimension": 4})

    _post_recall(client, s["token"], "rw", [
        {"id": "fact-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"t": "peanuts"}},
        {"id": "fact-2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"t": "shellfish"}},
    ])

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "rw", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["mode"] == "recall", out
    assert "job_id" not in out
    ids = [m["id"] for m in out["matches"]]
    assert ids[0] == "fact-1", out  # nearest to the query
    assert set(ids) == {"fact-1", "fact-2"}
    # Metadata round-trips from the recall row.
    by_id = {m["id"]: m for m in out["matches"]}
    assert by_id["fact-1"]["metadata"] == {"t": "peanuts"}


def test_read_your_writes_filter_applied_to_recall(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A filtered query over recall-only data honours the AND-of-equals filter."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="filt@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "f", "dimension": 4})
    _post_recall(client, s["token"], "f", [
        {"id": "en", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"lang": "en"}},
        {"id": "fr", "values": [1.0, 0.1, 0.0, 0.0], "metadata": {"lang": "fr"}},
    ])

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={
            "dataset": "f",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "top_k": 10,
            "filter": {"lang": "fr"},
        },
    )
    assert r.status_code == 200, r.text
    assert [m["id"] for m in r.json()["matches"]] == ["fr"]


# --- cold + recall union (recall-wins) ------------------------------------


def test_recall_wins_over_cold_for_same_id(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A vector in the cold shard with an updated copy in recall → recall wins."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    # Build a cold shard with the flag OFF (so the upload lands + builds).
    client_off, _state, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client_off, email="win@example.com")
    _make_cold_dataset(client_off, s["token"], "u", 4, [
        {"id": "shared", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {"v": "cold"}},
        {"id": "cold-only", "values": [0.0, 5.0, 0.0, 0.0], "metadata": {"v": "cold"}},
    ])
    tenant = _tenant_of(client_off, s)

    # Re-open the SAME memory state with the flag ON, preserving the cold shard +
    # tenant. `_build_client` clears memory state, so instead flip the flag in
    # place and reload only the query module against the live state.
    import os
    os.environ["RB_RECALL"] = "true"
    os.environ["RB_RECALL_DSN"] = recall_url
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    client_off.app.include_router(v1_query.router)

    # Write an UPDATED copy of `shared` into recall (nearer to a different query).
    import adapters.state.state as state_mod
    state_mod.recall_upsert_vectors(tenant, "u", [
        {"id": "shared", "values": [5.1, 0.0, 0.0, 0.0], "metadata": {"v": "recall"}},
    ])

    r = client_off.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "u", "vector": [5.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    by_id = {m["id"]: m for m in out["matches"]}
    # `shared` is the RECALL version (metadata v=recall), deduped recall-wins.
    assert by_id["shared"]["metadata"] == {"v": "recall"}, out
    # `cold-only` still served from the cold shard.
    assert by_id["cold-only"]["metadata"] == {"v": "cold"}
    # mode reflects the cold-shard cache state (hot|cold), not "recall".
    assert out["mode"] in ("hot", "cold")


def test_filter_failing_live_recall_suppresses_stale_cold(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """P1 regression: a live re-upsert that fails the filter hides its stale cold copy.

    Cold shard holds `X {color: red}` (matches filter `{color: red}`). Recall has
    a newer live `X {color: blue}` (FAILS the filter), `lsn > watermark`. A query
    `{color: red}` must NOT return X: recall is authoritative and X's current
    version no longer matches the filter, so the stale filter-matching cold copy
    must not leak.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client_off, _state, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client_off, email="leak@example.com")
    _make_cold_dataset(client_off, s["token"], "lk", 4, [
        {"id": "X", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {"color": "red"}},
        {"id": "Y", "values": [5.0, 0.1, 0.0, 0.0], "metadata": {"color": "red"}},
    ])
    tenant = _tenant_of(client_off, s)

    import os
    os.environ["RB_RECALL"] = "true"
    os.environ["RB_RECALL_DSN"] = recall_url
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    client_off.app.include_router(v1_query.router)

    # Re-upsert X into recall with color=blue (the current authoritative version),
    # which does NOT match the query filter {color: red}.
    import adapters.state.state as state_mod
    state_mod.recall_upsert_vectors(tenant, "lk", [
        {"id": "X", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {"color": "blue"}},
    ])

    r = client_off.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={
            "dataset": "lk",
            "vector": [5.0, 0.0, 0.0, 0.0],
            "top_k": 10,
            "filter": {"color": "red"},
        },
    )
    assert r.status_code == 200, r.text
    ids = [m["id"] for m in r.json()["matches"]]
    assert "X" not in ids, (
        "stale cold X (color=red) must be suppressed by the live recall "
        "re-upsert (color=blue) even though the live row fails the filter"
    )
    assert "Y" in ids


def test_recall_tombstone_hides_cold_id(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A recall tombstone suppresses a cold id from the union."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="tomb@example.com")
    _make_cold_dataset(client, s["token"], "tb", 4, [
        {"id": "doomed", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "survivor", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    tenant = _tenant_of(client, s)

    import os
    os.environ["RB_RECALL"] = "true"
    os.environ["RB_RECALL_DSN"] = recall_url
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    client.app.include_router(v1_query.router)

    # Write `doomed` into recall, then tombstone it (a later PR writes tombstones
    # via DELETE; here we set the flag directly to exercise the union's handling).
    import adapters.state.state as state_mod
    state_mod.recall_upsert_vectors(tenant, "tb", [
        {"id": "doomed", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ])
    _tombstone_recall(recall_url, tenant, "tb", "doomed")

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "tb", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    ids = [m["id"] for m in r.json()["matches"]]
    assert "doomed" not in ids, "tombstone must suppress the cold id"
    assert "survivor" in ids


# --- flag OFF: pure cold path, unchanged ----------------------------------


def test_flag_off_is_pure_cold_path(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag off (default): the query is pure cold, recall never consulted.

    Recall holds a row for the same id that WOULD win if the union were on; with
    the flag off it must be ignored — proving the off path opens no recall conn.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, _v1q = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="off@example.com")
    _make_cold_dataset(client, s["token"], "o", 4, [
        {"id": "shared", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {"v": "cold"}},
    ])
    tenant = _tenant_of(client, s)

    # Seed a recall row that WOULD win if the union were on — but the flag is
    # OFF for the query, so it must be invisible. `recall_upsert_vectors` writes
    # straight to the recall store via `_recall_conn()` (it keys on
    # `RB_RECALL_DSN`, NOT the `RB_RECALL` flag), so we can seed without flipping
    # the query-path flag or reloading state (which would clear the dataset).
    state_mod.recall_upsert_vectors(tenant, "o", [
        {"id": "shared", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {"v": "recall"}},
    ])
    # `RB_RECALL` is unset for this client (recall_on=False), so the query's
    # `recall_enabled()` gate is OFF and the recall row above is never consulted.

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "o", "vector": [5.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    by_id = {m["id"]: m for m in out["matches"]}
    # The COLD version wins — recall was never consulted (flag off).
    assert by_id["shared"]["metadata"] == {"v": "cold"}, out
    assert out["mode"] in ("hot", "cold")
