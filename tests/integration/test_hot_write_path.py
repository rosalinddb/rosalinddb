"""Integration coverage for the synchronous hot-tier write path (PR3).

Runs `POST /v1/datasets/{name}/vectors` end-to-end through the real FastAPI app
against a REAL pgvector container (`RB_HOT_DSN`) for the hot tier and the
session MinIO for landing. Proves both flag modes:

  - FLAG ON (`RB_DELTA_TIER=true` + `RB_HOT_DSN`): 200, rows land in
    `hot_vectors` with strictly-monotonic `lsn` and `deleted=false`, NO
    `VALIDATE_DATASET` is published, NO landing object is written; a re-send of
    the same id is last-write-wins (a single, updated row).
  - FLAG OFF (default): 202, a landing object IS written, a `VALIDATE_DATASET`
    IS published, and NO hot row exists.

The control plane stays on the default `memory://` state adapter — the hot path
is gated on `RB_HOT_DSN`, not the control-plane DSN. Each test reloads the app
modules so the patched env (`RB_HOT_DSN`, `RB_DELTA_TIER`, `LANDING_PREFIX`) is
picked up.
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
def hot_url():
    """One pgvector container for this module; yield a psycopg2 DSN.

    Same `pgvector/pgvector:pg15` image the compose `pgvector` service uses.
    """
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the hot-write suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _truncate_hot(dsn: str) -> None:
    """Drop all hot rows + reset the LSN sequence (per-test isolation)."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.hot_vectors'), "
                "to_regclass('public.hot_lsn_seq')"
            )
            hv, seq = cur.fetchone()
            if hv is not None:
                cur.execute("TRUNCATE hot_vectors")
            if seq is not None:
                cur.execute("TRUNCATE hot_lsn_seq")
    finally:
        conn.close()


def _on_client(monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    """Build a TestClient with the delta tier ON, pointed at the hot container."""
    monkeypatch.setenv("RB_HOT_DSN", hot_url)
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    return _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path)


def _off_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    """Build a TestClient with the delta tier OFF (default)."""
    monkeypatch.delenv("RB_HOT_DSN", raising=False)
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    return _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path)


def _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._HOT_MIGRATED = False
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

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app), state_mod


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_of(signup: dict) -> str:
    """Resolve the signup's tenant_id (the value `current_tenant_id` derives).

    It is the JWT `sub`, mirrored in the `tenant.id` field of the response.
    """
    return signup["tenant"]["id"]


def _drain_validate():
    from adapters.queue.queue import consume

    msgs = []
    while True:
        m = consume("VALIDATE_DATASET", block=False)
        if not m:
            break
        msgs.append(m)
    return msgs


def _hot_rows(dsn, tenant, dataset):
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, lsn, deleted FROM hot_vectors "
                "WHERE tenant_id=%s AND dataset=%s ORDER BY lsn",
                (tenant, dataset),
            )
            return cur.fetchall()
    finally:
        conn.close()


# --- flag ON --------------------------------------------------------------


def test_flag_on_writes_hot_rows_returns_200(
    monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag on: 200, rows in hot_vectors with monotonic lsn + deleted=false,
    NO VALIDATE_DATASET, NO landing object."""
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    monkeypatch.setenv("RB_HOT_DSN", hot_url)
    state_mod._HOT_MIGRATED = False
    state_mod.migrate_hot(force=True)
    _truncate_hot(hot_url)
    _drain_validate()

    client, _ = _on_client(
        monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
    )
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4})

    import adapters.storage.storage as storage_mod
    body = "\n".join(
        json.dumps({"id": f"r{i}", "values": [0.1 * i, 0.2, 0.3, 0.4], "metadata": {"n": i}})
        for i in range(3)
    )
    r = client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["accepted"] == 3 and out["rejected"] == 0 and out["errors"] == []
    assert "job_id" not in out

    # Tenant id is the bootstrap principal when auth is on -> the JWT's sub.
    tenant = _tenant_of(s)
    rows = _hot_rows(hot_url, tenant, "v")
    assert len(rows) == 3, rows
    lsns = [row[1] for row in rows]
    assert lsns == sorted(lsns) and len(set(lsns)) == 3, f"non-monotonic lsn: {lsns}"
    assert all(row[2] is False for row in rows), "rows must not be tombstoned"

    # No VALIDATE_DATASET published (the flag-on path neither lands nor
    # publishes — the two go together in the handler, so no-publish witnesses
    # no-land; the unit suite asserts the no-landing object directly).
    assert _drain_validate() == []
    _ = storage_mod  # imported for symmetry with the off test


def test_flag_on_last_write_wins_on_reupsert(
    monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Re-sending an id overwrites the row (single row, new higher lsn)."""
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    monkeypatch.setenv("RB_HOT_DSN", hot_url)
    state_mod._HOT_MIGRATED = False
    state_mod.migrate_hot(force=True)
    _truncate_hot(hot_url)

    client, _ = _on_client(
        monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
    )
    s = _signup(client, email="bob@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "w", "dimension": 4})
    tenant = _tenant_of(s)

    def _send(values):
        return client.post(
            "/v1/datasets/w/vectors",
            headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
            data=json.dumps({"id": "same", "values": values}),
        )

    assert _send([1.0, 0, 0, 0]).status_code == 200
    assert _send([9.0, 0, 0, 0]).status_code == 200

    rows = _hot_rows(hot_url, tenant, "w")
    assert len(rows) == 1, f"re-upsert must collapse to one row: {rows}"
    # The surviving row carries the second (higher) LSN.
    assert rows[0][1] == 2, rows

    # The stored embedding is the second write's value.
    conn = psycopg2.connect(hot_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM hot_vectors "
                "WHERE tenant_id=%s AND dataset=%s AND id='same'",
                (tenant, "w"),
            )
            emb = cur.fetchone()[0]
    finally:
        conn.close()
    assert emb.startswith("[9"), f"last write did not win: {emb}"


# --- flag OFF -------------------------------------------------------------


def test_flag_off_returns_202_lands_no_hot_row(
    monkeypatch, hot_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag off (default): 202, landing object written, no hot row.

    The pgvector container is up (migrated by other tests) but the off path must
    not touch it — proven by asserting the hot table has no row for this dataset.
    """
    _truncate_hot(hot_url)
    _drain_validate()

    client, _ = _off_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path
    )
    s = _signup(client, email="carol@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "x", "dimension": 4})
    tenant = _tenant_of(s)

    import adapters.storage.storage as storage_mod
    body = json.dumps({"id": "off1", "values": [1.0, 2.0, 3.0, 4.0]})
    r = client.post(
        "/v1/datasets/x/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    assert r.json()["job_id"].startswith("job_")

    # Landing object present + VALIDATE_DATASET published.
    msgs = _drain_validate()
    assert len(msgs) == 1 and msgs[0]["dataset"] == "x"
    assert storage_mod.exists(msgs[0]["uri"]), "flag-off must write a landing object"

    # No hot row (the off path never touched the hot store).
    assert _hot_rows(hot_url, tenant, "x") == []
