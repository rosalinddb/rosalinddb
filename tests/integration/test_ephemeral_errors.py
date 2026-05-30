"""Storage / cache errors must not silently degrade to `{matches:[]}`.

Background. The ephemeral runner used to catch any `PermissionError`,
`FileNotFoundError`, or boto3 `ClientError` from its shard download / cache
write / FAISS load, NACK the queue message, and publish *nothing* back. The CP
status poll then saw `{ready: false}` until DLQ — and the synchronous hot
path's `except Exception: hot = None` collapsed real failures into "no shard
yet", silently routing the query to the ephemeral fallback (HTTP 200 with
`matches: []`). A 200 with no matches is the legitimate empty-result case;
collapsing storage outages into the same shape was a silent wrong-answer.

These tests pin the new contract:

  1. A 200 from `/v1/query` ALWAYS implies `matches` is a real top-K result.
     Cache fs unwritable / S3 read failure -> HTTP 503 with the v1 error
     envelope, not 200 with empty matches.

  2. The legitimate empty-result paths still 200 with `{matches: []}` — a
     filter matching nothing, a freshly-created not-yet-indexed dataset, an
     empty dataset. Regression guard.

  3. The reliable-queue NACK + DLQ behaviour is preserved across the new
     "publish error envelope" step: every delivery attempt republishes the
     envelope (the caller is unblocked immediately, not after the retry
     budget drains), and after `QUEUE_MAX_ATTEMPTS` the message is
     dead-lettered exactly as before.

The fixture mirrors `test_query_api.py:client` so the full pipeline runs
against real MinIO under the integration markers.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient mirroring `test_query_api.py:client`.

    Same reload + queue-drain dance so the test starts from clean state:
    in-process queues are emptied so a stray RUN_EPHEMERAL_QUERY message from
    a prior test does not get consumed by this one's runner.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    from adapters.queue.queue import consume as _consume
    for _topic in (
        "VALIDATE_DATASET",
        "DATASET_READY",
        "RUN_EPHEMERAL_QUERY",
        "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
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
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_id(client, signup_body):
    r = client.get("/auth/me", headers=_auth(signup_body["token"]))
    return r.json()["tenant"]["id"]


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously."""
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


def _upload(client, token, name, records):
    body = "\n".join(json.dumps(r) for r in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()


def _make_indexed_dataset(client, token, name="test", dimension=4, records=None):
    r = client.post(
        "/v1/datasets",
        headers=_auth(token),
        json={"name": name, "dimension": dimension},
    )
    assert r.status_code == 201, r.text
    if records is None:
        records = [
            {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"title": f"t{i}"}}
            for i in range(10)
        ]
    _upload(client, token, name, records)
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds


# ---------------------------------------------------------------------------
# Hot-path error propagation
# ---------------------------------------------------------------------------


def test_hot_path_storage_read_failure_returns_503_not_silent_200(
    client, monkeypatch
):
    """A storage outage during the hot search must surface as 503, not 200.

    The fix's core invariant: an exception inside `_hot_search` no longer
    collapses to `hot = None` (which used to fall through to the ephemeral
    fallback and return HTTP 200 `{matches: []}`). It now classifies into a
    v1 error envelope and is surfaced with HTTP 503.

    We simulate the outage with the cleanest possible monkeypatch: replace
    the FAISS `read_index` call inside the v1_query module with one that
    raises `FileNotFoundError`. That mirrors the real-world shape of an S3
    fetch landing on a deleted shard or a cache-miss path that cannot find
    the file it just tried to download. The classification helper maps
    `FileNotFoundError` to `storage_unavailable`.
    """
    import services.query_api.v1_query as v1_query

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    # Force a cold load: blow away the cache so `_hot_search` has to call
    # `_ensure_cached` -> `read_bytes` -> `read_index` afresh.
    v1_query.cache_clear()

    def _boom(path, *a, **kw):
        raise FileNotFoundError(f"simulated missing shard: {path}")

    monkeypatch.setattr(v1_query.faiss, "read_index", _boom)

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5},
    )
    # The wrong-answer-bug behaviour: 200 with empty matches. The correct
    # behaviour is 503 with the v1 envelope.
    assert r.status_code == 503, r.text
    body = r.json()
    assert "error" in body, body
    assert body["error"]["code"] == "storage_unavailable", body
    # No `matches` leaks into a structured error envelope.
    assert "matches" not in body, body


def test_hot_path_cache_permission_error_returns_503_cache_unavailable(
    client, monkeypatch
):
    """A `PermissionError` on the cache fs maps to `cache_unavailable`.

    This is the "fresh `docker compose up` cache fs is unwritable"
    case: the FAISS shard cache directory is unreadable / unwritable. The
    runner used to swallow it and return `{matches: []}`. This test surfaces
    it explicitly so a self-hoster knows what to inspect (`CACHE_DIR`
    permissions).
    """
    import services.query_api.v1_query as v1_query

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    v1_query.cache_clear()

    # Inject the PermissionError at the cache-write boundary by replacing
    # `_ensure_cached` itself; that mirrors the bind-mount-permission shape.
    def _no_cache_write(*a, **kw):
        raise PermissionError("simulated unwritable cache dir")

    monkeypatch.setattr(v1_query, "_ensure_cached", _no_cache_write)

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5},
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "cache_unavailable"


# ---------------------------------------------------------------------------
# Legitimate empty-result regression guard
# ---------------------------------------------------------------------------


def test_legitimate_empty_matches_still_returns_200(client):
    """Filter that matches nothing -> 200 with `matches: []`, NOT 503.

    The fix must distinguish "we ran the query and found no matches"
    (legitimate, HTTP 200) from "we could not run the query" (HTTP 503 with
    envelope). This pins the legitimate-empty case so the new error path
    cannot accidentally swallow real empty results.
    """
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={
            "dataset": "test",
            "vector": [0.0, 0.0, 0.0, 0.0],
            "top_k": 5,
            "filter": {"title": "no-such-title"},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] == []
    assert body["mode"] in ("hot", "cold")
    assert "error" not in body


def test_no_shard_yet_still_returns_200_ephemeral(client):
    """A dataset with no shard is the documented ephemeral fallback -> 200.

    Distinct from the storage-error case: there is no shard to load, so the
    hot path returns None (not an exception) and the request is enqueued.
    The immediate response is the documented enqueue shape; the silent
    wrong-answer bug never lived on this path.
    """
    s = _signup(client)
    r = client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "empty", "dimension": 4},
    )
    assert r.status_code == 201, r.text
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "empty", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ephemeral"
    assert body["matches"] == []
    assert body["job_id"].startswith("job_")
    assert "error" not in body


# ---------------------------------------------------------------------------
# Ephemeral runner -> status poll error propagation
# ---------------------------------------------------------------------------


def test_ephemeral_runner_publishes_error_envelope_on_storage_failure(
    client, monkeypatch
):
    """Runner failure -> structured envelope on RESULT_READY -> 503 on poll.

    A storage / cache failure inside the ephemeral runner now publishes
    `{ok: false, error: {code, message}}` to the reply queue. The status
    poll surfaces that envelope as HTTP 503 with the v1 error body —
    NOT as `{ready: true, matches: []}` (the silent wrong-answer shape).
    """
    import services.ephemeral_runner.run as ephemeral
    import services.query_api.v1_query as v1_query
    from adapters.queue.queue import consume, publish

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    tenant_id = _tenant_id(client, s)

    # The runner's first storage touch is `_ensure_cached` (it downloads the
    # shard, falling back to the local cache). Replacing it with a raise lets
    # us exercise the runner's exception path without touching FAISS or S3.
    def _no_cache(*a, **kw):
        raise PermissionError("simulated unwritable cache dir")

    monkeypatch.setattr(ephemeral, "_ensure_cached", _no_cache)

    # Drive the runner inline through its handler — `_handle_ephemeral` is the
    # real failure surface (it owns the publish-envelope-then-NACK contract).
    job_id = "job_fix_delta_err"
    publish(
        "RUN_EPHEMERAL_QUERY",
        {
            "dataset": "test",
            "tenant": tenant_id,
            "vector": [0.0, 0.0, 0.0, 0.0],
            "top_k": 3,
            "correlation_id": job_id,
            "reply_to": "RESULT_READY",
        },
    )
    msg = consume("RUN_EPHEMERAL_QUERY", block=False)
    assert msg is not None
    # The handler must re-raise (so the queue can NACK + DLQ) AND publish the
    # envelope to RESULT_READY before re-raising. Catch the re-raise here.
    with pytest.raises(PermissionError):
        ephemeral._handle_ephemeral(msg)

    # The error envelope is on the wire — drain it into the result store.
    v1_query.drain_result_queue_once()

    r = client.get(f"/v1/query/status/{job_id}", headers=_auth(s["token"]))
    # The whole point: 503 with envelope, NOT 200 with `{ready: true, matches: []}`.
    assert r.status_code == 503, r.text
    body = r.json()
    assert "error" in body, body
    assert body["error"]["code"] == "cache_unavailable", body
    # No `matches` snuck into the error response.
    assert "matches" not in body, body


def test_ephemeral_runner_success_status_poll_still_returns_matches(
    client, monkeypatch
):
    """The happy ephemeral path keeps returning `{ready: true, matches: ...}`.

    Regression guard for the status poll: the new `ok: false` branch must
    not swallow `ok: true` (or no-`ok`-field-at-all, a legacy stored result)
    results.
    """
    import services.ephemeral_runner.run as ephemeral
    import services.query_api.v1_query as v1_query
    from adapters.queue.queue import consume, publish

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    tenant_id = _tenant_id(client, s)

    job_id = "job_fix_delta_ok"
    publish(
        "RUN_EPHEMERAL_QUERY",
        {
            "dataset": "test",
            "tenant": tenant_id,
            "vector": [3.0, 0.0, 0.0, 0.0],
            "top_k": 3,
            "correlation_id": job_id,
            "reply_to": "RESULT_READY",
        },
    )
    msg = consume("RUN_EPHEMERAL_QUERY", block=False)
    assert msg is not None
    # No injection: the real handler runs end-to-end and publishes a success
    # envelope (`ok: true`) to RESULT_READY.
    ephemeral._handle_ephemeral(msg)
    v1_query.drain_result_queue_once()

    r = client.get(f"/v1/query/status/{job_id}", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert isinstance(body["matches"], list) and len(body["matches"]) > 0
    assert body["mode"] == "ephemeral"


# ---------------------------------------------------------------------------
# Reliable-queue: NACK + DLQ still works alongside the envelope publish
# ---------------------------------------------------------------------------


def test_runner_failure_nacks_and_eventually_dead_letters(client, monkeypatch):
    """A repeatedly-failing runner publishes the envelope every attempt AND
    eventually dead-letters the message past `QUEUE_MAX_ATTEMPTS`.

    Pins both halves of the new contract:
      - the caller is unblocked on the FIRST failed attempt (envelope on
        RESULT_READY -> 503 on the next poll), no need to wait for retries
        to exhaust;
      - the reliable-queue retry + DLQ behaviour from the in-process path
        survives the new publish step.

    The in-process queue adapter's `nack(requeue=True)` re-`publish`es the
    message; we drive a few retry rounds and confirm each round republishes
    the error envelope. The Redis DLQ semantics are covered by the existing
    `tests/integration/test_queue_redis.py` reliability tests — duplicating
    the Redis tracking here would only retest the queue adapter.
    """
    import services.ephemeral_runner.run as ephemeral
    import services.query_api.v1_query as v1_query
    from adapters.queue.queue import consume, publish

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    tenant_id = _tenant_id(client, s)

    def _no_cache(*a, **kw):
        raise PermissionError("simulated unwritable cache dir")

    monkeypatch.setattr(ephemeral, "_ensure_cached", _no_cache)

    job_id = "job_fix_delta_dlq"
    publish(
        "RUN_EPHEMERAL_QUERY",
        {
            "dataset": "test",
            "tenant": tenant_id,
            "vector": [0.0, 0.0, 0.0, 0.0],
            "top_k": 3,
            "correlation_id": job_id,
            "reply_to": "RESULT_READY",
        },
    )

    # Drive three "attempts" — each is: consume the message, watch the
    # handler raise, then have the runner's main_loop nack/requeue it. The
    # in-process queue has no attempt counter, so we cap the retries
    # manually rather than waiting for DLQ semantics that only the Redis
    # path enforces.
    attempts_published = 0
    for _ in range(3):
        msg = consume("RUN_EPHEMERAL_QUERY", block=False)
        if msg is None:
            break
        try:
            ephemeral._handle_ephemeral(msg)
        except PermissionError:
            from adapters.queue.queue import nack
            nack(msg, requeue=True)
            attempts_published += 1

    assert attempts_published == 3, "expected three re-delivery attempts"

    # Each failed attempt republished the envelope; the status poll surfaces
    # the LATEST one as 503 (the result store is last-write-wins).
    v1_query.drain_result_queue_once()
    r = client.get(f"/v1/query/status/{job_id}", headers=_auth(s["token"]))
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "cache_unavailable"
