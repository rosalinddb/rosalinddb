"""`POST /admin/prewarm` on the Query-DP app.

A manual / smoke-test entry point that takes `{shard_uri}` in the body,
calls `shard_tier.prewarm(shard_uri)`, and returns:

  - 200 on success (with the local path in the body)
  - 503 + `cache_capacity_exceeded` on `CacheCapacityExceeded`
  - 404 + `shard_not_found` on `FileNotFoundError`

The endpoint is gated on `RB_ADMIN_ENDPOINTS=true` so the surface is
opt-in. With the gate off the route returns 404 (or the FastAPI default
"Not Found") — the route does not register at all unless the gate is
explicitly enabled.

These tests build a minimal FastAPI app that mounts the same router /
endpoint so the test does not need the full dp_app startup machinery.
"""
from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Build a Query-DP TestClient with the admin gate ON.

    Reloads the dp_app module so the gate is re-evaluated on import. The
    shard_tier module is also reloaded so the test's tmpdir env vars take
    effect.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", str(1024 * 1024))
    monkeypatch.setenv("RB_ADMIN_ENDPOINTS", "true")

    from adapters.storage import shard_tier
    importlib.reload(shard_tier)

    import services.query_api.dp_app as dp_app_mod
    importlib.reload(dp_app_mod)

    from fastapi.testclient import TestClient

    client = TestClient(dp_app_mod.app)
    yield client, dp_app_mod, shard_tier


@pytest.fixture
def app_env_gate_off(monkeypatch, tmp_path):
    """Build a Query-DP TestClient with the admin gate OFF (default)."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", str(1024 * 1024))
    monkeypatch.delenv("RB_ADMIN_ENDPOINTS", raising=False)

    from adapters.storage import shard_tier
    importlib.reload(shard_tier)

    import services.query_api.dp_app as dp_app_mod
    importlib.reload(dp_app_mod)

    from fastapi.testclient import TestClient

    client = TestClient(dp_app_mod.app)
    yield client, dp_app_mod, shard_tier


# --- tests ----------------------------------------------------------------


def test_admin_prewarm_returns_200_on_success(app_env, monkeypatch):
    """`POST /admin/prewarm` returns 200 when prewarm succeeds.

    The handler must call `shard_tier.prewarm(shard_uri)` and surface a
    JSON body that lets an operator confirm the admission happened.
    """
    client, dp_app_mod, shard_tier = app_env

    captured: dict = {}

    def _fake_prewarm(uri):
        captured["uri"] = uri
        return "/tmp/fake/local/path.bin"

    monkeypatch.setattr(shard_tier, "prewarm", _fake_prewarm)

    r = client.post(
        "/admin/prewarm",
        json={"shard_uri": "memory://bucket/prewarm-me.bin"},
    )

    assert r.status_code == 200, r.text
    assert captured == {"uri": "memory://bucket/prewarm-me.bin"}


def test_admin_prewarm_maps_capacity_exceeded_to_503(app_env, monkeypatch):
    """`CacheCapacityExceeded` -> 503 + `cache_capacity_exceeded` code.

    The endpoint mirrors the wire contract that the hot path's classifier
    uses, so operators get the same 503 + code regardless of whether
    capacity pressure surfaces on a manual prewarm or on a queued one.
    """
    client, dp_app_mod, shard_tier = app_env

    def _fail_prewarm(uri):  # noqa: ARG001
        raise shard_tier.CacheCapacityExceeded("tier full of young entries")

    monkeypatch.setattr(shard_tier, "prewarm", _fail_prewarm)

    r = client.post(
        "/admin/prewarm",
        json={"shard_uri": "memory://bucket/full.bin"},
    )

    assert r.status_code == 503, r.text
    body = r.json()
    assert body["error"]["code"] == "cache_capacity_exceeded", body


def test_admin_prewarm_maps_filenotfound_to_404(app_env, monkeypatch):
    """`FileNotFoundError` -> 404 + `shard_not_found` code.

    A prewarm against a URI the object store cannot resolve is a real
    operator error (typo, deleted shard); 404 is the right shape.
    """
    client, dp_app_mod, shard_tier = app_env

    def _missing_prewarm(uri):  # noqa: ARG001
        raise FileNotFoundError("memory:// key not found")

    monkeypatch.setattr(shard_tier, "prewarm", _missing_prewarm)

    r = client.post(
        "/admin/prewarm",
        json={"shard_uri": "memory://bucket/missing.bin"},
    )

    assert r.status_code == 404, r.text
    body = r.json()
    assert body["error"]["code"] == "shard_not_found", body


def test_admin_prewarm_invalid_body_is_400(app_env):
    """A request with no `shard_uri` is 400 (invalid_request)."""
    client, _, _ = app_env

    r = client.post("/admin/prewarm", json={})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_request", body


def test_admin_prewarm_non_json_body_is_400(app_env):
    """A non-JSON body is rejected as invalid_request, not 500.

    The handler's three 400 branches (non-JSON / non-dict / non-string
    shard_uri) all funnel to the same `invalid_request` envelope — this
    test pins the first branch so a future refactor cannot regress an
    operator typo (curl without -H 'Content-Type: application/json') into
    a 500.
    """
    client, _, _ = app_env

    r = client.post(
        "/admin/prewarm",
        content=b"this is not json",
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_request", body


def test_admin_prewarm_non_dict_body_is_400(app_env):
    """A JSON body that decodes to a non-dict (e.g. a list) is 400."""
    client, _, _ = app_env

    r = client.post("/admin/prewarm", json=["memory://bucket/x.bin"])
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_request", body


def test_admin_prewarm_non_string_shard_uri_is_400(app_env):
    """`shard_uri` present but not a string -> 400.

    The handler must validate the type so a caller that sends
    `{"shard_uri": 42}` does not slip through to `shard_tier.prewarm`
    where an int would raise an obscure TypeError deep in the stack.
    """
    client, _, _ = app_env

    r = client.post("/admin/prewarm", json={"shard_uri": 42})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_request", body


def test_admin_prewarm_is_404_when_gate_off(app_env_gate_off):
    """With `RB_ADMIN_ENDPOINTS` unset, the endpoint is not registered.

    Default-off rollback contract: the surface is opt-in. A deployment
    that has not opted in must not even accept the route — a stray
    operator request returns the FastAPI 404 for an unknown route.
    """
    client, _, _ = app_env_gate_off

    r = client.post(
        "/admin/prewarm",
        json={"shard_uri": "memory://bucket/anything.bin"},
    )
    assert r.status_code == 404, r.text
