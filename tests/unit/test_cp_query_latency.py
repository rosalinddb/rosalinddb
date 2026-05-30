"""Unit tests for CP query-latency optimisations.

TDD suite — write first, implement second.

Covers (in order of the fix scope):
  1. API-key auth resolution cache (item 1):
       - A second auth within the TTL does NOT call get_api_key_by_hash again.
       - A revoked key whose cache entry is busted stops authenticating
         immediately (does not wait for TTL expiry).
       - Negative / failed lookups are NEVER stored in the cache.
       - Cache is keyed on sha256(raw_key), not the raw key text.
  2. DP-pool cache (item 2):
       - A second get_tenant_dp_pool within the TTL does NOT call the DB
         again.
       - The cached value expires after the TTL.
  3. touch_api_key_last_used is non-blocking (item 3):
       - The resolved tenant_id is returned BEFORE the touch UPDATE has
         finished; i.e. the touch runs outside the hot path's critical
         section.
  4. Sidecar span (item 6):
       - read_shard_sidecar emits a 'shard.load_sidecar' OTel span.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import threading
import time

import pytest


# Use in-memory state for all tests in this module.
os.environ.setdefault("DATABASE_URL", "memory://test")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_jwt_utils():
    """Reload jwt_utils AND clear its auth cache so each test starts clean."""
    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    # Clear the cache if it already exists on the module.
    if hasattr(jwt_utils, "_clear_auth_cache"):
        jwt_utils._clear_auth_cache()
    return jwt_utils


def _reload_state():
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_API_KEYS_BY_HASH"):
        obj = getattr(state_mod, attr, None)
        if obj is not None:
            if isinstance(obj, dict):
                obj.clear()
            elif isinstance(obj, list):
                obj.clear()
    return state_mod


def _make_key_and_tenant(state_mod, jwt_utils):
    """Create a tenant + API key in the memory store; return (raw_key, tenant_id)."""
    tenant_id = "ten_cachetest"
    key_hash = hashlib.sha256(b"rb_live_testkey1234567890123456").hexdigest()
    state_mod.create_tenant(tenant_id, "cache@example.com", "pw_hash_ignored")
    state_mod.create_api_key("key_test1", tenant_id, key_hash, "Test")
    raw_key = "rb_live_testkey1234567890123456"
    return raw_key, tenant_id


# ---------------------------------------------------------------------------
# 1. API-key auth resolution cache
# ---------------------------------------------------------------------------


class TestApiKeyCacheHit:
    """A second call within TTL must not hit the DB again."""

    def test_second_resolve_skips_db_lookup(self, monkeypatch):
        """On a cache hit, get_api_key_by_hash is NOT called a second time."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        calls = {"n": 0}
        real_lookup = state_mod.get_api_key_by_hash

        def _spy(key_hash):
            calls["n"] += 1
            return real_lookup(key_hash)

        monkeypatch.setattr(state_mod, "get_api_key_by_hash", _spy)
        # Also patch the reference imported into jwt_utils.
        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        # First call: cold cache → DB hit.
        resolved1 = jwt_mod._resolve_api_key(raw_key)
        assert resolved1 == tenant_id, f"first resolve failed: {resolved1!r}"
        assert calls["n"] == 1, "first call should hit the DB"

        # Second call within TTL: warm cache → NO additional DB hit.
        resolved2 = jwt_mod._resolve_api_key(raw_key)
        assert resolved2 == tenant_id
        assert calls["n"] == 1, (
            f"second resolve within TTL should NOT hit the DB; "
            f"got {calls['n']} DB calls"
        )

    def test_cache_key_is_hash_not_raw_secret(self, monkeypatch):
        """The in-process cache dict must be keyed by sha256(key), NOT the raw key."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        jwt_mod._resolve_api_key(raw_key)

        # The raw key must NOT appear as a cache entry key.
        cache = jwt_mod._AUTH_CACHE  # type: ignore[attr-defined]
        assert raw_key not in cache, (
            "Cache must NOT be keyed on the raw API key secret"
        )
        # The SHA-256 of the raw key MUST be the cache key.
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert expected_hash in cache, (
            "Cache must be keyed on sha256(raw_key)"
        )

    def test_failed_lookup_not_cached(self, monkeypatch):
        """A failed (unknown/revoked) key resolution is NEVER stored in the cache."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        bad_key = "rb_live_" + "x" * 32
        result = jwt_mod._resolve_api_key(bad_key)
        assert result is None

        cache = jwt_mod._AUTH_CACHE  # type: ignore[attr-defined]
        bad_hash = hashlib.sha256(bad_key.encode()).hexdigest()
        assert bad_hash not in cache, (
            "Failed/unknown keys must NOT be stored in the auth cache"
        )

    def test_cache_expires_after_ttl(self, monkeypatch):
        """After the TTL, the cache entry is gone and the DB is hit again."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        calls = {"n": 0}
        real_lookup = state_mod.get_api_key_by_hash

        def _spy(key_hash):
            calls["n"] += 1
            return real_lookup(key_hash)

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)
        monkeypatch.setattr(state_mod, "get_api_key_by_hash", _spy)
        # A fresh reload of jwt_mod binds state_mod freshly; patch its own reference.
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        # Shorten the TTL to 0 seconds for this test.
        monkeypatch.setattr(jwt_mod, "_AUTH_CACHE_TTL_S", 0)

        jwt_mod._resolve_api_key(raw_key)
        assert calls["n"] == 1

        # Even with TTL=0, re-resolve should find the entry expired.
        # A tiny sleep ensures the monotonic clock advances past 0s TTL.
        time.sleep(0.01)
        jwt_mod._resolve_api_key(raw_key)
        assert calls["n"] == 2, (
            "After TTL expiry the DB should be hit again; "
            f"got {calls['n']} total DB calls"
        )


class TestRevocationCacheBust:
    """Revoking a key must immediately bust its cache entry (in this process)."""

    def test_revoked_key_rejected_immediately_after_bust(self, monkeypatch):
        """After revoke_api_key + bust, _resolve_api_key returns None immediately."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        import services.auth.jwt_utils as jwt_mod
        import services.auth.auth as auth_mod
        importlib.reload(auth_mod)
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        # First: resolve successfully (populates cache).
        resolved = jwt_mod._resolve_api_key(raw_key)
        assert resolved == tenant_id

        # Revoke via state + bust the cache explicitly (as the DELETE handler must).
        state_mod.revoke_api_key("key_test1", tenant_id)
        jwt_mod.bust_api_key_cache(raw_key)  # type: ignore[attr-defined]

        # Now the key must be rejected without waiting for TTL.
        result = jwt_mod._resolve_api_key(raw_key)
        assert result is None, (
            "A revoked key whose cache entry was busted must resolve to None"
        )

    def test_revoke_handler_busts_cache(self, monkeypatch):
        """DELETE /auth/keys/{id} immediately evicts the key from the auth cache.

        This is an effect test: after a DELETE the key must be rejected on the
        same worker without waiting for the TTL, even if the cache was warm.
        We also verify that bust_api_key_cache_by_hash is called on DELETE.
        """
        import adapters.state.state as state_mod_real
        importlib.reload(state_mod_real)
        for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_API_KEYS_BY_HASH"):
            obj = getattr(state_mod_real, attr, None)
            if obj is not None:
                if isinstance(obj, dict):
                    obj.clear()
                elif isinstance(obj, list):
                    obj.clear()
        import services.auth.jwt_utils as jwt_mod
        importlib.reload(jwt_mod)
        import services.auth.auth as auth_mod
        importlib.reload(auth_mod)
        import services.source_registry.main as main_mod
        importlib.reload(main_mod)
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)

        # Signup → get JWT + raw API key.
        r = client.post("/auth/signup", json={"email": "revoke@example.com", "password": "password123"})
        assert r.status_code == 201, r.text
        jwt_token = r.json()["token"]
        key_id = r.json()["first_api_key"]["id"]
        raw_key = r.json()["first_api_key"]["key"]

        # Prime the auth cache with a successful resolve.
        r2 = client.get("/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
        assert r2.status_code == 200, r2.text

        # Spy on bust_api_key_cache_by_hash.
        bust_calls = []
        real_bust = jwt_mod.bust_api_key_cache_by_hash  # type: ignore[attr-defined]

        def _spy_bust(key_hash):
            bust_calls.append(key_hash)
            return real_bust(key_hash)

        monkeypatch.setattr(jwt_mod, "bust_api_key_cache_by_hash", _spy_bust)
        monkeypatch.setattr(auth_mod, "bust_api_key_cache_by_hash", _spy_bust)  # type: ignore[attr-defined]

        # Delete the key.
        r3 = client.delete(f"/auth/keys/{key_id}", headers={"Authorization": f"Bearer {jwt_token}"})
        assert r3.status_code == 204, r3.text

        # The cache-bust function must have been called once.
        assert len(bust_calls) == 1, (
            f"bust_api_key_cache_by_hash should be called once on DELETE; got {bust_calls}"
        )

        # Effect: the revoked key must NOW be rejected (cache was busted).
        r4 = client.get("/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
        assert r4.status_code == 401, (
            "After DELETE + cache bust, the revoked key must be rejected immediately"
        )


# ---------------------------------------------------------------------------
# 2. DP-pool cache
# ---------------------------------------------------------------------------


class TestDpPoolCache:
    """get_tenant_dp_pool result is cached per tenant_id with a TTL."""

    def test_second_call_within_ttl_skips_db(self, monkeypatch):
        """A second get_tenant_dp_pool call within the TTL does not hit the DB."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        state_mod.create_tenant("ten_pool", "pool@example.com", "pw")

        calls = {"n": 0}
        real_fn = state_mod.get_tenant_dp_pool

        # We need to test the query_proxy module's caching wrapper, not the raw
        # state function. Import it fresh.
        import services.query_api.query_proxy as qp
        importlib.reload(qp)

        # Patch the underlying state function that qp delegates to.
        def _spy_pool(tenant_id):
            calls["n"] += 1
            return real_fn(tenant_id)

        monkeypatch.setattr(qp, "get_tenant_dp_pool", _spy_pool)
        # The wrapper inside qp calls qp.get_tenant_dp_pool; patch the cached wrapper too.
        # We test get_tenant_dp_pool_cached, the caching version.

        qp.get_tenant_dp_pool_cached("ten_pool")
        assert calls["n"] == 1

        qp.get_tenant_dp_pool_cached("ten_pool")
        assert calls["n"] == 1, (
            "Second call within TTL must not call the underlying DB function"
        )

    def test_dp_pool_cache_expires(self, monkeypatch):
        """After TTL expiry the DB is called again."""
        state_mod = _reload_state()

        state_mod.create_tenant("ten_exp", "exp@example.com", "pw")

        import services.query_api.query_proxy as qp
        importlib.reload(qp)

        calls = {"n": 0}

        def _spy_pool(tenant_id):
            calls["n"] += 1
            return "shared"

        monkeypatch.setattr(qp, "get_tenant_dp_pool", _spy_pool)
        monkeypatch.setattr(qp, "_DP_POOL_CACHE_TTL_S", 0)

        qp.get_tenant_dp_pool_cached("ten_exp")
        assert calls["n"] == 1

        time.sleep(0.01)
        qp.get_tenant_dp_pool_cached("ten_exp")
        assert calls["n"] == 2, "After TTL expiry the DB should be hit again"


# ---------------------------------------------------------------------------
# 3. touch_api_key_last_used is non-blocking on the hot query path
# ---------------------------------------------------------------------------


class TestTouchNonBlocking:
    """_resolve_api_key must return the tenant_id BEFORE the touch completes."""

    def test_resolve_returns_before_touch_completes(self, monkeypatch):
        """The tenant_id is returned without waiting for the touch UPDATE.

        We simulate a slow touch (100ms sleep) and assert the total time for
        _resolve_api_key is much less than 100ms — confirming the touch is
        fire-and-forget, not inline.
        """
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        touch_started = threading.Event()
        touch_complete = threading.Event()

        real_touch = state_mod.touch_api_key_last_used

        def _slow_touch(key_id):
            touch_started.set()
            time.sleep(0.1)  # 100ms artificial delay
            real_touch(key_id)
            touch_complete.set()

        monkeypatch.setattr(state_mod, "touch_api_key_last_used", _slow_touch)

        start = time.monotonic()
        resolved = jwt_mod._resolve_api_key(raw_key)
        elapsed = time.monotonic() - start

        assert resolved == tenant_id
        # Must return in << 100ms (the slow touch delay). Allow generous 50ms
        # budget for the rest of the path (dict lookup, hash, etc.).
        assert elapsed < 0.05, (
            f"_resolve_api_key blocked for {elapsed*1000:.1f}ms — "
            "touch_api_key_last_used must be fire-and-forget"
        )

        # The touch MUST eventually be called (just not blocking).
        assert touch_started.wait(timeout=1.0), "touch_api_key_last_used was never called"

    def test_touch_still_called_on_cache_miss(self, monkeypatch):
        """touch_api_key_last_used is still called on a cold (cache-miss) auth."""
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        touched = threading.Event()

        real_touch = state_mod.touch_api_key_last_used

        def _spy_touch(key_id):
            real_touch(key_id)
            touched.set()

        monkeypatch.setattr(state_mod, "touch_api_key_last_used", _spy_touch)

        result = jwt_mod._resolve_api_key(raw_key)
        assert result == tenant_id

        # touch must be called (fire-and-forget, but it must happen).
        assert touched.wait(timeout=1.0), "touch_api_key_last_used was not called on cache miss"

    def test_touch_not_called_on_cache_hit(self, monkeypatch):
        """On a cache HIT, touch_api_key_last_used is NOT called at all.

        The TTL bounds how often last_used_at can drift — at most once per TTL
        per key — which is the intended throttling behaviour.
        """
        state_mod = _reload_state()
        jwt_utils = _reload_jwt_utils()

        raw_key, tenant_id = _make_key_and_tenant(state_mod, jwt_utils)

        import services.auth.jwt_utils as jwt_mod
        monkeypatch.setattr(jwt_mod, "state_mod", state_mod)

        touch_count = {"n": 0}

        def _count_touch(key_id):
            touch_count["n"] += 1

        monkeypatch.setattr(state_mod, "touch_api_key_last_used", _count_touch)

        # First call (cold): touch must be called once.
        jwt_mod._resolve_api_key(raw_key)
        # Allow async touch to fire.
        time.sleep(0.05)
        assert touch_count["n"] == 1, "First (cold) resolve must call touch once"

        # Reset count. Second call (warm cache): touch must NOT be called.
        touch_count["n"] = 0
        jwt_mod._resolve_api_key(raw_key)
        time.sleep(0.05)
        assert touch_count["n"] == 0, (
            "Cache-hit resolve must NOT call touch_api_key_last_used"
        )


# ---------------------------------------------------------------------------
# 4. Sidecar span
# ---------------------------------------------------------------------------


class TestSidecarSpan:
    """read_shard_sidecar emits a 'shard.load_sidecar' OTel span."""

    @pytest.fixture
    def captured_spans(self, monkeypatch):
        """Install an isolated in-memory TracerProvider; yield the exporter."""
        from opentelemetry import trace as _trace_api
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(_trace_api, "_TRACER_PROVIDER", provider, raising=False)
        yield exporter
        exporter.clear()

    def test_read_shard_sidecar_emits_span(self, captured_spans, monkeypatch):
        """read_shard_sidecar emits a 'shard.load_sidecar' span."""
        import json
        from adapters.storage import storage as storage_mod
        from adapters.landing import parquet_reader

        importlib.reload(parquet_reader)

        storage_mod.memory_reset()
        shard_uri = "memory://shards/test_tenant/ds/shard.bin"
        sidecar_uri = f"{shard_uri}.meta.json"
        sidecar_data = {"1": {"id": "r1", "metadata": {}}}
        storage_mod.write_bytes(sidecar_uri, json.dumps(sidecar_data).encode())

        parquet_reader.read_shard_sidecar(shard_uri)

        names = [s.name for s in captured_spans.get_finished_spans()]
        assert "shard.load_sidecar" in names, (
            f"read_shard_sidecar must emit a 'shard.load_sidecar' span; "
            f"got: {names}"
        )

    def test_sidecar_span_uri_attribute(self, captured_spans, monkeypatch):
        """The 'shard.load_sidecar' span carries the sidecar URI attribute."""
        import json
        from opentelemetry import trace as _trace_api
        from adapters.storage import storage as storage_mod
        from adapters.landing import parquet_reader

        importlib.reload(parquet_reader)

        storage_mod.memory_reset()
        shard_uri = "memory://shards/test_tenant/ds/shard2.bin"
        sidecar_uri = f"{shard_uri}.meta.json"
        storage_mod.write_bytes(sidecar_uri, b"{}")

        parquet_reader.read_shard_sidecar(shard_uri)

        spans = captured_spans.get_finished_spans()
        sidecar_span = next((s for s in spans if s.name == "shard.load_sidecar"), None)
        assert sidecar_span is not None
        attrs = dict(sidecar_span.attributes or {})
        assert attrs.get("rosalinddb.uri") == sidecar_uri, (
            f"shard.load_sidecar span must carry rosalinddb.uri={sidecar_uri!r}; "
            f"got {attrs}"
        )
