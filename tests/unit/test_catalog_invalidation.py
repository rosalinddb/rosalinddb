"""Per-dataset `list_shards` cache + NOTIFY emission.

Covers two surfaces:

  1. The per-`(tenant, dataset)` `list_shards` cache wrapper that lives in
     `services/query_api/v1_query.py` and `services/ephemeral_runner/run.py`.
     Pinned contracts:
       - TTL fallback: cold call hits the source; second call within
         `RB_CATALOG_FRESHNESS_S` returns the cached value; third call after
         the TTL re-fetches.
       - Explicit invalidate hook: `invalidate(tenant, dataset)` evicts the
         entry so the next call re-fetches.
       - Bypass when the SSD tier is OFF (`RB_SHARD_TIER_BYTES` unset). The
         cache is only useful when paired with the tier's warm-hit path; with
         the tier off the wrapper must defer to the un-cached source so a
         deployment that has not opted in sees byte-for-byte unchanged
         behaviour.
       - Bypass when `RB_CATALOG_FRESHNESS_S=0` — operators can disable
         caching entirely without flipping the tier off.
       - Tenant/dataset isolation: a notify for `(tA, ds)` evicts only that
         entry; `(tB, ds)` and `(tA, other)` remain cached.

  2. The NOTIFY hook in `adapters/state/state.py:add_shard`. The Postgres
     path is mocked (no real PG in unit tests); the memory backend exposes
     a `subscribe_catalog_notify_memory` hook tests can listen to. Both
     paths must emit a payload carrying at minimum `{tenant, dataset,
     shard_uri}` so the DP's cache invalidator can route the eviction.
"""
from __future__ import annotations

import importlib
import json
from typing import List
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def state_mem(monkeypatch):
    """Fresh in-memory state module with the NOTIFY hook list cleared.

    A previous test may have subscribed a callback; clear the list so a
    leaked subscription does not leak into this test's assertions.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0
    # In-process NOTIFY hook list for the memory backend.
    state_mod._CATALOG_NOTIFY_HOOKS.clear()
    yield state_mod
    state_mod._CATALOG_NOTIFY_HOOKS.clear()


@pytest.fixture
def v1q_tier_on(monkeypatch, tmp_path):
    """Reload `services.query_api.v1_query` with the SSD tier ON.

    `RB_SHARD_TIER_BYTES` is the activation gate for the cache wrapper
    (no double-caching when the tier itself is off).
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", "65536")
    monkeypatch.setenv(
        "RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"),
    )
    # A non-zero TTL so the cache is *active* by default; individual tests
    # override via monkeypatch when they need 0.
    monkeypatch.setenv("RB_CATALOG_FRESHNESS_S", "5")
    # LISTEN deliberately OFF — the cache wrapper must work via TTL pull
    # without the LISTEN consumer.
    monkeypatch.delenv("RB_CATALOG_LISTEN", raising=False)
    import services.query_api.v1_query as v1q

    importlib.reload(v1q)
    yield v1q
    # Best-effort: a test that left an entry in the per-dataset cache
    # would not affect other modules, but clearing makes intent explicit.
    if hasattr(v1q, "_catalog_cache_clear"):
        v1q._catalog_cache_clear()


@pytest.fixture
def v1q_tier_off(monkeypatch, tmp_path):
    """Reload `services.query_api.v1_query` with the SSD tier OFF.

    The cache MUST be bypassed in this mode; pins the rollback contract.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.delenv("RB_SHARD_TIER_BYTES", raising=False)
    monkeypatch.setenv("RB_CATALOG_FRESHNESS_S", "5")
    monkeypatch.delenv("RB_CATALOG_LISTEN", raising=False)
    import services.query_api.v1_query as v1q

    importlib.reload(v1q)
    yield v1q
    if hasattr(v1q, "_catalog_cache_clear"):
        v1q._catalog_cache_clear()


@pytest.fixture
def eph_tier_on(monkeypatch, tmp_path):
    """Reload `services.ephemeral_runner.run` with the SSD tier ON."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards-eph"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", "65536")
    monkeypatch.setenv(
        "RB_SHARD_TIER_DIR", str(tmp_path / "shards-eph" / "tier-managed"),
    )
    monkeypatch.setenv("RB_CATALOG_FRESHNESS_S", "5")
    monkeypatch.delenv("RB_CATALOG_LISTEN", raising=False)
    import services.ephemeral_runner.run as eph_run

    importlib.reload(eph_run)
    yield eph_run
    if hasattr(eph_run, "_catalog_cache_clear"):
        eph_run._catalog_cache_clear()


# --- TTL fallback ---------------------------------------------------------


def test_ttl_first_hits_source_second_hits_cache_third_refetches(
    v1q_tier_on, monkeypatch
):
    """3-step TTL contract: cold -> warm-within-TTL -> cold-after-TTL.

    A mock clock drives time so the test does not sleep; the wrapper
    must consult the supplied clock rather than `time.time()` directly,
    or use the env-tunable TTL with an injectable now-source. The
    contract under test is the *count of source calls*, not how the
    wrapper measures time.
    """
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return [{"id": 1, "tenant_id": tenant, "dataset_name": dataset,
                 "shard_uri": "memory://x"}]

    monkeypatch.setattr(v1q_tier_on, "list_shards", fake_list_shards)

    # Patchable clock — the wrapper reads `_now()` so a test can advance
    # virtual time without a real sleep.
    now = {"t": 1000.0}
    monkeypatch.setattr(v1q_tier_on, "_now", lambda: now["t"])

    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1, "first call must hit the source"

    # Within TTL.
    now["t"] += 1.0
    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1, "second call within TTL must NOT re-hit the source"

    # Past TTL (default 5 s in the fixture).
    now["t"] += 10.0
    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 2, "call past TTL must re-hit the source"


def test_invalidate_evicts_single_dataset(v1q_tier_on, monkeypatch):
    """`invalidate(tenant, dataset)` forces the next call to re-fetch.

    The notify-driven path calls this on each `catalog_updates` event;
    other (tenant, dataset) entries must remain cached so a notify for
    `(tA, ds)` does not flush the whole process's cache.
    """
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return [{"id": 1, "shard_uri": "memory://x"}]

    monkeypatch.setattr(v1q_tier_on, "list_shards", fake_list_shards)
    now = {"t": 1000.0}
    monkeypatch.setattr(v1q_tier_on, "_now", lambda: now["t"])

    v1q_tier_on._cached_list_shards("tA", "ds")
    v1q_tier_on._cached_list_shards("tB", "ds")
    v1q_tier_on._cached_list_shards("tA", "other")
    assert len(calls) == 3, "three distinct datasets -> three source calls"

    # Warm hits — no new source calls.
    v1q_tier_on._cached_list_shards("tA", "ds")
    v1q_tier_on._cached_list_shards("tB", "ds")
    v1q_tier_on._cached_list_shards("tA", "other")
    assert len(calls) == 3

    # Invalidate JUST (tA, ds). The other two must remain cached.
    v1q_tier_on._invalidate_catalog_cache("tA", "ds")
    v1q_tier_on._cached_list_shards("tA", "ds")
    assert len(calls) == 4, "evicted entry must re-fetch"
    v1q_tier_on._cached_list_shards("tB", "ds")
    v1q_tier_on._cached_list_shards("tA", "other")
    assert len(calls) == 4, "neighbouring entries must remain cached"


# --- activation gates -----------------------------------------------------


def test_cache_bypassed_when_tier_off(v1q_tier_off, monkeypatch):
    """With `RB_SHARD_TIER_BYTES` unset the wrapper must defer to the source every call.

    The cache only adds value when paired with the tier's warm-hit path
    (the legacy single-flight path already caches the bytes on disk; the
    Postgres lookup is fast). Caching here on top would double-cache and
    risk staleness for a deployment that has not opted in.
    """
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return []

    monkeypatch.setattr(v1q_tier_off, "list_shards", fake_list_shards)
    now = {"t": 1000.0}
    if hasattr(v1q_tier_off, "_now"):
        monkeypatch.setattr(v1q_tier_off, "_now", lambda: now["t"])

    v1q_tier_off._cached_list_shards("t1", "ds")
    v1q_tier_off._cached_list_shards("t1", "ds")
    v1q_tier_off._cached_list_shards("t1", "ds")
    assert len(calls) == 3, (
        f"tier off -> wrapper must defer every call to source; got {len(calls)} calls"
    )


def test_cache_bypassed_when_freshness_zero(v1q_tier_on, monkeypatch):
    """`RB_CATALOG_FRESHNESS_S=0` disables the cache entirely.

    Operator emergency knob: a stale-cache bug suspected in production
    can be turned off without redeploying. Setting the TTL to 0 must
    force every call to the source.
    """
    monkeypatch.setenv("RB_CATALOG_FRESHNESS_S", "0")
    # Re-read the env so the new value takes effect.
    importlib.reload(v1q_tier_on)

    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return []

    monkeypatch.setattr(v1q_tier_on, "list_shards", fake_list_shards)
    if hasattr(v1q_tier_on, "_now"):
        now = {"t": 1000.0}
        monkeypatch.setattr(v1q_tier_on, "_now", lambda: now["t"])

    v1q_tier_on._cached_list_shards("t1", "ds")
    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 2, (
        f"freshness=0 -> wrapper must defer every call; got {len(calls)} calls"
    )


# --- NOTIFY emission ------------------------------------------------------


def test_add_shard_memory_emits_notify_hook(state_mem):
    """`add_shard` on the memory backend invokes registered notify hooks.

    The hook is the in-process equivalent of `pg_notify` for the memory
    backend used in unit tests. A registered callback must receive a
    payload dict with at minimum `{tenant, dataset, shard_uri}` so the
    DP's cache invalidator can route the eviction.
    """
    captured: list[dict] = []
    state_mem.subscribe_catalog_notify_memory(captured.append)

    state_mem.add_shard(
        "t1", "ds", "memory://idx/shard-1.bin",
        checksum="c1", vector_count=10, index_type="flat",
    )

    assert len(captured) == 1, f"expected 1 notify; got {captured}"
    payload = captured[0]
    assert payload["tenant"] == "t1"
    assert payload["dataset"] == "ds"
    assert payload["shard_uri"] == "memory://idx/shard-1.bin"


def test_add_shard_memory_hook_failure_does_not_break_insert(state_mem):
    """A subscriber that raises does NOT prevent the shard insert.

    The hook is a best-effort observability/invalidation signal, not a
    transaction participant. A buggy subscriber must not corrupt the
    catalog's source of truth.
    """
    def bad(_payload):
        raise RuntimeError("subscriber blew up")

    state_mem.subscribe_catalog_notify_memory(bad)
    sid = state_mem.add_shard(
        "t1", "ds", "memory://idx/shard-2.bin",
        checksum="c2", vector_count=10, index_type="flat",
    )
    assert sid >= 1
    # The catalog row exists, regardless of the hook failure.
    rows = state_mem.list_shards("t1", "ds")
    assert len(rows) == 1
    assert rows[0]["shard_uri"] == "memory://idx/shard-2.bin"


def test_add_shard_pg_emits_pg_notify(monkeypatch):
    """The Postgres path runs `pg_notify('catalog_updates', payload)`.

    Real PG is not available in unit tests, so we patch `pooled_conn` to
    yield a fake connection that records every `cur.execute(...)` call.
    The contract under test is that the INSERT statement is followed (or
    preceded — order does not matter as long as both ride the same
    transaction) by a `pg_notify` invocation carrying the same payload
    keys the memory hook delivers.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub/forced-pg-mode")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)

    executed: list[tuple[str, tuple]] = []

    class _FakeCur:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=()):
            executed.append((sql, tuple(params) if params else ()))

        def fetchone(self):
            # Mimic `RETURNING id` from the INSERT.
            return (42,)

        def close(self):
            pass

    class _FakeConn:
        def cursor(self, *args, **kwargs):
            return _FakeCur()

    import contextlib

    @contextlib.contextmanager
    def fake_pooled_conn(*args, **kwargs):
        yield _FakeConn()

    monkeypatch.setattr(state_mod, "pooled_conn", fake_pooled_conn)

    sid = state_mod.add_shard(
        "t1", "ds", "memory://idx/shard-pg.bin",
        checksum="cpg", vector_count=10, index_type="flat",
    )
    assert sid == 42

    # Find the pg_notify call.
    notify_calls = [
        (sql, params) for sql, params in executed if "pg_notify" in sql.lower()
    ]
    assert len(notify_calls) == 1, (
        f"expected exactly one pg_notify; got {len(notify_calls)} from {executed}"
    )
    _sql, params = notify_calls[0]
    # The payload is a single JSON-encoded parameter. Its dict must carry
    # at minimum tenant/dataset/shard_uri.
    payload_str = params[-1] if isinstance(params[-1], str) else params[0]
    payload = json.loads(payload_str)
    assert payload["tenant"] == "t1"
    assert payload["dataset"] == "ds"
    assert payload["shard_uri"] == "memory://idx/shard-pg.bin"


# --- LISTEN-driven invalidation integration -------------------------------


def test_listen_callback_invalidates_cache(v1q_tier_on, monkeypatch):
    """A NOTIFY payload routed through the listener evicts the cached entry.

    Pins the wiring between `services._common.catalog_listener` and the
    per-dataset cache: the listener delivers a `dict` payload; the
    cache's notify handler reads `tenant` + `dataset` and calls
    `_invalidate_catalog_cache(tenant, dataset)`.

    The test bypasses the real listener thread by calling the cache's
    handler directly — the listener's own delivery contract is covered
    in `test_catalog_listener.py`. This test guards the handler shape
    and the eviction it triggers.
    """
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return []

    monkeypatch.setattr(v1q_tier_on, "list_shards", fake_list_shards)
    now = {"t": 1000.0}
    monkeypatch.setattr(v1q_tier_on, "_now", lambda: now["t"])

    # Warm cache.
    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1
    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1  # cached

    # Simulate a listener-delivered notify.
    v1q_tier_on._on_catalog_notify(
        {"tenant": "t1", "dataset": "ds", "shard_uri": "memory://idx/new"}
    )

    v1q_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 2, "post-notify call must re-fetch (cache evicted)"


# --- ephemeral runner mirror ----------------------------------------------


def test_eph_ttl_first_hits_source_second_hits_cache(eph_tier_on, monkeypatch):
    """Mirror of the v1_query TTL test for the ephemeral runner.

    The ephemeral path is a duplicate `_ensure_cached` for the same
    rationale (avoid the circular import); the catalog cache must be
    duplicated for the same reason and must behave identically.
    """
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return [{"id": 1, "shard_uri": "memory://x"}]

    monkeypatch.setattr(eph_tier_on, "list_shards", fake_list_shards)
    now = {"t": 1000.0}
    monkeypatch.setattr(eph_tier_on, "_now", lambda: now["t"])

    eph_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1
    now["t"] += 1.0
    eph_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1
    now["t"] += 10.0
    eph_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 2


def test_eph_invalidate_evicts(eph_tier_on, monkeypatch):
    """Mirror invalidate test for the ephemeral runner."""
    calls: list[tuple[str, str]] = []

    def fake_list_shards(tenant, dataset):
        calls.append((tenant, dataset))
        return []

    monkeypatch.setattr(eph_tier_on, "list_shards", fake_list_shards)
    now = {"t": 1000.0}
    monkeypatch.setattr(eph_tier_on, "_now", lambda: now["t"])

    eph_tier_on._cached_list_shards("t1", "ds")
    eph_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 1
    eph_tier_on._invalidate_catalog_cache("t1", "ds")
    eph_tier_on._cached_list_shards("t1", "ds")
    assert len(calls) == 2
