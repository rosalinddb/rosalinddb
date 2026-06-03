from __future__ import annotations

"""State adapter for dataset and shard catalogs.

Supports an in-memory mode (default when `DATABASE_URL` starts with `memory://`)
and a Postgres-backed mode otherwise. Provides functions to create schema,
manage datasets, and track index shards.

Every dataset/shard function takes `tenant_id` as its FIRST positional
argument. The in-memory mode keys `_MEM_DATASETS` by `(tenant_id, dataset_name)`
tuples and adds a `tenant_id` field to every `_MEM_SHARDS` row. Migration
002_datasets_tenant_isolation.sql rewrites the catalog tables to include
`tenant_id` in the primary key.
"""

import contextlib
import contextvars
import datetime as _dt
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

from adapters.observability.tracing import state_connect_span


_MEMORY_MODE = os.getenv("DATABASE_URL", "memory://local").startswith("memory://")

# A single process-wide lock guards the in-memory quota counters so that the
# lazy daily reset + the consume check + the increment happen as one atomic
# step. Postgres mode does not use this — it relies on a single
# `UPDATE ... WHERE ... RETURNING` statement, which the DB serialises for us.
_MEM_QUOTA_LOCK = threading.Lock()

# `migrate()` is idempotent, but its idempotent DDL still takes brief
# `AccessExclusiveLock`s on Postgres. When several services run as threads in
# one process (a single-process dev/test harness) their `migrate()` calls would
# race and deadlock on those locks. This lock + run-once flag serialise
# migration within a process: the first caller applies the schema, the rest
# return immediately. Harmless in production where each service process calls
# `migrate()` once.
_MIGRATE_LOCK = threading.Lock()
_MIGRATED = False

# Datasets are keyed by (tenant_id, dataset_name) so two tenants can reuse
# the same dataset name without collision. Shards carry tenant_id so the
# index builder and ephemeral runner can filter without joining back through
# the dataset table.
_MEM_DATASETS: dict[Tuple[str, str], dict] = {}
_MEM_SHARDS: list[dict] = []
_MEM_SHARD_ID = 0

# In-memory tenant + api_key stores. Indexed by primary identifier for
# O(1) lookup; we also expose `get_tenant_by_email` and `get_tenant_by_id`
# so callers do not depend on which key we used.
_MEM_TENANTS: dict[str, dict] = {}  # tenant_id -> row
_MEM_TENANTS_BY_EMAIL: dict[str, str] = {}  # email -> tenant_id

# In-memory api_keys store. A flat list preserving insertion order (used by
# `list_api_keys`), plus a `key_hash -> row` index so auth-time resolution
# is O(1) — see `get_api_key_by_hash`. Postgres mode reads from the
# `api_keys` table populated by migration 001_tenants_and_keys.sql; the
# `api_keys_hash_idx` index makes the equivalent lookup O(1) there too.
# Rows carry SHA-256(raw_key), not the raw key: the raw key only exists at
# creation time and is handed to the caller in the HTTP response. SHA-256
# (not bcrypt) is used so the hash is deterministic and directly indexable
# — correct for the high-entropy random `rb_live_` tokens (bcrypt stays for
# passwords).
_MEM_API_KEYS: list[dict] = []  # rows, insertion order
_MEM_API_KEYS_BY_HASH: dict[str, dict] = {}  # key_hash -> row

# In-process notify hooks for catalog invalidation (memory backend). Postgres
# mode emits `pg_notify('catalog_updates', payload)` from `add_shard`, which
# the DP's `services._common.catalog_listener` thread relays to its
# subscribers. The memory backend has no Postgres connection to NOTIFY through,
# so unit tests that need to observe an invalidation register a hook here
# instead — see `tests/unit/test_catalog_invalidation.py`. Hooks fire in
# registration order and a raising hook is logged-and-skipped (best-effort:
# the catalog INSERT must not depend on a subscriber's correctness).
_CATALOG_NOTIFY_HOOKS: list[Callable[[dict], None]] = []
_CATALOG_NOTIFY_HOOKS_LOCK = threading.Lock()


def subscribe_catalog_notify_memory(callback: Callable[[dict], None]) -> Callable[[dict], None]:
    """Register a hook fired on `add_shard` in memory mode.

    Returns the callback so a test can pass it directly to `unsubscribe`
    without keeping a separate handle. Idempotent: registering the same
    callback twice fires it twice (mirrors Postgres LISTEN semantics —
    each subscriber gets one delivery per notify).
    """
    with _CATALOG_NOTIFY_HOOKS_LOCK:
        _CATALOG_NOTIFY_HOOKS.append(callback)
    return callback


def unsubscribe_catalog_notify_memory(callback: Callable[[dict], None]) -> bool:
    """Remove a previously registered hook. Returns True if removed."""
    with _CATALOG_NOTIFY_HOOKS_LOCK:
        try:
            _CATALOG_NOTIFY_HOOKS.remove(callback)
            return True
        except ValueError:
            return False


def _fire_catalog_notify_memory(payload: dict) -> None:
    """Fan a notify payload out to memory-mode hook subscribers.

    A subscriber that raises is logged-and-skipped — the catalog insert
    has already completed before this runs, and we MUST NOT let a buggy
    cache-invalidation subscriber make the catalog row look like it
    failed. Snapshot the list under the lock so a concurrent
    unsubscribe does not mutate it mid-iteration.
    """
    with _CATALOG_NOTIFY_HOOKS_LOCK:
        snapshot = list(_CATALOG_NOTIFY_HOOKS)
    for cb in snapshot:
        try:
            cb(payload)
        except Exception:  # noqa: BLE001 - best-effort dispatch
            logging.getLogger(__name__).warning(
                "catalog notify hook raised; continuing", exc_info=True,
            )


# In-memory import_jobs store. Keyed by (tenant_id, import_id) so the
# cross-tenant 404 contract is enforced in the data layer, exactly like
# `_MEM_DATASETS`. Postgres mode reads from the `import_jobs` table created
# by migration 004_import_jobs.sql.
_MEM_IMPORTS: dict[Tuple[str, str], dict] = {}  # (tenant_id, import_id) -> row
# Monotonic insertion counter for `_MEM_IMPORTS`. `created_at` is only
# second-resolution, so several jobs created within one second would tie on a
# newest-first sort; this sequence is the tiebreaker. Postgres mode does not
# need it — `now()` there is microsecond-resolution.
_MEM_IMPORT_SEQ = 0

# In-memory mirror of the `dp_shard_residency` table (SSD-cache feature).
# Keyed by `(dp_id, shard_uri)` so the PK semantics match the SQL branch —
# two DPs holding the same shard are independent rows. The value tuple is
# `(warm_since, last_query_at)` matching the column order. Postgres mode
# reads/writes the `dp_shard_residency` table populated by migration
# 007_dp_shard_residency.sql; the writer (services/_common/residency_writer.py)
# is the only producer in either backend.
_MEM_DP_RESIDENCY: dict[Tuple[str, str], Tuple[float, float]] = {}
_MEM_DP_RESIDENCY_LOCK = threading.Lock()


def _dsn() -> str:
    """Return the Postgres DSN from the environment (Postgres mode only)."""
    return os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectors"
    )


def _conn():
    """Return a live, DEDICATED psycopg2 connection (Postgres mode only).

    NOT pooled: every call opens a fresh dedicated connection the caller owns
    and must close itself. This is load-bearing for `dataset_build_lock` — a
    session-level `pg_advisory_lock` must be acquired and released on the SAME
    connection, and the connection must not be shared with or returned to
    anything else while the lock is held. A pooled connection returned while
    still holding a session lock would leak the lock to the next borrower, so
    `dataset_build_lock` (and the startup migration path) keep using this.

    General query/write work goes through `pooled_conn()` instead — see that
    function. `_conn()` is reserved for callers with a connection-scope
    dependency that the pool cannot honour.
    """
    if _MEMORY_MODE:
        raise RuntimeError("memory state has no SQL connection")
    # Unannotated `state.connect` span — the legacy dedicated/unpooled path. A
    # genuine connect every time; the annotated reuse/open variants are emitted
    # by `pooled_conn()`.
    with state_connect_span():
        return psycopg2.connect(_dsn())


# --- Application-side connection pool -------------------------------------
#
# `_conn()` opened a brand-new connection per call — a full TCP + TLS + auth
# handshake (~15ms against a managed database). A single `POST /v1/query`
# opened five, spending the bulk of the request purely on handshakes. The pool
# below opens a small set of connections once and hands them out / takes them
# back via `pooled_conn()`.
#
# One pool PER PROCESS. Each uvicorn worker / worker process group builds its
# own — that is correct and expected; a `psycopg2.pool` object cannot be
# shared across processes. The managed database has a finite connection cap,
# so the per-process ceiling is kept small and is overridable (see below).

_DEFAULT_PG_POOL_MAX = 10  # per-process ceiling; override with RB_PG_POOL_MAX
_PG_POOL_MIN = 1  # connections kept warm even when idle
#
# Pool sizing assessment:
#
# Before the caching fixes, a warm query hit Postgres 4 times per request:
#   (1) get_api_key_by_hash      (auth lookup)
#   (2) touch_api_key_last_used  (auth touch)
#   (3) try_consume_query        (quota write)
#   (4) get_tenant_dp_pool       (routing lookup)
# With the request-scoped connection these shared one pool checkout, so the
# effective pool pressure was 1 checkout per request but 4 sequential DB
# round-trips holding the connection.
#
# After the caching fixes:
#   (1) auth lookup   → skipped on cache hit (auth cache)
#   (2) auth touch    → fire-and-forget on daemon thread; 1 standalone checkout
#                       fired off critical path, ~once per 30 s per key
#   (3) try_consume_query → remains (quota write, standalone=True)
#   (4) dp_pool lookup → skipped on cache hit (dp_pool cache)
# Per-query pool pressure is now ~1 checkout (just quota) on warm requests,
# ~2-3 on cold (first-seen key / TTL expiry: auth lookup + maybe touch + quota).
# This is a ~4× reduction in pool pressure.
#
# Conclusion: leave _DEFAULT_PG_POOL_MAX at 10. With the caching above, 10
# connections per worker is comfortably adequate for the expected concurrency.
# Raising it is NOT done here because:
#   - Managed Postgres providers commonly cap total connections per database;
#     this per-process pool cap leaves headroom for multiple worker processes
#     before that ceiling is hit. The deep fix is a connection pooler /
#     PgBouncer fan-out, NOT a larger per-process pool.
#   - The caching fix reduces pressure ~4×, making the current pool size
#     adequate without a cap increase.

# Block-with-timeout pool checkout.
#
# `ThreadedConnectionPool.getconn()` is fail-fast — the instant the pool is
# exhausted it raises `psycopg2.pool.PoolError`. Under a burst that became an
# HTTP 500. `pooled_conn()` instead poll-retries on `PoolError` until a total
# deadline; only a genuinely *sustained* exhaustion (deadline exceeded) raises
# `PoolCheckoutTimeout`, which the ASGI apps map to a 503 — never a 500.
_DEFAULT_POOL_CHECKOUT_TIMEOUT_S = 2.5  # total deadline; RB_PG_POOL_CHECKOUT_TIMEOUT_S
_POOL_CHECKOUT_POLL_S = 0.015  # sleep between fail-fast getconn() retries (~15ms)


class PoolCheckoutTimeout(RuntimeError):
    """The pool stayed exhausted past the block-with-timeout deadline.

    Raised by `pooled_conn()` when every retry of `getconn()` hit a
    `PoolError` and the total checkout deadline elapsed. It signals a genuine
    *sustained* overload — the ASGI apps map it to HTTP 503 (service
    unavailable), never a 500. A transient exhaustion that clears within the
    deadline is invisible: the checkout simply blocks then succeeds.
    """

# The process-wide pool. Built lazily on the first `pooled_conn()` checkout
# (NOT at import time — in tests `DATABASE_URL` is `memory://local`, which has
# no SQL server and no pool). `_POOL_LOCK` guards construction so two threads
# racing the first checkout do not build two pools.
_POOL: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_POOL_LOCK = threading.Lock()


def _pool_max_size() -> int:
    """Return the per-process pool ceiling.

    Defaults to `_DEFAULT_PG_POOL_MAX` (10). `RB_PG_POOL_MAX`, when set to a
    positive integer, overrides it — useful to fit the pool under a managed
    database's connection cap when many worker processes run side by side. A
    missing or malformed value falls back to the default rather than crashing.
    """
    raw = os.getenv("RB_PG_POOL_MAX")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return _DEFAULT_PG_POOL_MAX


def _pool_checkout_timeout_s() -> float:
    """Return the total block-with-timeout deadline for a pool checkout.

    Defaults to `_DEFAULT_POOL_CHECKOUT_TIMEOUT_S` (2.5s). `RB_PG_POOL_CHECKOUT_TIMEOUT_S`,
    when set to a positive number, overrides it — so an operator can tune how
    long a request waits for a free connection before the pool checkout gives
    up with a 503. Read live (per call) so a test can retune it without a
    module reload. A missing or malformed value falls back to the default.
    """
    raw = os.getenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return _DEFAULT_POOL_CHECKOUT_TIMEOUT_S


def _get_pool(maxconn_override: Optional[int] = None) -> psycopg2.pool.ThreadedConnectionPool:
    """Return the process-wide connection pool, building it lazily on first use.

    `ThreadedConnectionPool` (not `SimpleConnectionPool`) because the services
    run threaded under uvicorn — checkout/return must be thread-safe.

    `maxconn_override` forces the pool's max size; it is a test hook (a tiny
    max-1 pool makes a connection leak immediately fatal). In production the
    size comes from `_pool_max_size()`.
    """
    global _POOL
    if _MEMORY_MODE:
        raise RuntimeError("memory state has no SQL connection pool")
    if _POOL is None:
        with _POOL_LOCK:
            # Re-check under the lock — another thread may have built it while
            # this one waited.
            if _POOL is None:
                maxconn = maxconn_override or _pool_max_size()
                _POOL = psycopg2.pool.ThreadedConnectionPool(
                    minconn=_PG_POOL_MIN,
                    maxconn=max(_PG_POOL_MIN, maxconn),
                    dsn=_dsn(),
                )
    return _POOL


def _close_pool() -> None:
    """Close every connection in the pool and discard it.

    Mainly a test teardown hook so a test that rebinds `DATABASE_URL` does not
    leave a pool pinned to a stopped container. Safe to call when no pool
    exists. Production processes are long-lived and let the pool live for the
    process lifetime.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.closeall()
            _POOL = None


# --- request-scoped connection ------------------------------------------------
#
# Each `pooled_conn()` block historically did its own `getconn`/`commit`/
# `putconn`, so an HTTP request that called five state functions cycled the
# pool five times — five checkouts, five commits, five returns. The k6 sweeps
# showed that churn dominating a request and exhausting the pool under burst.
#
# The `request_scoped_connection()` context manager (driven by the ASGI
# middleware in `adapters/state/conn_middleware.py`) checks ONE connection out
# of the pool for the whole request, binds it to `_REQUEST_CONN`, and returns
# it to the pool exactly once at request end. While a request connection is
# bound, `pooled_conn()` yields THAT connection and does NOT commit/return on
# block exit — the middleware owns the lifecycle (one commit / one rollback /
# one return per request). When nothing is bound (worker processes, scripts,
# tests, non-HTTP callers) `pooled_conn()` behaves EXACTLY as before: its own
# per-call checkout + commit + return.
#
# `contextvars.ContextVar` is the right primitive: it is task/thread-local and
# is *copied* into the thread `asyncio.to_thread` runs work on, so a connection
# bound by the middleware before an offloaded sync state call is still visible
# inside that call. `_conn()`-based advisory-lock flows are deliberately NOT
# routed through here — they need their own dedicated session connection.
_REQUEST_CONN: contextvars.ContextVar[Optional["psycopg2.extensions.connection"]] = (
    contextvars.ContextVar("rb_request_conn", default=None)
)


def _getconn_with_timeout(pool, deadline_s: float) -> "psycopg2.extensions.connection":
    """Check a connection out of `pool`, blocking up to `deadline_s` seconds.

    `ThreadedConnectionPool.getconn()` is fail-fast: it raises
    `psycopg2.pool.PoolError` the instant the pool is exhausted. This wrapper
    turns that into a block-with-timeout: on `PoolError` it sleeps a short
    poll interval and retries until the total deadline elapses. A transient
    exhaustion that clears within the deadline is therefore invisible (the
    checkout just blocks then succeeds). A *sustained* exhaustion raises
    `PoolCheckoutTimeout`, which the apps map to a 503 — never the bare 500
    a raw `PoolError` would become.
    """
    start = time.monotonic()
    while True:
        try:
            return pool.getconn()
        except psycopg2.pool.PoolError:
            if time.monotonic() - start >= deadline_s:
                raise PoolCheckoutTimeout(
                    "connection pool exhausted: no connection became available "
                    f"within {deadline_s:.2f}s"
                )
            time.sleep(_POOL_CHECKOUT_POLL_S)


@contextlib.contextmanager
def pooled_conn(
    maxconn_override: Optional[int] = None,
    *,
    standalone: bool = False,
) -> Iterator["psycopg2.extensions.connection"]:
    """Check a connection out of the pool; commit/rollback and return on exit.

    This is the pool-aware replacement for `with _conn() as conn`. A psycopg2
    connection used as a context manager only manages the *transaction* — it
    does NOT close (or return) the connection. So the old `with _conn()` call
    sites never returned their connection to anything; with a pool that would
    be a leak.

    Two modes, decided by the `_REQUEST_CONN` contextvar:

    * **Request-scoped** — a request connection IS bound (the ASGI middleware
      bound it). `pooled_conn()` simply yields that ONE connection and does
      NOT commit/rollback/return on block exit: the middleware owns the
      lifecycle (one commit on a clean response, one rollback on an exception,
      one return at request end). N `pooled_conn()` blocks in a request share
      the single connection → one checkout, not N.

    * **Standalone** — nothing is bound (worker processes, scripts, tests,
      non-HTTP callers). `pooled_conn()` behaves EXACTLY as before:
        - on enter: check a connection out of the pool (block-with-timeout);
        - on normal exit: `commit()`, then return the connection to the pool;
        - on exception: `rollback()`, then return the connection, then
          re-raise — the connection is returned in EVERY path, never leaked,
          and a borrower never inherits an aborted transaction.

    `standalone=True` **forces** the standalone path even when a request
    connection IS bound: it IGNORES `_REQUEST_CONN` and always does its own
    block-with-timeout `getconn` + `commit`/`rollback` + `putconn`. This is for
    writes that MUST commit and release their row locks immediately, rather
    than riding the request-scoped transaction that only commits at request end.
    A hot-row `UPDATE` — e.g. `try_consume_query` / `try_consume_vectors` —
    held under the request scope would keep its row lock for the WHOLE request,
    including the CP→DP proxy round-trip, serialising concurrent requests from
    one tenant. Passing `standalone=True` commits the write in its own short
    transaction so the lock is released at once. The write is then NOT part of
    the request transaction and is NOT rolled back if the request later fails
    — which is the intended semantics for an already-counted quota consumption.

    `maxconn_override` is a test hook forwarded to `_get_pool` (see there). It
    is ignored in request-scoped mode — the request connection already exists.

    On sustained pool exhaustion the checkout raises `PoolCheckoutTimeout`
    (mapped to HTTP 503), instead of the fail-fast `PoolError` (an HTTP 500).

    Memory mode has no pool — callers must branch on `_MEMORY_MODE` before
    reaching here, exactly as they already do for `_conn()`.
    """
    bound = None if standalone else _REQUEST_CONN.get()
    if bound is not None:
        # Request-scoped: reuse the middleware-owned connection. No span (the
        # connection was already opened/reused once for the request), no
        # commit, no putconn — the middleware does all three exactly once.
        yield bound
        return

    pool = _get_pool(maxconn_override=maxconn_override)
    # `state.connect` span — annotated so a trace distinguishes a ~0ms reuse
    # from a genuine new-backend open. The pool keeps `_PG_POOL_MIN` warm and
    # only opens a new backend when every kept connection is checked out, so a
    # checkout that finds a free connection in `_pool` is a reuse.
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        conn = _getconn_with_timeout(pool, _pool_checkout_timeout_s())
    try:
        yield conn
    except BaseException:
        # Roll back so the connection is returned to the pool clean — never
        # carrying a half-applied or aborted transaction into the next
        # borrower — then re-raise.
        try:
            conn.rollback()
        finally:
            pool.putconn(conn)
        raise
    else:
        conn.commit()
        pool.putconn(conn)


def _checkout_request_conn(
    maxconn_override: Optional[int] = None,
) -> Optional["psycopg2.extensions.connection"]:
    """Check ONE connection out of the pool for an HTTP request.

    Returns the connection, or `None` in memory mode (no pool). This is the
    *blocking* half of the request-scoped lifecycle — the ASGI middleware runs
    it off the event loop. Binding the contextvar and commit/rollback/return
    are handled by `bind_request_conn` / `finish_request_conn`, which the
    middleware runs in its own coroutine context so the binding is visible to
    the route handler (and to off-loop sync state calls via the copied
    `contextvars`).
    """
    if _MEMORY_MODE:
        return None
    pool = _get_pool(maxconn_override=maxconn_override)
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        return _getconn_with_timeout(pool, _pool_checkout_timeout_s())


def bind_request_conn(
    conn: Optional["psycopg2.extensions.connection"],
) -> "contextvars.Token":
    """Bind a request connection to `_REQUEST_CONN`; return the reset token.

    Called by the ASGI middleware IN ITS OWN COROUTINE CONTEXT (not in a
    worker thread) so the binding is visible to the downstream route handler.
    `conn` may be `None` (memory mode) — binding `None` is the no-op case and
    `pooled_conn()` then falls through to its standalone path.
    """
    return _REQUEST_CONN.set(conn)


def unbind_request_conn(token: "contextvars.Token") -> None:
    """Reset `_REQUEST_CONN` to its pre-request value.

    Must run in the SAME context `bind_request_conn` ran in — the ASGI
    middleware calls both in its coroutine. A no-op-safe pair: if `set`
    bound `None` (memory mode) this still cleanly unwinds.
    """
    _REQUEST_CONN.reset(token)


def finish_request_conn(
    conn: Optional["psycopg2.extensions.connection"],
    *,
    failed: bool,
) -> None:
    """Commit/rollback the request transaction and return the connection.

    The closing half of the request-scoped lifecycle. `failed` is True when
    the request raised; the single request transaction is rolled back, else
    committed. The connection is returned to the pool exactly once. A no-op
    when `conn` is `None` (memory mode).

    This is blocking I/O (commit + `putconn`) — the ASGI middleware runs it
    off the event loop. The contextvar reset is separate (`unbind_request_conn`),
    so this function is context-agnostic and safe to run in a worker thread.
    """
    if conn is None:
        return
    pool = _get_pool()
    try:
        if failed:
            conn.rollback()
        else:
            conn.commit()
    finally:
        pool.putconn(conn)


@contextlib.contextmanager
def request_scoped_connection(
    maxconn_override: Optional[int] = None,
) -> Iterator[None]:
    """Bind ONE pooled connection to the request for its whole lifetime.

    Driven by the ASGI middleware (`adapters/state/conn_middleware.py`). It
    checks a single connection out of the pool, binds it to `_REQUEST_CONN`
    for the duration of the `with` body (the HTTP request), and on exit:

      - clean exit  -> `commit()` once, return the connection to the pool;
      - exception   -> `rollback()` once, return the connection, re-raise.

    Every `pooled_conn()` block inside the body transparently reuses this one
    connection (see `pooled_conn`), so a request that touches N state
    functions costs ONE pool checkout instead of N.

    Memory mode has no pool — this is a clean no-op there (it yields without
    binding anything), so the middleware stays correct in `memory://` tests.

    Transaction semantics: this collapses a request's previously-separate
    per-call transactions into ONE request-scoped transaction (a single commit
    at request end). Within a request, a later `pooled_conn()` block now reads
    on the SAME connection as an earlier one, so it sees the earlier block's
    uncommitted writes — which is fine and generally desirable. `_conn()`-based
    advisory-lock flows are separate and keep their own dedicated connection.
    """
    if _MEMORY_MODE:
        # No SQL pool in memory mode — nothing to bind. The middleware calls
        # this unconditionally; here it is a pure no-op.
        yield
        return

    pool = _get_pool(maxconn_override=maxconn_override)
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        conn = _getconn_with_timeout(pool, _pool_checkout_timeout_s())
    token = _REQUEST_CONN.set(conn)
    try:
        yield
    except BaseException:
        try:
            conn.rollback()
        finally:
            _REQUEST_CONN.reset(token)
            pool.putconn(conn)
        raise
    else:
        try:
            conn.commit()
        finally:
            _REQUEST_CONN.reset(token)
            pool.putconn(conn)


def migrate(force: bool = False):
    """Apply control-plane schema migrations in order: 001 → … → 008.

    001 (tenants/keys), 002 (datasets tenant isolation), 003 (shard catalog
    incremental-indexing columns), 004 (import_jobs for async bulk ingest),
    005 (dataset `status_updated_at` for the reconciliation reaper), 006
    (`tenants.dp_pool` routing), 007 (`dp_shard_residency` SSD-cache registry),
    008 (`shard_catalog.durable_lsn`, the delta-tier watermark — default 0, so
    it is a no-op for behaviour until the delta tier is enabled).
    002 references the `tenants` table for FKs so it must run after 001; 003
    recreates `shard_catalog` so it must run after 002; 004 FKs to
    `dataset_catalog` so it must run after 002. All files live under
    `adapters/state/migrations/`. Memory mode is a no-op.

    This migrates ONLY the control-plane Postgres. The hot-tier (pgvector)
    instance has its own DSN (`RB_HOT_DSN`) and a separate runner, `migrate_hot()`
    — see that function and `scripts/migrate.py`.

    `_MIGRATE_LOCK` is a `threading.Lock` — it serialises migration *within* a
    single process only. Cross-process safety is provided by a Postgres
    advisory lock taken inside `_apply_migrations()` (`pg_advisory_xact_lock`):
    several *separate processes* all calling `migrate()` at once (a
    multi-process deployment, or `scripts/multiworker_all.py` starting N
    uvicorn workers + the pipeline workers together) serialise in the database
    instead of deadlocking on the `AccessExclusiveLock`s the migration DDL
    takes. The first process to grab the advisory lock applies the schema; the
    rest block, then re-enter `_apply_migrations()`. That re-run is a genuine
    no-op — `_apply_migrations()` tracks applied migration versions in a
    `schema_migrations` ledger and skips every version already recorded, so a
    loser never re-runs the destructive `DROP TABLE` in 002/003 and never
    touches data. (The advisory lock alone would only serialise data loss, not
    prevent it — see `_apply_migrations()`.)

    When migrations run as a separate one-shot step (e.g. an init container
    or k8s job), services can be started with `RB_SKIP_MIGRATE=1` so they do
    NOT migrate on boot — the schema is applied out-of-band. Unset (the
    default), behaviour for a single process is unchanged: the first caller
    applies the schema.

    `force=True` bypasses the `RB_SKIP_MIGRATE` early-return: this is the one
    caller (`scripts/migrate.py`, the dedicated migration entrypoint) whose
    entire job IS to apply the schema. The migration runner typically inherits
    the same env as the services (so it too sees `RB_SKIP_MIGRATE=1`) —
    without `force` it would short-circuit and apply nothing, and the workers
    skip migration too, so a fresh deploy would come up with no schema.
    `force` only skips the env guard; everything else (the `threading.Lock`,
    `_apply_migrations()` with its `pg_advisory_xact_lock` + `schema_migrations`
    ledger) is unchanged.
    """
    if _MEMORY_MODE:
        # Memory mode has no schema to apply, but the OSS bootstrap "default"
        # tenant must still exist so that — when `RB_REQUIRE_AUTH` is off —
        # `current_tenant_id` resolving to "default" can write
        # tenant-keyed rows that look up successfully against the in-memory
        # tenants store (see _bootstrap_default_tenant_memory).
        _bootstrap_default_tenant_memory()
        return
    if not force and os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes"):
        # Schema applied out-of-band (see docstring) — skip to avoid the
        # cross-process AccessExclusiveLock deadlock. `force=True` (the release
        # command) bypasses this: it must apply the schema regardless.
        return
    global _MIGRATED
    with _MIGRATE_LOCK:
        if _MIGRATED:
            return
        _apply_migrations()
        _MIGRATED = True


def _bootstrap_default_tenant_memory() -> None:
    """Idempotently seed the bootstrap "default" tenant in the memory store.

    The OSS opt-in path (`RB_REQUIRE_AUTH` unset/false) collapses every
    request's tenant_id to the literal string "default" via
    `current_tenant_id`'s short-circuit. Postgres mode seeds the row in
    `_apply_migrations()` (see the INSERT ... ON CONFLICT DO NOTHING at the
    end of that function); memory mode mirrors that here so the two adapters
    behave the same.

    Postgres mode is a guarded no-op — the in-memory dicts are not the
    authoritative store there and the migration runner has already taken
    care of the row. Memory mode (and only memory mode) seeds the dict.

    `password_hash='!disabled!'` is a non-bcrypt sentinel — bcrypt.checkpw
    will never match — so even if a self-hoster later flips
    `RB_REQUIRE_AUTH=true`, no one can `/auth/login` as this tenant.
    """
    if not _MEMORY_MODE:
        return
    if "default" in _MEM_TENANTS:
        return
    vector_quota, query_quota = _quota_defaults()
    row = {
        "id": "default",
        "email": "self-host@localhost",
        "password_hash": "!disabled!",
        "plan": "oss",
        "vector_quota": vector_quota,
        "daily_query_quota": query_quota,
        "vectors_used": 0,
        "queries_today": 0,
        "queries_reset_at": _dt.date.today().isoformat(),
        "created_at": _now_iso(),
    }
    _MEM_TENANTS["default"] = row
    _MEM_TENANTS_BY_EMAIL["self-host@localhost"] = "default"


# Constant 64-bit key for the schema-migration advisory lock. Any process
# applying migrations takes `pg_advisory_xact_lock(_MIGRATE_LOCK_KEY)` first,
# so concurrent callers serialise in Postgres instead of deadlocking on the
# `AccessExclusiveLock`s the idempotent DDL takes. The value is arbitrary but
# fixed and namespaced to this purpose; it must never change once deployed.
_MIGRATE_LOCK_KEY = 0x726F73616C696E64  # ASCII "rosalind", a stable constant


# Ordered list of migration versions. The version string is the filename
# stem; this list IS the canonical apply order. Appending a new migration
# means adding its stem here — never reorder or remove an existing entry, the
# `schema_migrations` ledger keys on these exact strings.
_MIGRATION_VERSIONS = (
    "001_tenants_and_keys",
    "002_datasets_tenant_isolation",
    "003_shard_incremental_indexing",
    "004_import_jobs",
    "005_dataset_status_updated_at",
    "006_tenants_dp_pool",
    "007_dp_shard_residency",
    "008_shard_durable_lsn",
)


# Ordered list of HOT-instance migration versions. These run against the
# SEPARATE data-plane pgvector instance (RB_HOT_DSN), NOT the control-plane
# Postgres above (see docs/architecture/delta-tier.md, "Blast radius"). They
# live under `migrations/hot/` and are applied by `migrate_hot()` with the same
# advisory-lock + version-ledger discipline as the control-plane set. The two
# ledgers live in different databases by design and never share a connection.
_HOT_MIGRATION_VERSIONS = ("001_hot_vectors",)


def _apply_migrations() -> None:
    """Apply the ordered migration files exactly once each (version-tracked).

    Cross-process safety: the transaction first takes a transaction advisory
    lock (`pg_advisory_xact_lock`). Only one process holds it at a time — the
    rest block until the holder's transaction commits and releases it. The
    lock auto-releases when the transaction ends (the `with _conn()` block
    commits), so there is no explicit unlock and a crashed process cannot
    leave it stuck.

    Genuine idempotency via version tracking (Finding 1 fix). The advisory
    lock only *serialises* migrators — it does not make a re-run safe. Two of
    the migration files (002, 003) do `DROP TABLE ... CASCADE` + unconditional
    `CREATE TABLE`. Re-running them drops the catalogs and recreates them
    empty, silently destroying any rows another process committed in between.
    So `_apply_migrations()` now keeps a `schema_migrations(version, ...)`
    ledger and applies a migration file ONLY IF its version is not already
    recorded. A re-run of `migrate()` therefore applies nothing and touches no
    data — the destructive 002/003 run exactly once, on a fresh DB.

    Bootstrap of a legacy DB. A dev database created by the old (no-ledger)
    code already has 001-005 applied but no `schema_migrations` table. When
    this function first creates the ledger on such a DB it must NOT then re-run
    (and drop) everything. Detection: if a core table (`tenants`) already
    exists when the ledger was just created, the schema is already current —
    every current version is backfilled into the ledger as applied before the
    apply loop runs, so the loop finds nothing to do. A genuinely fresh DB has
    no `tenants` table, so nothing is backfilled and all migrations apply
    normally.
    """
    migrations_dir = Path(__file__).parent / "migrations"
    # A dedicated, non-pooled connection. Migrations run once at startup, off
    # the hot path, and before the pool is otherwise needed — there is nothing
    # to gain from pooling them, and a dedicated connection keeps the pool
    # unbuilt until the schema actually exists. `contextlib.closing` closes the
    # connection on every exit path; the inner `with conn` manages only the
    # transaction (commit on success / rollback on error).
    with contextlib.closing(_conn()) as conn, conn, conn.cursor() as cur:
        # Serialise concurrent migrators in the database. Blocking (not
        # `try`): a process that loses the race must wait and then proceed,
        # not skip — it still needs the schema applied before it serves
        # traffic, and after the lock is released the version-tracked apply
        # loop is a clean no-op for it.
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATE_LOCK_KEY,))

        # The version ledger. Created if absent; `IF NOT EXISTS` so a process
        # that lost the advisory-lock race (ledger already created by the
        # winner) does not error.
        cur.execute("SELECT to_regclass('public.schema_migrations')")
        ledger_existed = cur.fetchone()[0] is not None
        cur.execute(
            """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
            """
        )

        # Bootstrap: a legacy DB has the full schema applied by the old code
        # but no ledger. If we JUST created the ledger AND a core table
        # already exists, the schema is already current — backfill every
        # version as applied so the loop below re-runs (and re-drops) nothing.
        if not ledger_existed:
            cur.execute("SELECT to_regclass('public.tenants')")
            legacy_schema_present = cur.fetchone()[0] is not None
            if legacy_schema_present:
                for version in _MIGRATION_VERSIONS:
                    cur.execute(
                        "INSERT INTO schema_migrations(version) VALUES (%s) "
                        "ON CONFLICT (version) DO NOTHING",
                        (version,),
                    )

        # Which versions are already applied (after any bootstrap backfill).
        cur.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in cur.fetchall()}

        # Apply each migration file exactly once, in order. Recording the
        # version in the SAME transaction means the apply + the ledger write
        # commit (or roll back) together — a crash mid-migration never leaves
        # a version marked applied that did not run, or vice versa.
        for version in _MIGRATION_VERSIONS:
            if version in applied:
                continue
            sql = (migrations_dir / f"{version}.sql").read_text(encoding="utf-8")
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations(version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING",
                (version,),
            )

        # OSS bootstrap: idempotently insert the `default` tenant.
        #
        # When `RB_REQUIRE_AUTH` is unset/false (the self-host headline path)
        # `current_tenant_id` short-circuits to the literal string "default"
        # for every request — every dataset / api_keys / shard row created in
        # OSS mode therefore carries `tenant_id = 'default'`. Those tables FK
        # to `tenants(id)` (see 002_datasets_tenant_isolation.sql) so the row
        # must exist before any request lands; the simplest place to guarantee
        # that is at migration time. ON CONFLICT keeps the bootstrap idempotent
        # across restarts and across the legacy-DB backfill path above.
        #
        # `password_hash='!disabled!'` is intentionally a non-bcrypt-hash
        # sentinel: bcrypt.checkpw() will never match against it, so even if a
        # self-hoster later flips `RB_REQUIRE_AUTH=true` no one can `/auth/login`
        # as this tenant by guessing a password. The email is a documented dud
        # (`self-host@localhost`) so it cannot collide with any real signup.
        cur.execute(
            """
INSERT INTO tenants (id, email, password_hash, plan)
VALUES ('default', 'self-host@localhost', '!disabled!', 'oss')
ON CONFLICT (id) DO NOTHING
            """
        )


# --- Hot-tier (delta tier) schema migration -------------------------------
#
# The hot tier is a SEPARATE data-plane pgvector instance (RB_HOT_DSN), kept off
# the control-plane Postgres so a tenant write-storm cannot starve the metadata
# reads on every query's critical path (docs/architecture/delta-tier.md, "Blast
# radius & control/data-plane isolation"). Its schema is migrated independently
# from the control-plane schema above, with the same advisory-lock + ledger
# discipline, against its own DSN. The whole hot path is DEFAULT-OFF: when
# RB_HOT_DSN is unset `migrate_hot()` is a no-op, so a flag-off deploy behaves
# byte-identically to today and nothing ever connects to a hot instance.

# Distinct advisory-lock key for the hot ledger. It lives in a different
# database from `_MIGRATE_LOCK_KEY`, so collision is impossible either way, but
# a separate constant keeps the intent clear. Fixed forever once deployed.
_HOT_MIGRATE_LOCK_KEY = 0x726F73685F686F74  # ASCII "rosh_hot", a stable constant

# Process-local serialisation for hot migrations, mirroring the control-plane
# `_MIGRATE_LOCK` / `_MIGRATED` pair (cross-process safety is the Postgres
# advisory lock inside `_apply_hot_migrations`).
_HOT_MIGRATE_LOCK = threading.Lock()
_HOT_MIGRATED = False


def _hot_dsn() -> Optional[str]:
    """Return the hot-tier (pgvector) DSN from `RB_HOT_DSN`, or None if unset.

    `None` means the delta tier is not configured — every hot path is a no-op
    and the deploy behaves exactly as it does today. A blank/whitespace value is
    treated as unset so an empty compose default cannot accidentally enable it.
    """
    raw = os.getenv("RB_HOT_DSN")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def migrate_hot(force: bool = False) -> bool:
    """Apply the hot-tier (pgvector) schema against `RB_HOT_DSN`.

    Returns True if migrations were applied (or would have been — i.e. a hot DSN
    is configured), False if the hot tier is OFF (no `RB_HOT_DSN`) and nothing
    was done. The boolean lets the migration entrypoint print an accurate
    "skipped (delta tier off)" vs "applied" line.

    DEFAULT-OFF: with no `RB_HOT_DSN` this is a pure no-op — it never opens a
    connection, never imports a pgvector dependency, and leaves behaviour
    identical to today. This is the property that keeps a flag-off
    `docker compose up` byte-identical.

    Memory mode (`DATABASE_URL=memory://...`) does NOT suppress this: the hot
    tier is a wholly separate instance addressed by its own DSN, so a test or a
    deploy can point `RB_HOT_DSN` at a real pgvector while the control plane runs
    in memory. The gate is `RB_HOT_DSN`, not `DATABASE_URL`.

    The `_HOT_MIGRATE_LOCK` / `_HOT_MIGRATED` pair serialises within a process;
    `_apply_hot_migrations()` takes a Postgres `pg_advisory_xact_lock` so several
    processes booting at once serialise in the hot database instead of racing the
    DDL locks — identical discipline to `migrate()`.

    `force=True` mirrors `migrate(force=True)`: it bypasses the `RB_SKIP_MIGRATE`
    early-return so the dedicated migration entrypoint always applies the schema
    even when it inherits that flag from the service env.
    """
    dsn = _hot_dsn()
    if dsn is None:
        # Delta tier off — nothing to migrate, nothing connects to a hot store.
        return False
    if not force and os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes"):
        # Schema applied out-of-band (same contract as the control-plane
        # migrate()); a hot DSN IS configured, so report True.
        return True
    global _HOT_MIGRATED
    with _HOT_MIGRATE_LOCK:
        if _HOT_MIGRATED:
            return True
        _apply_hot_migrations(dsn)
        _HOT_MIGRATED = True
    return True


def _apply_hot_migrations(dsn: str) -> None:
    """Apply the ordered hot migration files exactly once each (version-tracked).

    A faithful copy of `_apply_migrations()`'s cross-process-safe pattern, but
    against the hot DSN and the `hot_schema_migrations` ledger:

      - a dedicated (non-pooled) connection to `dsn` — the hot instance has no
        application connection pool;
      - `pg_advisory_xact_lock(_HOT_MIGRATE_LOCK_KEY)` first, so concurrent
        migrators serialise in the hot database rather than deadlocking on the
        `CREATE EXTENSION` / `CREATE TABLE` locks;
      - a `hot_schema_migrations(version, applied_at)` ledger so a re-run applies
        only un-applied versions. The hot migrations are all `IF NOT EXISTS`
        (additive, non-destructive), so even a direct re-execute is safe; the
        ledger keeps the common path a clean skip.

    There is no legacy-bootstrap branch (unlike the control plane): the hot
    instance is brand new with the delta tier, so there is never a pre-existing
    hot schema without a ledger to reconcile.
    """
    migrations_dir = Path(__file__).parent / "migrations" / "hot"
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s)", (_HOT_MIGRATE_LOCK_KEY,)
            )
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS hot_schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
                """
            )
            cur.execute("SELECT version FROM hot_schema_migrations")
            applied = {row[0] for row in cur.fetchall()}
            for version in _HOT_MIGRATION_VERSIONS:
                if version in applied:
                    continue
                sql = (migrations_dir / f"{version}.sql").read_text(
                    encoding="utf-8"
                )
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO hot_schema_migrations(version) VALUES (%s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (version,),
                )
    finally:
        conn.close()


# --- Hot-tier (delta tier) synchronous write path -------------------------
#
# The flag-gated, default-off write path that makes a `POST /vectors` durable
# and immediately queryable. It writes ONLY to the separate data-plane hot
# pgvector instance (`RB_HOT_DSN`) — never the control-plane Postgres — so a
# tenant write-storm cannot starve the catalog reads on every query's critical
# path (docs/architecture/delta-tier.md, "Blast radius"). Everything here is a
# no-op unless `delta_tier_enabled()` is True, which the service checks before
# calling in; so a flag-off deploy never opens a hot connection.
#
# Scope note: this is the WRITE path only — the query union, the hot→cold
# flush, and hot-delete tombstoning are later PRs.


def delta_tier_enabled() -> bool:
    """Whether the synchronous hot-tier write path is active.

    Mirrors `quotas_enabled()` (the OSS opt-in idiom): defaults OFF and reads
    the env fresh on every call so a test can flip it via monkeypatch without a
    module reload. It is the MASTER switch for delta-tier behaviour.

    Two conditions must BOTH hold for it to be on:
      - `RB_DELTA_TIER` is truthy (`1`/`true`/`yes`/`on`, case-insensitive), and
      - `RB_HOT_DSN` is configured (non-empty) — there is a hot store to write to.

    Requiring the DSN as well as the flag means a deploy that flips
    `RB_DELTA_TIER=true` but forgets to point `RB_HOT_DSN` at an instance stays
    on the byte-identical flag-off path rather than erroring on every write.
    """
    if os.getenv("RB_DELTA_TIER", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return _hot_dsn() is not None


def _hot_conn() -> "psycopg2.extensions.connection":
    """Return a fresh, DEDICATED psycopg2 connection to the hot store.

    A brand-new connection to `RB_HOT_DSN` per call (NOT pooled): the hot tier
    is a separate data-plane instance with no application pool of its own, and
    the per-write path is short — allocate an LSN, UPSERT, commit. The caller
    owns the connection and MUST close it (see `hot_upsert_vectors`, which uses
    `contextlib.closing`).

    Lazy by construction: this is only ever reached from `hot_upsert_vectors`,
    which the service calls only when `delta_tier_enabled()` is True. With the
    flag off no caller reaches here, so no connection to a hot instance is ever
    opened — the property that keeps a flag-off deploy byte-identical.

    Raises `RuntimeError` if `RB_HOT_DSN` is unset — that is a programming error
    (a caller reached the hot path with the tier off), not an expected runtime
    state.
    """
    dsn = _hot_dsn()
    if dsn is None:
        raise RuntimeError("hot tier is off (RB_HOT_DSN unset); no hot connection")
    return psycopg2.connect(dsn)


def _to_pgvector_literal(values: List[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. `[1.0,2.0,3.0]`.

    pgvector accepts its `vector` input as a bracketed, comma-separated string;
    psycopg2 binds it as the parameter for the unparameterised `embedding`
    column. The dimension is NOT enforced here — the service validates each
    record's length against `dataset.dimension` before this is called, exactly
    as the cold path does.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def hot_upsert_vectors(
    tenant_id: str,
    dataset: str,
    records: List[dict],
) -> int:
    """Synchronously UPSERT validated records into the hot tier; return the count.

    For each record (`{"id", "values", "metadata"}`, already validated by the
    service): allocate a strictly-monotonic per-`(tenant, dataset)` LSN from the
    hot store's `hot_lsn_seq` (the atomic upsert-increment), then UPSERT the row
    into `hot_vectors` with last-write-wins on `(tenant_id, dataset, id)` and
    `deleted = false`. A re-sent id overwrites the prior row (new embedding,
    metadata, and a fresh higher LSN) — never a duplicate.

    The LSN is generated in the HOT store, so this path never touches the
    control-plane Postgres (docs/architecture/delta-tier.md, "The watermark").
    All records for one call share a single hot connection + transaction: the
    whole batch commits atomically, so a mid-batch failure leaves the hot tier
    unchanged rather than half-applied.

    The LSN per record is allocated one statement at a time (not a single bulk
    statement) because the upsert-increment `RETURNING last_lsn` is what makes
    the value monotonic and visible; for the small, flush-bounded ingest batch
    the hot tier targets this is cheap. Returns the number of rows written.

    Only ever called when `delta_tier_enabled()` is True (the service gates it),
    so it never runs — and never opens a connection — with the flag off.
    """
    if not records:
        return 0
    written = 0
    # One dedicated hot connection for the whole batch. `contextlib.closing`
    # guarantees the socket is closed on every exit path; the `with conn` block
    # manages the transaction (commit on success, rollback on error) so the
    # batch is all-or-nothing.
    with contextlib.closing(_hot_conn()) as conn, conn, conn.cursor() as cur:
        for rec in records:
            # 1. Allocate the next LSN for this (tenant, dataset). The upsert-
            #    increment is serialised by Postgres on the single seq row, so
            #    `last_lsn` is strictly monotonic with no cross-dataset contention.
            cur.execute(
                """
INSERT INTO hot_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, 1)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = hot_lsn_seq.last_lsn + 1
RETURNING last_lsn
                """,
                (tenant_id, dataset),
            )
            lsn = cur.fetchone()[0]
            # 2. UPSERT the vector. Last-write-wins on (tenant, dataset, id):
            #    a re-sent id overwrites embedding/metadata/lsn and clears any
            #    prior tombstone (deleted -> false), so an upserted id is live.
            cur.execute(
                """
INSERT INTO hot_vectors (tenant_id, dataset, id, embedding, metadata, lsn, deleted)
VALUES (%s, %s, %s, %s, %s, %s, FALSE)
ON CONFLICT (tenant_id, dataset, id)
DO UPDATE SET
  embedding = EXCLUDED.embedding,
  metadata  = EXCLUDED.metadata,
  lsn       = EXCLUDED.lsn,
  deleted   = FALSE
                """,
                (
                    tenant_id,
                    dataset,
                    rec["id"],
                    _to_pgvector_literal(rec["values"]),
                    json.dumps(rec.get("metadata") or {}),
                    lsn,
                ),
            )
            written += 1
    return written


# --- Per-dataset build advisory lock (multi-worker safety) ----------------


# Namespace constant for the per-dataset builder advisory lock. The two-int
# form `pg_try_advisory_lock(classid, objid)` is used: `_BUILD_LOCK_CLASS`
# fixes the high 32 bits so a builder lock can never collide with any other
# advisory lock in the system, and the low 32 bits are a hash of
# `tenant + dataset` so distinct datasets get distinct locks.
_BUILD_LOCK_CLASS = 0x42554C44  # ASCII "BULD" — the builder-lock namespace


def _dataset_lock_objid(tenant: str, dataset: str) -> int:
    """Hash `(tenant, dataset)` to a signed 32-bit advisory-lock object id.

    A stable hash (not Python's salted `hash()`) so every process derives the
    same lock id for the same dataset. Masked to a signed 32-bit int — the
    range `pg_try_advisory_lock(int4, int4)` accepts.
    """
    digest = hashlib.sha1(f"{tenant}\x00{dataset}".encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "big")
    # Map an unsigned 32-bit value into the signed int4 range Postgres expects.
    if val >= 0x80000000:
        val -= 0x100000000
    return val


@contextlib.contextmanager
def dataset_build_lock(tenant: str, dataset: str) -> Iterator[bool]:
    """Try to acquire the per-dataset builder advisory lock; yield True/False.

    Multi-worker safety (Change 3): when `index_builder` is replicated, two
    builder replicas can pick up two `DATASET_READY` messages for the SAME
    dataset concurrently (or a redelivered message races the original). Both
    would read the same landing parts and fold them in — double-indexing.

    This serialises builds *per dataset* with a Postgres SESSION-level advisory
    lock (`pg_try_advisory_lock`). It is NON-blocking: a caller that loses the
    race yields `False` and the builder skips the build. Skipping is only safe
    if the skipped `DATASET_READY` message is RE-DELIVERED, not discarded — the
    skipped message may represent a NEWER upload than the in-progress build, so
    those parts must still be indexed eventually. The consume loop must
    therefore `nack(msg, requeue=True)` on a skip (NOT `ack`); the retry then
    either re-indexes any still-unindexed landing parts or is a clean no-op via
    the newest shard's `indexed_landing_uris` manifest. See `run_once` /
    `index_builder.main_loop`.

    Connection-scope dependency (Finding 3). The session-level advisory lock
    MUST be acquired and released on the same connection, which is why this
    uses a dedicated `_conn()`. Correctness depends on `_conn()` NOT being
    pooled — a pooled connection returned to the pool while still holding the
    session lock would leak the lock to the next borrower. See `_conn`.

    Concurrent builds of *different* datasets get distinct lock ids and still
    run fully in parallel.

    `memory://` / single-process test mode has no Postgres and no concurrency,
    so there is nothing to serialise — it always yields `True` (the build
    proceeds) and is a pure no-op.

    Usage:
        with dataset_build_lock(tenant, dataset) as acquired:
            if not acquired:
                return  # another builder owns this dataset; skip
            ...  # do the build
    """
    if _MEMORY_MODE:
        yield True
        return
    objid = _dataset_lock_objid(tenant, dataset)
    conn = _conn()
    try:
        # autocommit so the lock is held at the SESSION level for the whole
        # `with` body, independent of any transaction the build itself runs.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (_BUILD_LOCK_CLASS, objid),
            )
            acquired = bool(cur.fetchone()[0])
        try:
            yield acquired
        finally:
            if acquired:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(%s, %s)",
                        (_BUILD_LOCK_CLASS, objid),
                    )
    finally:
        # Closing the connection also releases any still-held session lock —
        # a safety net if the explicit unlock above could not run.
        conn.close()


# --- Tenants --------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 with trailing Z (per v1 contract)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_DEFAULT_VECTOR_QUOTA = 100000
_DEFAULT_DAILY_QUERY_QUOTA = 10000


def _quota_defaults() -> Tuple[int, int]:
    """Return `(vector_quota, daily_query_quota)` to stamp on a new tenant.

    Test hook: `RB_TEST_VECTOR_QUOTA` / `RB_TEST_QUERY_QUOTA` env vars, when
    set, override the default per-tenant limits at tenant-creation time. They are
    read fresh on every signup so a test (or the E2E harness) can flip them
    before calling `POST /auth/signup` to get a low-quota tenant and trigger a
    real 429 without issuing thousands of requests. NOT for production use —
    leaving them unset yields the contract defaults (100000 / 10000).
    """
    vq = os.getenv("RB_TEST_VECTOR_QUOTA")
    qq = os.getenv("RB_TEST_QUERY_QUOTA")
    vector_quota = int(vq) if vq and vq.isdigit() else _DEFAULT_VECTOR_QUOTA
    query_quota = int(qq) if qq and qq.isdigit() else _DEFAULT_DAILY_QUERY_QUOTA
    return vector_quota, query_quota


def create_tenant(tenant_id: str, email: str, password_hash: str) -> dict:
    """Insert a new tenant row.

    Returns the persisted row as a dict. Raises ValueError("duplicate_email")
    if the email already exists. Defaults (plan='free', quotas, etc.) are
    populated server-side to match the v1 API contract. Quota values honour
    the `RB_TEST_*` overrides (see `_quota_defaults`).
    """
    vector_quota, query_quota = _quota_defaults()
    if _MEMORY_MODE:
        if email in _MEM_TENANTS_BY_EMAIL:
            raise ValueError("duplicate_email")
        row = {
            "id": tenant_id,
            "email": email,
            "password_hash": password_hash,
            "plan": "free",
            "vector_quota": vector_quota,
            "daily_query_quota": query_quota,
            "vectors_used": 0,
            "queries_today": 0,
            "queries_reset_at": _dt.date.today().isoformat(),
            "created_at": _now_iso(),
        }
        _MEM_TENANTS[tenant_id] = row
        _MEM_TENANTS_BY_EMAIL[email] = tenant_id
        return dict(row)

    try:
        with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
INSERT INTO tenants(id, email, password_hash, vector_quota, daily_query_quota)
VALUES (%s, %s, %s, %s, %s)
RETURNING *
                """,
                (tenant_id, email, password_hash, vector_quota, query_quota),
            )
            row = cur.fetchone()
            return dict(row)
    except psycopg2.errors.UniqueViolation as exc:
        raise ValueError("duplicate_email") from exc


# --- Quotas ---------------------------------------------------------------


def _usage_from_row(row: dict) -> dict:
    """Project a tenant row down to the v1 `usage` shape.

    `queries_reset_at` is normalised to a `YYYY-MM-DD` string regardless of
    whether it came from Postgres (a `date`) or the memory adapter (a string).
    """
    reset = row.get("queries_reset_at")
    if isinstance(reset, (_dt.date, _dt.datetime)):
        reset = reset.isoformat()[:10]
    elif isinstance(reset, str):
        reset = reset[:10]
    return {
        "vectors_used": int(row.get("vectors_used", 0)),
        "vector_quota": int(row.get("vector_quota", _DEFAULT_VECTOR_QUOTA)),
        "queries_today": int(row.get("queries_today", 0)),
        "daily_query_quota": int(row.get("daily_query_quota", _DEFAULT_DAILY_QUERY_QUOTA)),
        "queries_reset_at": reset,
    }


def reset_daily_if_needed(tenant_id: str) -> None:
    """Lazy daily reset: if `queries_reset_at` is older than today, zero
    `queries_today` and bump the date to today.

    Standalone helper; `get_usage` / `try_consume_query` perform the same
    reset inline so the reset and the consume are atomic. Memory mode takes
    the quota lock; Postgres mode does it in one UPDATE.
    """
    today = _dt.date.today()
    if _MEMORY_MODE:
        with _MEM_QUOTA_LOCK:
            row = _MEM_TENANTS.get(tenant_id)
            if row is None:
                return
            _mem_reset_locked(row, today)
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
UPDATE tenants
SET queries_today = 0, queries_reset_at = CURRENT_DATE
WHERE id = %s AND queries_reset_at < CURRENT_DATE
            """,
            (tenant_id,),
        )


def _mem_reset_locked(row: dict, today: _dt.date) -> None:
    """Apply the lazy daily reset to an in-memory row. Caller holds the lock."""
    reset = row.get("queries_reset_at")
    if isinstance(reset, str):
        try:
            reset_date = _dt.date.fromisoformat(reset[:10])
        except ValueError:
            reset_date = today
    elif isinstance(reset, (_dt.date, _dt.datetime)):
        reset_date = reset if isinstance(reset, _dt.date) else reset.date()
    else:
        reset_date = today
    if reset_date < today:
        row["queries_today"] = 0
        row["queries_reset_at"] = today.isoformat()


def get_usage(tenant_id: str) -> dict:
    """Return the current usage/quota snapshot for `tenant_id`.

    Performs a lazy daily reset first so a stale `queries_today` from a prior
    day is never reported. Returns the v1 `usage` shape; raises
    ValueError("tenant_not_found") if the tenant does not exist.
    """
    today = _dt.date.today()
    if _MEMORY_MODE:
        with _MEM_QUOTA_LOCK:
            row = _MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            _mem_reset_locked(row, today)
            return _usage_from_row(row)
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
UPDATE tenants
SET queries_today = CASE WHEN queries_reset_at < CURRENT_DATE THEN 0 ELSE queries_today END,
    queries_reset_at = CASE WHEN queries_reset_at < CURRENT_DATE THEN CURRENT_DATE ELSE queries_reset_at END
WHERE id = %s
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
            """,
            (tenant_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("tenant_not_found")
        return _usage_from_row(dict(row))


def try_consume_query(tenant_id: str) -> Tuple[bool, dict]:
    """Atomically consume one unit of daily query quota.

    Lazy-resets first, then if `queries_today < daily_query_quota` increments
    `queries_today` and returns `(True, usage)`; otherwise `(False, usage)`
    without incrementing. The post-state `usage` is returned in both cases so
    the caller can surface `details.limit` / `details.reset_at` on a 429.

    Memory mode: the reset + check + increment run under one lock. Postgres
    mode: a single conditional UPDATE — the DB serialises concurrent callers,
    so two requests can never both slip past the cap.

    This is a **hot-row write** that MUST commit and release its row lock
    immediately. It uses `pooled_conn(standalone=True)` so it always runs in
    its own short transaction, never riding the request-scoped transaction —
    under the request scope the row lock would be held for the whole request
    (including the CP→DP proxy hop), serialising concurrent queries from one
    tenant.
    """
    today = _dt.date.today()
    if _MEMORY_MODE:
        with _MEM_QUOTA_LOCK:
            row = _MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            _mem_reset_locked(row, today)
            quota = int(row.get("daily_query_quota", _DEFAULT_DAILY_QUERY_QUOTA))
            used = int(row.get("queries_today", 0))
            if used < quota:
                row["queries_today"] = used + 1
                return True, _usage_from_row(row)
            return False, _usage_from_row(row)
    with pooled_conn(standalone=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        # First: apply the lazy reset so the conditional increment below sees
        # a fresh counter at the day boundary. Both run in one transaction.
        cur.execute(
            """
UPDATE tenants
SET queries_today = 0, queries_reset_at = CURRENT_DATE
WHERE id = %s AND queries_reset_at < CURRENT_DATE
            """,
            (tenant_id,),
        )
        cur.execute(
            """
UPDATE tenants
SET queries_today = queries_today + 1
WHERE id = %s AND queries_today < daily_query_quota
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
            """,
            (tenant_id,),
        )
        row = cur.fetchone()
        if row is not None:
            return True, _usage_from_row(dict(row))
        # Either the tenant is missing or the cap is hit — read back to tell.
        cur.execute(
            """
SELECT vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
FROM tenants WHERE id = %s
            """,
            (tenant_id,),
        )
        snap = cur.fetchone()
        if snap is None:
            raise ValueError("tenant_not_found")
        return False, _usage_from_row(dict(snap))


def try_consume_vectors(tenant_id: str, count: int) -> Tuple[bool, dict]:
    """Atomically consume `count` units of the lifetime vector quota.

    If `vectors_used + count <= vector_quota` adds `count` to `vectors_used`
    and returns `(True, usage)`; otherwise `(False, usage)` unchanged. A
    `count` of 0 always succeeds without touching the row.

    Same atomicity discipline as `try_consume_query`: memory mode under the
    quota lock, Postgres mode via one conditional UPDATE ... RETURNING.

    Like `try_consume_query` this is a hot-row write and uses
    `pooled_conn(standalone=True)` so the `UPDATE` commits and releases its
    row lock in its own short transaction instead of being held for the whole
    request-scoped transaction.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if _MEMORY_MODE:
        with _MEM_QUOTA_LOCK:
            row = _MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            quota = int(row.get("vector_quota", _DEFAULT_VECTOR_QUOTA))
            used = int(row.get("vectors_used", 0))
            if used + count <= quota:
                row["vectors_used"] = used + count
                return True, _usage_from_row(row)
            return False, _usage_from_row(row)
    with pooled_conn(standalone=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
UPDATE tenants
SET vectors_used = vectors_used + %s
WHERE id = %s AND vectors_used + %s <= vector_quota
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
            """,
            (count, tenant_id, count),
        )
        row = cur.fetchone()
        if row is not None:
            return True, _usage_from_row(dict(row))
        cur.execute(
            """
SELECT vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
FROM tenants WHERE id = %s
            """,
            (tenant_id,),
        )
        snap = cur.fetchone()
        if snap is None:
            raise ValueError("tenant_not_found")
        return False, _usage_from_row(dict(snap))


def get_tenant_by_email(email: str) -> Optional[dict]:
    """Return the tenant row for `email`, or None."""
    if _MEMORY_MODE:
        tenant_id = _MEM_TENANTS_BY_EMAIL.get(email)
        if tenant_id is None:
            return None
        return dict(_MEM_TENANTS[tenant_id])
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM tenants WHERE email=%s", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_tenant_by_id(tenant_id: str) -> Optional[dict]:
    """Return the tenant row for `tenant_id`, or None."""
    if _MEMORY_MODE:
        row = _MEM_TENANTS.get(tenant_id)
        return dict(row) if row else None
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM tenants WHERE id=%s", (tenant_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# Default DP pool — the value migration 006 stamps on every tenant. The CP
# query proxy falls back to this for an unknown tenant or a NULL column so a
# routing lookup can never fail open into an unroutable pool.
_DEFAULT_DP_POOL = "shared"


def get_tenant_dp_pool(tenant_id: str) -> str:
    """Return the Query-DP pool name a tenant's `/v1/query` traffic routes to.

    The CP reverse proxy calls this per request to resolve `tenant_id -> DP
    pool` (then `resolve_dp_base_url` maps the pool to a base URL). The value
    comes from the `tenants.dp_pool` column (migration 006).

    Defaults to `'shared'` for an unknown tenant or a NULL column — a missing
    routing target must never fail open, so an unrecognised tenant transparently
    uses the shared pool exactly as a freshly-created one does.

    Memory mode: the in-memory tenant rows created by `create_tenant` predate
    this column, so a row with no `dp_pool` key reads back as `'shared'`. A test
    (or a future provisioning path) can set `_MEM_TENANTS[tid]["dp_pool"]` to
    simulate a dedicated-pool tenant and this returns that value.
    """
    if _MEMORY_MODE:
        row = _MEM_TENANTS.get(tenant_id)
        if row is None:
            return _DEFAULT_DP_POOL
        return row.get("dp_pool") or _DEFAULT_DP_POOL
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT dp_pool FROM tenants WHERE id=%s", (tenant_id,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            return _DEFAULT_DP_POOL
        return row[0]


# --- API keys -------------------------------------------------------------


def create_api_key(key_id: str, tenant_id: str, key_hash: str, name: str) -> dict:
    """Insert a new api_keys row and return the persisted dict.

    `key_hash` is the SHA-256 hex digest of the raw `rb_live_...` token;
    the raw value is never stored. SHA-256 is deterministic so the hash
    can be looked up directly via the `api_keys_hash_idx` index (see
    `get_api_key_by_hash`) — bcrypt's per-row salt would have made that
    impossible. `last_used_at` and `revoked_at` start as NULL.
    """
    if _MEMORY_MODE:
        row = {
            "id": key_id,
            "tenant_id": tenant_id,
            "key_hash": key_hash,
            "name": name,
            "created_at": _now_iso(),
            "last_used_at": None,
            "revoked_at": None,
        }
        _MEM_API_KEYS.append(row)
        _MEM_API_KEYS_BY_HASH[key_hash] = row
        return dict(row)
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
INSERT INTO api_keys(id, tenant_id, key_hash, name)
VALUES (%s, %s, %s, %s)
RETURNING *
            """,
            (key_id, tenant_id, key_hash, name),
        )
        return dict(cur.fetchone())


def list_api_keys(tenant_id: str) -> List[dict]:
    """Return all api_keys rows for `tenant_id`, oldest-first.

    Revoked keys are included; the row carries `revoked_at`. Callers
    project to the v1 response shape themselves (no `key_hash` leaks).
    """
    if _MEMORY_MODE:
        rows = [r for r in _MEM_API_KEYS if r["tenant_id"] == tenant_id]
        return [dict(r) for r in rows]
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM api_keys WHERE tenant_id=%s ORDER BY created_at ASC",
            (tenant_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_api_key(key_id: str, tenant_id: str) -> Optional[dict]:
    """Return the api_keys row for `(key_id, tenant_id)`, or None.

    Filtering on `tenant_id` here keeps the cross-tenant 404 contract
    enforced in the data layer rather than relying on the caller.
    """
    if _MEMORY_MODE:
        for r in _MEM_API_KEYS:
            if r["id"] == key_id and r["tenant_id"] == tenant_id:
                return dict(r)
        return None
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM api_keys WHERE id=%s AND tenant_id=%s",
            (key_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def revoke_api_key(key_id: str, tenant_id: str) -> bool:
    """Mark the key revoked. Returns True iff a row was modified.

    Idempotent: revoking an already-revoked key leaves the original
    `revoked_at` in place and returns False, so callers can distinguish
    "no such key" from "already revoked" if they need to (the API maps
    both to 404 per the contract, but the primitive stays honest).
    """
    if _MEMORY_MODE:
        for r in _MEM_API_KEYS:
            if r["id"] == key_id and r["tenant_id"] == tenant_id:
                if r["revoked_at"] is not None:
                    return False
                r["revoked_at"] = _now_iso()
                return True
        return False
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
UPDATE api_keys
SET revoked_at = now()
WHERE id=%s AND tenant_id=%s AND revoked_at IS NULL
            """,
            (key_id, tenant_id),
        )
        return cur.rowcount > 0


def get_api_key_by_hash(key_hash: str) -> Optional[dict]:
    """Return the api_keys row whose `key_hash` equals `key_hash`, or None.

    This is the auth-time resolution primitive: an inbound `rb_live_...`
    token is reduced to its SHA-256 hex digest by the caller, then looked
    up here in O(1) — a dict lookup in memory mode, an indexed
    `WHERE key_hash = %s` (backed by `api_keys_hash_idx`, also UNIQUE) in
    Postgres mode. The cost is independent of the total number of keys in
    the system. Revocation is NOT filtered here; the caller inspects
    `revoked_at` on the returned row so it can reject a revoked key.
    """
    if _MEMORY_MODE:
        row = _MEM_API_KEYS_BY_HASH.get(key_hash)
        return dict(row) if row else None
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM api_keys WHERE key_hash=%s", (key_hash,))
        row = cur.fetchone()
        return dict(row) if row else None


def touch_api_key_last_used(key_id: str) -> None:
    """Set `last_used_at = now()` on a successful auth.

    Fire-and-forget: we do not return success/failure. If the row is
    gone (deleted tenant cascade) the update is a no-op.

    Uses `pooled_conn(standalone=True)` so this `UPDATE api_keys` commits and
    releases the api_keys row lock immediately, in its own short transaction.
    It runs during auth at the START of every authenticated request; left on
    the request-scoped connection its row lock would be held for the WHOLE
    request (including the CP->DP proxy hop), so concurrent requests sharing
    one API key would serialize on it — the same hot-row contention that
    `try_consume_query` addresses via `standalone=True`.
    """
    if _MEMORY_MODE:
        for r in _MEM_API_KEYS:
            if r["id"] == key_id:
                r["last_used_at"] = _now_iso()
                return
        return
    with pooled_conn(standalone=True) as conn, conn.cursor() as cur:
        cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id=%s", (key_id,))


# --- Datasets -------------------------------------------------------------


def create_dataset(tenant_id: str, dataset_name: str, dimension: int) -> dict:
    """Insert a brand-new dataset row owned by `tenant_id`.

    Returns the persisted row dict. Raises ValueError("dataset_exists") if
    a non-deleted row with the same `(tenant_id, dataset_name)` already
    exists. Soft-deleted rows (`deleted_at IS NOT NULL`) are treated as
    absent: re-creating a previously-deleted dataset is permitted and
    resurrects the slot with status=`empty`.
    """
    now = _now_iso()
    if _MEMORY_MODE:
        key = (tenant_id, dataset_name)
        existing = _MEM_DATASETS.get(key)
        if existing is not None and not existing.get("deleted_at"):
            raise ValueError("dataset_exists")
        row = {
            "tenant_id": tenant_id,
            "dataset_name": dataset_name,
            "dimension": dimension,
            "row_count": 0,
            "status": "empty",
            "error_message": None,
            "source_uris": [],
            "landing_format": "jsonl",
            "object_store_uri": None,
            "last_indexed_at": None,
            "status_updated_at": now,
            "deleted_at": None,
            "created_at": now,
        }
        _MEM_DATASETS[key] = row
        return dict(row)
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check existing non-deleted row first to give a clean error code.
        cur.execute(
            """
SELECT 1 FROM dataset_catalog
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
            """,
            (tenant_id, dataset_name),
        )
        if cur.fetchone() is not None:
            raise ValueError("dataset_exists")
        # Upsert: a soft-deleted row with the same PK gets resurrected so we
        # do not violate the (tenant_id, dataset_name) primary key.
        cur.execute(
            """
INSERT INTO dataset_catalog(tenant_id, dataset_name, dimension)
VALUES (%s, %s, %s)
ON CONFLICT (tenant_id, dataset_name) DO UPDATE
SET dimension=EXCLUDED.dimension,
    row_count=0,
    status='empty',
    error_message=NULL,
    source_uris='{}',
    last_indexed_at=NULL,
    deleted_at=NULL,
    created_at=now()
RETURNING *
            """,
            (tenant_id, dataset_name, dimension),
        )
        return dict(cur.fetchone())


def get_dataset(tenant_id: str, dataset_name: str) -> Optional[dict]:
    """Return the dataset row for `(tenant_id, dataset_name)`, excluding
    soft-deleted rows. Returns None if missing OR not owned by caller —
    the v1 contract maps both to 404 `dataset_not_found`.
    """
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return None
        return dict(row)
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
SELECT * FROM dataset_catalog
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
            """,
            (tenant_id, dataset_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_datasets(tenant_id: str) -> List[dict]:
    """Return non-deleted datasets owned by `tenant_id`, ordered by name."""
    if _MEMORY_MODE:
        out = [
            dict(row)
            for (tid, _), row in _MEM_DATASETS.items()
            if tid == tenant_id and not row.get("deleted_at")
        ]
        out.sort(key=lambda r: r["dataset_name"])
        return out
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
SELECT * FROM dataset_catalog
WHERE tenant_id=%s AND deleted_at IS NULL
ORDER BY dataset_name
            """,
            (tenant_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_dataset(tenant_id: str, dataset_name: str) -> bool:
    """Soft-delete a dataset AND purge its `shard_catalog` rows.

    Returns True iff a non-deleted dataset row was found and flipped.

    Why the shard purge:

      `dataset_catalog` is soft-deleted (`deleted_at=now()`) so audit /
      tenancy lookups can still resolve the row, but `shard_catalog` is
      HARD-deleted for the `(tenant_id, dataset_name)` pair. Without the
      shard purge a same-name re-create resurrects the dataset row but
      `list_shards` still returns the old shards — the query path resolves
      `latest = shards[0]`, hits whatever is still in the FAISS shard
      cache (or refaults the bytes from object storage), and serves the
      deleted dataset's vectors under the new dataset's name. The stress
      driver's I-01 scenario was exactly this: create -> ingest 5 -> delete
      -> create -> query returns 5 ghost rows with `mode:"hot"`.

      Object-storage bytes are left in place (no readers can reach them
      without a catalog row); the documented background sweeper claim shifts
      from "shards too" to "S3 objects only". The in-process FAISS shard
      cache and SSD shard tier are keyed by `shard_id` / `shard_uri`
      respectively and become unreachable on the next `list_shards`
      returning [] — LRU eviction reclaims them naturally; no special
      teardown is required for correctness.

    Why notify on delete:

      The per-`(tenant, dataset)` `_CATALOG_CACHE` on the DP caches the
      `list_shards` result for `RB_CATALOG_FRESHNESS_S` (default 5 s). A
      stale cached entry could serve the OLD shard list for up to that TTL
      after delete, which means a fast delete -> create -> query sequence
      can still hit ghosts within the 5 s window. We fire the same
      `pg_notify('catalog_updates', ...)` payload `add_shard` uses so the
      DP's existing `_on_catalog_notify` evicts the cached entry
      synchronously — no new wiring on the DP side. The notify is
      best-effort (with `RB_CATALOG_LISTEN=false` the TTL pull is the
      only invalidation), but the shard-row purge above is the
      correctness mechanism: even with the cache stale, a stale list of
      shards points at rows that no longer exist, so the query path
      resolves `shards = []` after the cache expires.

    The shard purge runs BEFORE the dataset UPDATE so the FK
    `shard_catalog -> dataset_catalog` constraint stays valid throughout
    (a child without a parent is impossible for a row instant). Both
    statements ride the same transaction.
    """
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return False
        row["deleted_at"] = _now_iso()
        # Hard-delete the matching shard rows so list_shards returns []
        # for this (tenant, dataset) immediately.
        _MEM_SHARDS[:] = [
            r for r in _MEM_SHARDS
            if not (
                r["tenant_id"] == tenant_id
                and r["dataset_name"] == dataset_name
            )
        ]
        # Fire the memory-backend NOTIFY so the DP's catalog cache (and
        # any other in-process subscriber) sees the invalidation.
        _fire_catalog_notify_memory({
            "tenant": tenant_id,
            "dataset": dataset_name,
            "shard_uri": "",
        })
        return True
    with pooled_conn() as conn, conn.cursor() as cur:
        # Purge first to keep the FK valid throughout the transaction.
        cur.execute(
            """
DELETE FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s
            """,
            (tenant_id, dataset_name),
        )
        cur.execute(
            """
UPDATE dataset_catalog
SET deleted_at = now()
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
            """,
            (tenant_id, dataset_name),
        )
        modified = cur.rowcount > 0
        if modified:
            # Best-effort NOTIFY — payload schema matches `add_shard` so
            # the DP's existing `_on_catalog_notify` invalidates the
            # `(tenant, dataset)` cache entry without any new wiring. The
            # shard purge above is the correctness mechanism; this is the
            # latency optimisation that closes the `RB_CATALOG_FRESHNESS_S`
            # stale-cache window.
            try:
                cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    (
                        "catalog_updates",
                        json.dumps({
                            "tenant": tenant_id,
                            "dataset": dataset_name,
                            "shard_uri": "",
                        }),
                    ),
                )
            except Exception:  # noqa: BLE001 - best-effort emission
                logging.getLogger(__name__).warning(
                    "pg_notify(catalog_updates) on delete_dataset failed; "
                    "TTL safety net will cover",
                    exc_info=True,
                )
        return modified


def update_dataset_status(
    tenant_id: str,
    dataset_name: str,
    status: str,
    error_message: Optional[str] = None,
    last_indexed_at: Optional[str] = None,
) -> None:
    """Set the dataset's status and optional `error_message`/`last_indexed_at`.

    Used by the validator (`validating` → `indexing`/`error`) and the index
    builder (`indexed` on success, `error` on failure). Passing a non-None
    `error_message` overwrites any previous error. Passing None does NOT leave
    `error_message` untouched: on any non-`error` transition the column is
    actively cleared to NULL (the SQL `CASE` and the memory path both do
    this), so a stale failure message never lingers on a dataset that has
    since moved on to a healthy status. A None `error_message` *into* an
    `error` status leaves the existing message in place.

    Every call stamps `status_updated_at = now()` so the reconciliation reaper
    (`adapters/queue/reaper.py`) can age out a dataset stranded in a
    non-terminal status by a hung/dead worker.
    """
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None:
            return
        row["status"] = status
        row["status_updated_at"] = _now_iso()
        if error_message is not None:
            row["error_message"] = error_message
        elif status != "error":
            # Clear stale error on a non-error transition.
            row["error_message"] = None
        if last_indexed_at is not None:
            row["last_indexed_at"] = last_indexed_at
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        if error_message is not None:
            cur.execute(
                """
UPDATE dataset_catalog
SET status=%s, error_message=%s, status_updated_at=now()
    {set_indexed}
WHERE tenant_id=%s AND dataset_name=%s
                """.format(set_indexed=", last_indexed_at=now()" if last_indexed_at else ""),
                (status, error_message, tenant_id, dataset_name),
            )
        else:
            cur.execute(
                """
UPDATE dataset_catalog
SET status=%s, status_updated_at=now(),
    error_message = CASE WHEN %s = 'error' THEN error_message ELSE NULL END
    {set_indexed}
WHERE tenant_id=%s AND dataset_name=%s
                """.format(set_indexed=", last_indexed_at=now()" if last_indexed_at else ""),
                (status, status, tenant_id, dataset_name),
            )


_NON_TERMINAL_STATUSES = ("validating", "indexing")


def find_stale_datasets(
    older_than_seconds: float,
    statuses: Tuple[str, ...] = _NON_TERMINAL_STATUSES,
) -> List[dict]:
    """Return non-deleted datasets stuck in a non-terminal `status` too long.

    "Too long" means `status_updated_at` is older than `older_than_seconds`
    ago. This is the reconciliation reaper's backstop for a worker that hangs
    (or dies mid-job in a way that escaped queue redelivery): the reaper flips
    each returned dataset to `error` so a customer's `GET /v1/datasets/{name}`
    can never report a silently-stuck `validating`/`indexing` forever.

    Each returned dict carries at least `tenant_id`, `dataset_name`, `status`.
    """
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=older_than_seconds
    )
    if _MEMORY_MODE:
        out: List[dict] = []
        for row in _MEM_DATASETS.values():
            if row.get("deleted_at") or row.get("status") not in statuses:
                continue
            updated = _parse_iso(row.get("status_updated_at"))
            if updated is None or updated <= cutoff:
                out.append(dict(row))
        return out
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
SELECT * FROM dataset_catalog
WHERE deleted_at IS NULL
  AND status = ANY(%s)
  AND status_updated_at <= %s
            """,
            (list(statuses), cutoff),
        )
        return [dict(r) for r in cur.fetchall()]


def fail_dataset_if_stale(
    tenant_id: str,
    dataset_name: str,
    older_than_seconds: float,
    error_message: str,
    statuses: Tuple[str, ...] = _NON_TERMINAL_STATUSES,
) -> bool:
    """Conditionally flip a stuck dataset to `error` — compare-and-set.

    The reconciliation reaper observes a dataset as stale, then flips it. In
    the gap between those two steps a worker may legitimately finish the job
    and write a terminal status (`indexed`). An unconditional
    `update_dataset_status(..., "error")` would then clobber that good result.

    This is the guarded flip: it sets `status='error'` ONLY IF the dataset is
    STILL in a non-terminal status AND still stale. In Postgres the `WHERE`
    clause is the compare-and-set — the DB serialises it against the worker's
    own `UPDATE`, so whichever commits last is the only writer that matters
    and a terminal status is never overwritten. The memory path re-checks the
    status and the timestamp under no lock but in a single function (the
    `_MEM_QUOTA_LOCK` does not cover datasets); it is best-effort guarded —
    good enough for the test-only in-process mode.

    Returns True iff the dataset was actually flipped to `error`.
    """
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=older_than_seconds
    )
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return False
        if row.get("status") not in statuses:
            # A worker already moved it to a terminal status — do not clobber.
            return False
        updated = _parse_iso(row.get("status_updated_at"))
        if updated is not None and updated > cutoff:
            return False
        row["status"] = "error"
        row["status_updated_at"] = _now_iso()
        row["error_message"] = error_message
        return True
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
UPDATE dataset_catalog
SET status='error', error_message=%s, status_updated_at=now()
WHERE tenant_id=%s AND dataset_name=%s
  AND deleted_at IS NULL
  AND status = ANY(%s)
  AND status_updated_at <= %s
            """,
            (error_message, tenant_id, dataset_name, list(statuses), cutoff),
        )
        return cur.rowcount > 0


def _parse_iso(value) -> Optional[_dt.datetime]:
    """Parse an ISO-8601 string (or pass through a datetime) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, str):
        try:
            dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=_dt.timezone.utc)
    return None


def upsert_dataset(
    tenant_id: str,
    dataset_name: str,
    dimension: int,
    source_uri: str,
    landing_format: str,
) -> None:
    """Insert or update a dataset catalog row (validator-side helper).

    Adds the source URI to `source_uris` and updates `landing_format`. Used
    by the validator worker which may receive records for a dataset that
    was registered out-of-band (legacy `/register-source`) and not yet in
    the catalog.
    """
    if _MEMORY_MODE:
        key = (tenant_id, dataset_name)
        row = _MEM_DATASETS.get(key)
        if row is None:
            row = {
                "tenant_id": tenant_id,
                "dataset_name": dataset_name,
                "dimension": dimension,
                "row_count": 0,
                "status": "empty",
                "error_message": None,
                "source_uris": [],
                "landing_format": landing_format,
                "object_store_uri": None,
                "last_indexed_at": None,
                "status_updated_at": _now_iso(),
                "deleted_at": None,
                "created_at": _now_iso(),
            }
            _MEM_DATASETS[key] = row
        row["landing_format"] = landing_format
        row.setdefault("source_uris", []).append(source_uri)
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
INSERT INTO dataset_catalog(tenant_id, dataset_name, dimension, source_uris, landing_format)
VALUES (%s, %s, %s, ARRAY[%s]::text[], %s)
ON CONFLICT (tenant_id, dataset_name) DO UPDATE
SET source_uris = array_append(dataset_catalog.source_uris, EXCLUDED.source_uris[1]),
    landing_format = EXCLUDED.landing_format
            """,
            (tenant_id, dataset_name, dimension, source_uri, landing_format),
        )


def increment_row_count(tenant_id: str, dataset_name: str, count: int) -> None:
    """Increment accumulated ingested row count for a dataset.

    Called by the validator after a successful landing write so the dataset's
    `row_count` reflects newly-accepted rows even before the builder commits.
    This is a transient over-count when the batch upserts existing ids:
    `set_row_count` (called by the builder after `add_shard`) reconciles
    `row_count` to the true count of unique live ids — the newest shard's
    `vector_count` — at build commit. Without that reconcile, a re-ingest of
    the same id would double `row_count` on every retry.
    """
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row:
            row["row_count"] = row.get("row_count", 0) + count
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
UPDATE dataset_catalog
SET row_count = row_count + %s
WHERE tenant_id=%s AND dataset_name=%s
            """,
            (count, tenant_id, dataset_name),
        )


def set_row_count(tenant_id: str, dataset_name: str, count: int) -> None:
    """Set the dataset's `row_count` to an exact value (build-time reconcile).

    The index builder calls this AFTER `add_shard` with the just-built
    shard's `vector_count` — equal to `index.ntotal` after the upsert's
    `remove_ids` + `add_with_ids`, which is the authoritative count of
    unique live ids in the dataset (one shard per dataset is the steady-
    state invariant: the sweep retains the newest shard, the second-newest
    is the in-flight-query grace buffer, older are purged).

    This reconciles the validator's `increment_row_count` over-count when
    a batch upserts existing ids — the validator increments by `len(good)`
    regardless of whether those ids already exist, so a re-ingest of the
    same id would otherwise double `row_count`. Setting (not incrementing)
    makes the reconcile idempotent and self-healing for any pre-existing drift.

    Floored at 0 — a negative `count` is treated as 0 rather than letting a
    bug elsewhere store a nonsense value.
    """
    value = max(0, int(count))
    if _MEMORY_MODE:
        row = _MEM_DATASETS.get((tenant_id, dataset_name))
        if row:
            row["row_count"] = value
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
UPDATE dataset_catalog
SET row_count = %s
WHERE tenant_id=%s AND dataset_name=%s
            """,
            (value, tenant_id, dataset_name),
        )


def add_shard(
    tenant_id: str,
    dataset_name: str,
    shard_uri: str,
    checksum: str,
    vector_count: int,
    index_type: str,
    build_type: str = "full",
    indexed_landing_uris: Optional[List[str]] = None,
) -> int:
    """Insert a new shard record and return its ID.

    Two columns support incremental indexing:
      - `build_type`: `'full'` (trained-from-scratch), `'incremental'`
        (existing index loaded, only new vectors `index.add()`-ed), or
        `'delete'` (existing index loaded, one vector removed by id — a
        delete-driven rebuild, labelled distinctly so deletes are not
        miscounted as ingests in `build_type`-keyed metrics).
      - `indexed_landing_uris`: the manifest of landing parquet part URIs
        already folded into this shard. The index builder reads the *newest*
        shard's manifest to decide which landing parts are new, so a
        subsequent ingest never re-reads previously indexed uploads.
    """
    uris = list(indexed_landing_uris or [])
    # The payload format is shared across the memory hook and the `pg_notify`
    # channel so the DP's catalog-cache invalidator can use one parser. Keep
    # keys minimal — `pg_notify`'s payload is capped at 8000 bytes by
    # Postgres, and a DP only needs `(tenant, dataset)` to route the
    # eviction. `shard_uri` is included for diagnostics (operator can
    # `LISTEN catalog_updates` from psql and see which shard fired).
    notify_payload = {
        "tenant": tenant_id,
        "dataset": dataset_name,
        "shard_uri": shard_uri,
    }
    if _MEMORY_MODE:
        global _MEM_SHARD_ID
        _MEM_SHARD_ID += 1
        record = {
            "id": _MEM_SHARD_ID,
            "tenant_id": tenant_id,
            "dataset_name": dataset_name,
            "shard_uri": shard_uri,
            "checksum": checksum,
            "vector_count": vector_count,
            "index_type": index_type,
            "build_type": build_type,
            "indexed_landing_uris": uris,
            "sealed": True,
            "supersedes": [],
            "created_at": time.time(),
        }
        _MEM_SHARDS.append(record)
        # Fire AFTER the row is appended so any subscriber that immediately
        # re-reads `list_shards` sees the new row (the contract the DP cache
        # listener relies on).
        _fire_catalog_notify_memory(notify_payload)
        return _MEM_SHARD_ID
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
INSERT INTO shard_catalog(
  tenant_id, dataset_name, shard_uri, checksum, vector_count, index_type,
  build_type, indexed_landing_uris)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                tenant_id, dataset_name, shard_uri, checksum, vector_count,
                index_type, build_type, uris,
            ),
        )
        shard_id = cur.fetchone()[0]
        # `pg_notify(channel, payload)` runs IN THE SAME TRANSACTION as the
        # INSERT. Postgres holds the notify until commit and delivers it to
        # every subscriber atomically with the row becoming visible — so no
        # listener can observe the notify and then re-query `list_shards`
        # without seeing the new row. Best-effort: if `pg_notify` itself
        # raises (it shouldn't — the call is in-process to Postgres), log and
        # continue. The catalog row is the source of truth; the notify is
        # the optimisation, not the correctness mechanism (the DP's TTL
        # safety net still discovers the row within `RB_CATALOG_FRESHNESS_S`).
        try:
            cur.execute(
                "SELECT pg_notify(%s, %s)",
                ("catalog_updates", json.dumps(notify_payload)),
            )
        except Exception:  # noqa: BLE001 - best-effort emission
            logging.getLogger(__name__).warning(
                "pg_notify(catalog_updates) failed; TTL safety net will cover",
                exc_info=True,
            )
        return shard_id


def list_shards(tenant_id: str, dataset_name: str) -> List[dict]:
    """Return shards for a `(tenant_id, dataset_name)` sorted newest-first."""
    if _MEMORY_MODE:
        return [
            dict(r)
            for r in sorted(_MEM_SHARDS, key=lambda x: x["id"], reverse=True)
            if r["tenant_id"] == tenant_id and r["dataset_name"] == dataset_name
        ]
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
SELECT * FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s
ORDER BY created_at DESC
            """,
            (tenant_id, dataset_name),
        )
        return [dict(r) for r in cur.fetchall()]


def get_latest_shard(tenant_id: str, dataset_name: str) -> Optional[dict]:
    """Return the newest shard for `(tenant_id, dataset_name)`, or None.

    The newest shard is the current/authoritative one: the query path loads
    it, and the index builder reads its `indexed_landing_uris` manifest to
    decide which landing parts still need indexing. Newest-first ordering is
    `id` desc in memory mode and `created_at` desc in Postgres — identical to
    `list_shards`, just the head element.
    """
    shards = list_shards(tenant_id, dataset_name)
    return shards[0] if shards else None


def superseded_shards(
    tenant_id: str, dataset_name: str, keep: int = 2
) -> List[dict]:
    """Return shards eligible for sweeping — every shard older than the newest `keep`.

    `list_shards` is newest-first, so the first `keep` rows are retained and the
    rest are superseded. `keep=2` retains the newest shard plus the one
    immediately before it: that previous shard is the grace buffer for an
    in-flight query that resolved it as `get_latest_shard` just before the
    newest shard was written, and is still faulting its `.bin`/`.meta.json` into
    the local cache. Anything older than that can never be the target of a
    query that started after it was superseded.
    """
    keep = max(1, keep)
    return list_shards(tenant_id, dataset_name)[keep:]


def delete_shards(tenant_id: str, dataset_name: str, shard_ids: List[int]) -> int:
    """Delete `shard_catalog` rows by id for a `(tenant_id, dataset_name)`.

    Returns the number of rows removed. Object-storage cleanup of the shard
    `.bin`/`.meta.json` is the caller's responsibility (the catalog adapter
    does not reach into storage). Scoped by tenant/dataset so a stray id from
    another tenant can never be deleted.
    """
    if not shard_ids:
        return 0
    id_set = set(int(s) for s in shard_ids)
    if _MEMORY_MODE:
        before = len(_MEM_SHARDS)
        _MEM_SHARDS[:] = [
            r for r in _MEM_SHARDS
            if not (
                r["tenant_id"] == tenant_id
                and r["dataset_name"] == dataset_name
                and r["id"] in id_set
            )
        ]
        return before - len(_MEM_SHARDS)
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
DELETE FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s AND id = ANY(%s)
            """,
            (tenant_id, dataset_name, list(id_set)),
        )
        return cur.rowcount


# --- Import jobs (async bulk ingest) --------------------------------------


_IMPORT_FIELDS = (
    "import_id", "tenant_id", "dataset", "format", "status", "error_mode",
    "max_bad_records", "upload_uri", "records_processed", "records_accepted",
    "records_rejected", "rejected_uri", "error_message", "created_at",
    "completed_at",
)


def create_import_job(
    import_id: str,
    tenant_id: str,
    dataset: str,
    fmt: str,
    error_mode: str,
    max_bad_records: Optional[int],
    upload_uri: str,
) -> dict:
    """Insert a new `import_jobs` row in status `awaiting_upload`.

    `upload_uri` is the deterministic object-storage key the client stages the
    file at via its presigned upload. Counters start at 0; the job advances
    through `validating` → `indexing` → `completed` (or `failed`) as the
    validator/builder process it.
    """
    now = _now_iso()
    row = {
        "import_id": import_id,
        "tenant_id": tenant_id,
        "dataset": dataset,
        "format": fmt,
        "status": "awaiting_upload",
        "error_mode": error_mode,
        "max_bad_records": max_bad_records,
        "upload_uri": upload_uri,
        "records_processed": 0,
        "records_accepted": 0,
        "records_rejected": 0,
        "rejected_uri": None,
        "error_message": None,
        "created_at": now,
        "completed_at": None,
    }
    if _MEMORY_MODE:
        global _MEM_IMPORT_SEQ
        _MEM_IMPORT_SEQ += 1
        row["_seq"] = _MEM_IMPORT_SEQ
        _MEM_IMPORTS[(tenant_id, import_id)] = row
        return dict(row)
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
INSERT INTO import_jobs(import_id, tenant_id, dataset, format, error_mode,
                        max_bad_records, upload_uri)
VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING *
            """,
            (import_id, tenant_id, dataset, fmt, error_mode, max_bad_records, upload_uri),
        )
        return dict(cur.fetchone())


def get_import_job(tenant_id: str, import_id: str) -> Optional[dict]:
    """Return the import job for `(tenant_id, import_id)`, or None.

    Tenant-scoped: a job belonging to another tenant returns None, which the
    HTTP layer maps to 404 (the v1 never-leak-existence rule).
    """
    if _MEMORY_MODE:
        row = _MEM_IMPORTS.get((tenant_id, import_id))
        return dict(row) if row else None
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM import_jobs WHERE tenant_id=%s AND import_id=%s",
            (tenant_id, import_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_import_job_by_id(import_id: str) -> Optional[dict]:
    """Return an import job by `import_id` alone (worker-side helper).

    The validator worker only carries the `import_id` on the queue message;
    it has already been admission-checked at create time, so a tenant-scoped
    lookup is unnecessary here.
    """
    if _MEMORY_MODE:
        for row in _MEM_IMPORTS.values():
            if row["import_id"] == import_id:
                return dict(row)
        return None
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM import_jobs WHERE import_id=%s", (import_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_import_jobs(tenant_id: str, dataset: str) -> List[dict]:
    """Return a dataset's import jobs, newest-first."""
    if _MEMORY_MODE:
        rows = [
            dict(r)
            for (tid, _), r in _MEM_IMPORTS.items()
            if tid == tenant_id and r["dataset"] == dataset
        ]
        rows.sort(key=lambda r: r.get("_seq", 0), reverse=True)
        return rows
    with pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
SELECT * FROM import_jobs
WHERE tenant_id=%s AND dataset=%s
ORDER BY created_at DESC
            """,
            (tenant_id, dataset),
        )
        return [dict(r) for r in cur.fetchall()]


def update_import_job(import_id: str, **fields) -> None:
    """Patch an import job's mutable columns by `import_id`.

    Accepts any of: `status`, `records_processed`, `records_accepted`,
    `records_rejected`, `rejected_uri`, `error_message`, `completed_at`.
    Unknown keys are ignored. Used by the validator worker as it advances a
    job through its lifecycle.
    """
    mutable = {
        "status", "records_processed", "records_accepted", "records_rejected",
        "rejected_uri", "error_message", "completed_at",
    }
    patch = {k: v for k, v in fields.items() if k in mutable}
    if not patch:
        return
    if _MEMORY_MODE:
        for row in _MEM_IMPORTS.values():
            if row["import_id"] == import_id:
                row.update(patch)
                return
        return
    cols = ", ".join(f"{k}=%s" for k in patch)
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE import_jobs SET {cols} WHERE import_id=%s",
            (*patch.values(), import_id),
        )


# --- DP residency registry (SSD-cache feature) ----------------------------
#
# The four functions below back the `dp_shard_residency` table (migration
# 007). The producer is the residency writer daemon
# (`services/_common/residency_writer.py`); `list_dp_residency_for_shard`
# is the intended read path for residency-aware CP routing (which prefers
# DPs that already have a shard cached), but is not yet wired into the CP.
# All four functions branch on `_MEMORY_MODE` so the unit suite (in-memory)
# and the integration suite (Postgres) share the same call sites.
#
# Why URI-keyed not catalog-id-keyed: the SSD tier itself is URI-keyed
# (`shard_tier.fetch(uri)` / `shard_tier.evict(uri)`) because a
# content-addressed URI is the stable identifier across builds. Keying
# the registry the same way avoids a join through `shard_catalog` every
# time the writer reconciles.


def register_dp_shard_warm(
    dp_id: str,
    shard_uri: str,
    warm_since: float,
    last_query_at: float,
) -> None:
    """UPSERT a residency row for `(dp_id, shard_uri)`.

    First write inserts; subsequent writes update only `last_query_at`.
    `warm_since` is set once on first admit and intentionally left alone on
    refresh — the operator reads it as "how long has this DP held the
    shard cached", which would be defeated by refreshing it on every cycle.
    The Postgres branch's `ON CONFLICT ... DO UPDATE SET last_query_at` is
    the canonical pattern; the memory branch mirrors it explicitly.

    Both `warm_since` and `last_query_at` are unix epoch seconds
    (`time.time()`), matching the `DOUBLE PRECISION` column type.
    """
    if _MEMORY_MODE:
        # The lock serialises concurrent writers in-process. The full
        # check-then-set must happen under one lock so a concurrent insert
        # cannot land between the existence check and the update.
        with _MEM_DP_RESIDENCY_LOCK:
            existing = _MEM_DP_RESIDENCY.get((dp_id, shard_uri))
            if existing is None:
                _MEM_DP_RESIDENCY[(dp_id, shard_uri)] = (warm_since, last_query_at)
            else:
                # Preserve the original warm_since; only refresh last_query_at.
                _MEM_DP_RESIDENCY[(dp_id, shard_uri)] = (existing[0], last_query_at)
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
INSERT INTO dp_shard_residency(dp_id, shard_uri, warm_since, last_query_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (dp_id, shard_uri)
DO UPDATE SET last_query_at = EXCLUDED.last_query_at
            """,
            (dp_id, shard_uri, warm_since, last_query_at),
        )


def unregister_dp_shard_warm(dp_id: str, shard_uri: str) -> None:
    """DELETE the residency row for `(dp_id, shard_uri)`. Idempotent.

    Removing a row that does not exist is a no-op — the SQL `DELETE`'s
    natural semantics, mirrored in memory by `dict.pop(..., None)`. The
    writer relies on this: a diff cycle may compute the same delete twice
    if a residency entry was already missing on the previous cycle.
    """
    if _MEMORY_MODE:
        with _MEM_DP_RESIDENCY_LOCK:
            _MEM_DP_RESIDENCY.pop((dp_id, shard_uri), None)
        return
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM dp_shard_residency WHERE dp_id=%s AND shard_uri=%s",
            (dp_id, shard_uri),
        )


def list_dp_residency_for_shard(
    shard_uri: str,
) -> List[Tuple[str, float, float]]:
    """Return `[(dp_id, warm_since, last_query_at), ...]` for `shard_uri`.

    Intended read path for residency-aware routing: "given a shard the
    request needs, which DPs already hold it cached?". Not yet wired into
    the CP. The returned list is unordered — the caller picks a winner
    from the set (e.g. most recently queried, or any). Returns an empty
    list when no DP holds the shard, so callers can write
    `for dp_id, _, _ in list_dp_residency_for_shard(...)` without a None
    check.
    """
    if _MEMORY_MODE:
        with _MEM_DP_RESIDENCY_LOCK:
            return [
                (dp_id, warm_since, last_query_at)
                for (dp_id, uri), (warm_since, last_query_at) in _MEM_DP_RESIDENCY.items()
                if uri == shard_uri
            ]
    with pooled_conn() as conn, conn.cursor() as cur:
        # The `dp_shard_residency_shard_uri_idx` (migration 007) makes this
        # an indexed scan even for a popular shard with many resident DPs.
        cur.execute(
            "SELECT dp_id, warm_since, last_query_at "
            "FROM dp_shard_residency WHERE shard_uri=%s",
            (shard_uri,),
        )
        return [(row[0], float(row[1]), float(row[2])) for row in cur.fetchall()]


def list_dp_residency_for_dp(
    dp_id: str,
) -> List[Tuple[str, float, float]]:
    """Return `[(shard_uri, warm_since, last_query_at), ...]` for `dp_id`.

    Operator / observability primitive: "what is this DP currently
    holding?". Used by a future admin surface and operator dashboards.
    The PK `(dp_id, shard_uri)` is the index that makes this an indexed
    lookup.
    """
    if _MEMORY_MODE:
        with _MEM_DP_RESIDENCY_LOCK:
            return [
                (uri, warm_since, last_query_at)
                for (rid, uri), (warm_since, last_query_at) in _MEM_DP_RESIDENCY.items()
                if rid == dp_id
            ]
    with pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT shard_uri, warm_since, last_query_at "
            "FROM dp_shard_residency WHERE dp_id=%s",
            (dp_id,),
        )
        return [(row[0], float(row[1]), float(row[2])) for row in cur.fetchall()]
