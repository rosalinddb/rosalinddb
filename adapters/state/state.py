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
from psycopg2.extras import RealDictCursor, execute_values

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
    008 (`shard_catalog.consolidated_lsn`, the recall-tier watermark — default 0,
    so it is a no-op for behaviour until the recall tier is enabled).
    002 references the `tenants` table for FKs so it must run after 001; 003
    recreates `shard_catalog` so it must run after 002; 004 FKs to
    `dataset_catalog` so it must run after 002. All files live under
    `adapters/state/migrations/`. Memory mode is a no-op.

    This migrates ONLY the control-plane Postgres. The recall-tier (pgvector)
    instance has its own DSN (`RB_RECALL_DSN`) and a separate runner,
    `migrate_recall()` — see that function and `scripts/migrate.py`.

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
    "008_shard_consolidated_lsn",
)


# Ordered list of RECALL-instance migration versions. These run against the
# SEPARATE data-plane pgvector instance (RB_RECALL_DSN), NOT the control-plane
# Postgres above (see docs/architecture/recall-consolidate.md, "Blast radius").
# They live under `migrations/recall/` and are applied by `migrate_recall()`
# with the same advisory-lock + version-ledger discipline as the control-plane
# set. The two ledgers live in different databases by design and never share a
# connection.
_RECALL_MIGRATION_VERSIONS = ("001_recall_vectors",)


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


# --- Recall-tier schema migration -----------------------------------------
#
# The recall tier is a SEPARATE data-plane pgvector instance (RB_RECALL_DSN),
# kept off the control-plane Postgres so a tenant write-storm cannot starve the
# metadata reads on every query's critical path
# (docs/architecture/recall-consolidate.md, "Blast radius & control/data-plane
# isolation"). Its schema is migrated independently from the control-plane
# schema above, with the same advisory-lock + ledger discipline, against its own
# DSN. The whole recall path is DEFAULT-OFF: when RB_RECALL_DSN is unset
# `migrate_recall()` is a no-op, so a flag-off deploy behaves byte-identically to
# today and nothing ever connects to a recall instance.

# Distinct advisory-lock key for the recall ledger. It lives in a different
# database from `_MIGRATE_LOCK_KEY`, so collision is impossible either way, but
# a separate constant keeps the intent clear. Fixed forever once deployed.
_RECALL_MIGRATE_LOCK_KEY = 0x726F73685F686F74  # ASCII "rosh_hot", a stable constant

# Process-local serialisation for recall migrations, mirroring the control-plane
# `_MIGRATE_LOCK` / `_MIGRATED` pair (cross-process safety is the Postgres
# advisory lock inside `_apply_recall_migrations`).
_RECALL_MIGRATE_LOCK = threading.Lock()
_RECALL_MIGRATED = False


def _recall_dsn() -> Optional[str]:
    """Return the recall-tier (pgvector) DSN from `RB_RECALL_DSN`, or None if unset.

    `None` means the recall tier is not configured — every recall path is a
    no-op and the deploy behaves exactly as it does today. A blank/whitespace
    value is treated as unset so an empty compose default cannot accidentally
    enable it.
    """
    raw = os.getenv("RB_RECALL_DSN")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def migrate_recall(force: bool = False) -> bool:
    """Apply the recall-tier (pgvector) schema against `RB_RECALL_DSN`.

    Returns True if migrations were applied (or would have been — i.e. a recall
    DSN is configured), False if the recall tier is OFF (no `RB_RECALL_DSN`) and
    nothing was done. The boolean lets the migration entrypoint print an accurate
    "skipped (recall tier off)" vs "applied" line.

    DEFAULT-OFF: with no `RB_RECALL_DSN` this is a pure no-op — it never opens a
    connection, never imports a pgvector dependency, and leaves behaviour
    identical to today. This is the property that keeps a flag-off
    `docker compose up` byte-identical.

    Memory mode (`DATABASE_URL=memory://...`) does NOT suppress this: the recall
    tier is a wholly separate instance addressed by its own DSN, so a test or a
    deploy can point `RB_RECALL_DSN` at a real pgvector while the control plane
    runs in memory. The gate is `RB_RECALL_DSN`, not `DATABASE_URL`.

    The `_RECALL_MIGRATE_LOCK` / `_RECALL_MIGRATED` pair serialises within a
    process; `_apply_recall_migrations()` takes a Postgres `pg_advisory_xact_lock`
    so several processes booting at once serialise in the recall database instead
    of racing the DDL locks — identical discipline to `migrate()`.

    `force=True` mirrors `migrate(force=True)`: it bypasses the `RB_SKIP_MIGRATE`
    early-return so the dedicated migration entrypoint always applies the schema
    even when it inherits that flag from the service env.
    """
    dsn = _recall_dsn()
    if dsn is None:
        # Recall tier off — nothing to migrate, nothing connects to a recall store.
        return False
    if not force and os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes"):
        # Schema applied out-of-band (same contract as the control-plane
        # migrate()); a recall DSN IS configured, so report True.
        return True
    global _RECALL_MIGRATED
    with _RECALL_MIGRATE_LOCK:
        if _RECALL_MIGRATED:
            return True
        _apply_recall_migrations(dsn)
        _RECALL_MIGRATED = True
    return True


def _apply_recall_migrations(dsn: str) -> None:
    """Apply the ordered recall migration files exactly once each (version-tracked).

    A faithful copy of `_apply_migrations()`'s cross-process-safe pattern, but
    against the recall DSN and the `recall_schema_migrations` ledger:

      - a dedicated (non-pooled) connection to `dsn` — the recall instance has no
        application connection pool;
      - `pg_advisory_xact_lock(_RECALL_MIGRATE_LOCK_KEY)` first, so concurrent
        migrators serialise in the recall database rather than deadlocking on the
        `CREATE EXTENSION` / `CREATE TABLE` locks;
      - a `recall_schema_migrations(version, applied_at)` ledger so a re-run
        applies only un-applied versions. The recall migrations are all
        `IF NOT EXISTS` (additive, non-destructive), so even a direct re-execute
        is safe; the ledger keeps the common path a clean skip.

    There is no legacy-bootstrap branch (unlike the control plane): the recall
    instance is brand new with the recall tier, so there is never a pre-existing
    recall schema without a ledger to reconcile.
    """
    migrations_dir = Path(__file__).parent / "migrations" / "recall"
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s)", (_RECALL_MIGRATE_LOCK_KEY,)
            )
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS recall_schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
                """
            )
            cur.execute("SELECT version FROM recall_schema_migrations")
            applied = {row[0] for row in cur.fetchall()}
            for version in _RECALL_MIGRATION_VERSIONS:
                if version in applied:
                    continue
                sql = (migrations_dir / f"{version}.sql").read_text(
                    encoding="utf-8"
                )
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO recall_schema_migrations(version) VALUES (%s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (version,),
                )
    finally:
        conn.close()


# --- Recall-tier synchronous write path -----------------------------------
#
# The flag-gated, default-off write path that makes a `POST /vectors` durable
# and immediately queryable. It writes ONLY to the separate data-plane recall
# pgvector instance (`RB_RECALL_DSN`) — never the control-plane Postgres — so a
# tenant write-storm cannot starve the catalog reads on every query's critical
# path (docs/architecture/recall-consolidate.md, "Blast radius"). Everything
# here is a no-op unless `recall_enabled()` is True, which the service checks
# before calling in; so a flag-off deploy never opens a recall connection.
#
# Scope note: this is the UPSERT (write) path. The query union, the
# recall→consolidated consolidation, and the get/list/delete CRUD union
# (including recall-delete tombstoning) live in their own sections below.


def recall_enabled() -> bool:
    """Whether the synchronous recall-tier write path is active.

    Mirrors `quotas_enabled()` (the OSS opt-in idiom): defaults OFF and reads
    the env fresh on every call so a test can flip it via monkeypatch without a
    module reload. It is the MASTER switch for recall-tier behaviour.

    Two conditions must BOTH hold for it to be on:
      - `RB_RECALL` is truthy (`1`/`true`/`yes`/`on`, case-insensitive), and
      - `RB_RECALL_DSN` is configured (non-empty) — there is a recall store to
        write to.

    Requiring the DSN as well as the flag means a deploy that flips
    `RB_RECALL=true` but forgets to point `RB_RECALL_DSN` at an instance stays
    on the byte-identical flag-off path rather than erroring on every write.
    """
    if os.getenv("RB_RECALL", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return _recall_dsn() is not None


def _recall_conn() -> "psycopg2.extensions.connection":
    """Return a fresh, DEDICATED psycopg2 connection to the recall store.

    A brand-new connection to `RB_RECALL_DSN` (a full TCP + TLS + auth
    handshake). This is NO LONGER the per-op recall path: every recall CRUD /
    consolidation function now checks a connection out of the recall pool via
    `recall_pooled_conn()` instead (see below). `_recall_conn()` is retained for
    the migration runner's dedicated-connection needs and as a low-level
    factory, never as the hot per-query path it used to be (~48ms/op of pure
    handshake when one was opened per recall op).

    Lazy by construction: only reached when `recall_enabled()` is True. With the
    flag off no caller reaches here and no recall pool is built, so no connection
    to a recall instance is ever opened — the property that keeps a flag-off
    deploy byte-identical.

    Raises `RuntimeError` if `RB_RECALL_DSN` is unset — that is a programming
    error (a caller reached the recall path with the tier off), not an expected
    runtime state.
    """
    dsn = _recall_dsn()
    if dsn is None:
        raise RuntimeError(
            "recall tier is off (RB_RECALL_DSN unset); no recall connection"
        )
    return psycopg2.connect(dsn)


# --- Recall-tier application-side connection pool --------------------------
#
# `_recall_conn()` opened a brand-new connection per recall op — a full TCP +
# TLS + auth handshake (~48ms against a managed pgvector instance). A single
# `POST /vectors` or `POST /v1/query` paid that tax on every recall read/write.
# This pool mirrors the control-plane pool (`_POOL` / `pooled_conn()` above)
# one-for-one, but against the SEPARATE recall DSN (`RB_RECALL_DSN`): it opens a
# small set of recall connections once and hands them out / takes them back via
# `recall_pooled_conn()`.
#
# Independence from the control-plane pool is deliberate and load-bearing: the
# recall tier is a DIFFERENT database instance (its own DSN), so it MUST have its
# own pool — a control-plane connection cannot talk to the recall store and vice
# versa. One recall pool PER PROCESS, exactly like the control-plane pool.
#
# DEFAULT-OFF / lazy: the pool is built on the FIRST `recall_pooled_conn()`
# checkout, which is only ever reached when `recall_enabled()` is True. With
# `RB_RECALL_DSN` unset no recall function runs, so the pool is NEVER built and
# no connection to a recall instance is ever opened — the flag-off deploy stays
# byte-identical (the `psycopg2.connect`-raises tests still pass untouched).

_DEFAULT_RECALL_POOL_MAX = 10  # per-process ceiling; override with RB_RECALL_POOL_MAX
_RECALL_POOL_MIN = 1  # connections kept warm even when idle

# The process-wide recall pool, plus the DSN it was built for. Built lazily on
# the first `recall_pooled_conn()` checkout (NOT at import — flag-off never gets
# here). Keyed on the DSN so a test (or a reconfigure) that rebinds
# `RB_RECALL_DSN` transparently rebuilds the pool against the new instance rather
# than handing out connections to the old one. `_RECALL_POOL_LOCK` guards
# construction so two threads racing the first checkout do not build two pools.
_RECALL_POOL: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_RECALL_POOL_DSN: Optional[str] = None
_RECALL_POOL_LOCK = threading.Lock()


def _recall_pool_max_size() -> int:
    """Return the per-process recall-pool ceiling.

    Mirrors `_pool_max_size()`: defaults to `_DEFAULT_RECALL_POOL_MAX` (10).
    `RB_RECALL_POOL_MAX`, when set to a positive integer, overrides it — useful
    to fit the recall pool under the recall instance's connection cap when many
    worker processes run side by side. A missing or malformed value falls back
    to the default rather than crashing.
    """
    raw = os.getenv("RB_RECALL_POOL_MAX")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return _DEFAULT_RECALL_POOL_MAX


def _get_recall_pool(
    maxconn_override: Optional[int] = None,
) -> psycopg2.pool.ThreadedConnectionPool:
    """Return the process-wide recall pool, building it lazily on first use.

    Mirrors `_get_pool()`: a `ThreadedConnectionPool` (the services run threaded
    under uvicorn, so checkout/return must be thread-safe) built against the
    recall DSN. Keyed on `RB_RECALL_DSN` — if the DSN changed since the pool was
    built (a test rebind, a reconfigure), the old pool is torn down and a fresh
    one is built against the new instance, so a checkout never returns a
    connection to a stale recall store.

    `maxconn_override` forces the pool's max size; it is a test hook (a tiny
    max-1 pool makes a connection leak immediately fatal). In production the size
    comes from `_recall_pool_max_size()`.

    Raises `RuntimeError` if `RB_RECALL_DSN` is unset — reaching here with the
    tier off is a programming error, identical to `_recall_conn()`.
    """
    global _RECALL_POOL, _RECALL_POOL_DSN
    dsn = _recall_dsn()
    if dsn is None:
        raise RuntimeError(
            "recall tier is off (RB_RECALL_DSN unset); no recall connection pool"
        )
    if _RECALL_POOL is None or _RECALL_POOL_DSN != dsn:
        with _RECALL_POOL_LOCK:
            # Re-check under the lock — another thread may have built it while
            # this one waited.
            if _RECALL_POOL is None or _RECALL_POOL_DSN != dsn:
                if _RECALL_POOL is not None:
                    # DSN changed: discard the pool bound to the old instance.
                    try:
                        _RECALL_POOL.closeall()
                    except Exception:
                        pass
                maxconn = maxconn_override or _recall_pool_max_size()
                _RECALL_POOL = psycopg2.pool.ThreadedConnectionPool(
                    minconn=_RECALL_POOL_MIN,
                    maxconn=max(_RECALL_POOL_MIN, maxconn),
                    dsn=dsn,
                )
                _RECALL_POOL_DSN = dsn
    return _RECALL_POOL


def _close_recall_pool() -> None:
    """Close every connection in the recall pool and discard it.

    Mainly a test teardown hook so a test that rebinds `RB_RECALL_DSN` does not
    leave a pool pinned to a stopped recall container. Safe to call when no pool
    exists. Production processes are long-lived and let the pool live for the
    process lifetime.
    """
    global _RECALL_POOL, _RECALL_POOL_DSN
    with _RECALL_POOL_LOCK:
        if _RECALL_POOL is not None:
            _RECALL_POOL.closeall()
            _RECALL_POOL = None
            _RECALL_POOL_DSN = None


@contextlib.contextmanager
def recall_pooled_conn(
    maxconn_override: Optional[int] = None,
) -> Iterator["psycopg2.extensions.connection"]:
    """Check a recall connection out of the recall pool; commit/return on exit.

    The pool-aware replacement for `with contextlib.closing(_recall_conn()) as
    conn, conn:` at every recall call site. It mirrors `pooled_conn()`'s
    STANDALONE path exactly (the recall tier has no request-scoped connection
    middleware — it is a separate data-plane instance, so there is no
    `_REQUEST_CONN` analogue here; every recall op is its own short transaction):

      - on enter: check a connection out of the recall pool (block-with-timeout —
        on `PoolError` it poll-retries until the checkout deadline, so a
        transient burst blocks rather than 500s, and only a sustained exhaustion
        raises `PoolCheckoutTimeout`);
      - on normal exit: `commit()`, then return the connection to the pool;
      - on exception: `rollback()`, then return the connection, then re-raise —
        the connection is returned in EVERY path, never leaked, and a borrower
        never inherits a half-applied or aborted transaction.

    This yields ONE connection for the WHOLE `with` block, so a multi-statement
    recall transaction (e.g. `recall_upsert_vectors`: allocate an LSN block with
    `UPDATE ... RETURNING`, then UPSERT, then commit) runs entirely on a SINGLE
    checked-out connection — the LSN allocation and the UPSERT are the same
    transaction on the same backend, committed once on clean exit. The
    single-connection-per-transaction guarantee the old `closing(_recall_conn())`
    block gave is preserved verbatim; only the connection's lifecycle changed
    from open/close to checkout/return.

    `maxconn_override` is a test hook forwarded to `_get_recall_pool`.

    Only ever reached when `recall_enabled()` is True (every caller is gated), so
    with the flag off the recall pool is never built and no recall connection is
    ever opened.
    """
    pool = _get_recall_pool(maxconn_override=maxconn_override)
    # `state.connect` span — annotated so a trace distinguishes a ~0ms reuse from
    # a genuine new-backend open, exactly as the control-plane pool does. The
    # pool keeps `_RECALL_POOL_MIN` warm and only opens a new backend when every
    # kept connection is checked out, so a checkout that finds a free connection
    # in `_pool` is a reuse.
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        conn = _getconn_with_timeout(pool, _pool_checkout_timeout_s())
    try:
        yield conn
    except BaseException:
        # Roll back so the connection is returned to the pool clean — never
        # carrying a half-applied or aborted transaction into the next borrower —
        # then re-raise.
        try:
            conn.rollback()
        finally:
            pool.putconn(conn)
        raise
    else:
        # Return the connection on EVERY exit — even if commit() itself raises
        # (broken backend / TLS reset / statement_timeout on COMMIT). Otherwise a
        # commit failure leaks the conn from the pool's accounting, permanently
        # shrinking the pool. psycopg2._putconn inspects transaction_status and
        # rolls back / closes a non-idle or server-lost conn before re-pooling,
        # so a post-commit-failure conn is returned clean, not poisoned.
        try:
            conn.commit()
        finally:
            pool.putconn(conn)


def _to_pgvector_literal(values: List[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. `[1.0,2.0,3.0]`.

    pgvector accepts its `vector` input as a bracketed, comma-separated string;
    psycopg2 binds it as the parameter for the unparameterised `embedding`
    column. The dimension is NOT enforced here — the service validates each
    record's length against `dataset.dimension` before this is called, exactly
    as the cold path does.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def recall_upsert_vectors(
    tenant_id: str,
    dataset: str,
    records: List[dict],
) -> int:
    """Synchronously UPSERT validated records into the recall tier; return the count.

    For each record (`{"id", "values", "metadata"}`, already validated by the
    service): the row is UPSERTed into `recall_vectors` with last-write-wins on
    `(tenant_id, dataset, id)` and `deleted = false`, stamped with a strictly-
    monotonic per-`(tenant, dataset)` LSN from the recall store's `recall_lsn_seq`.
    A re-sent id overwrites the prior row (new embedding, metadata, and a fresh
    higher LSN) — never a duplicate.

    The LSN is generated in the RECALL store, so this path never touches the
    control-plane Postgres (docs/architecture/recall-consolidate.md, "The
    watermark"). All records for one call share a single recall connection +
    transaction: the whole batch commits atomically, so a mid-batch failure
    leaves the recall tier unchanged rather than half-applied.

    Set-based round-trips (was 2N, now ~2): the prior implementation allocated
    one LSN and ran one UPSERT *per record* (2N round-trips on the read-your-
    writes path). Instead this allocates ALL N LSNs in a SINGLE upsert-increment
    (`last_lsn = last_lsn + N RETURNING last_lsn`) and applies the batch in a
    SINGLE multi-row UPSERT. The seq-row upsert is still serialised by Postgres,
    so `last_lsn` stays strictly monotonic per (tenant, dataset) with no
    cross-dataset contention; the allocated block is `last_lsn-N+1 .. last_lsn`,
    assigned in input order.

    Intra-batch duplicate ids: if the same id appears more than once in one
    batch, only the LAST occurrence is written (last input wins), so the batch
    is collapsed to one row per id BEFORE allocation — N is the number of
    distinct ids, and a single multi-row UPSERT never lists the same conflict
    key twice (which Postgres would reject as "cannot affect row a second time").

    Only ever called when `recall_enabled()` is True (the service gates it),
    so it never runs — and never opens a connection — with the flag off.
    """
    if not records:
        return 0

    # Collapse intra-batch duplicate ids: later input wins (last-write-wins,
    # consistent with the cross-batch ON CONFLICT below). Preserve input order
    # of each surviving id — its position is that of its LAST occurrence, so the
    # LSN block is assigned in the order the winning records were sent. A dict
    # keeps insertion order (Python 3.7+); re-assigning an existing key updates
    # the value but NOT its position, so we del-then-set to move a repeated id
    # to the end (its latest occurrence).
    deduped: dict[str, dict] = {}
    for rec in records:
        rid = rec["id"]
        if rid in deduped:
            del deduped[rid]
        deduped[rid] = rec
    winners = list(deduped.values())
    n = len(winners)

    # ONE pooled recall connection for the WHOLE batch — the LSN-block allocation
    # and the multi-row UPSERT run in a SINGLE transaction on a SINGLE checked-out
    # connection. `recall_pooled_conn()` owns the lifecycle: it commits on clean
    # exit (the batch is all-or-nothing), rolls back on error, and returns the
    # connection to the recall pool on EVERY path. Holding one connection for the
    # duration of the txn — allocate, upsert, commit, then return — is what keeps
    # the LSN allocation and the UPSERT atomic (read-your-writes correctness).
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        # 1. Allocate the whole LSN block in ONE statement. The upsert-increment
        #    is serialised by Postgres on the single seq row, so `last_lsn` is
        #    strictly monotonic with no cross-dataset contention. Bumping by N at
        #    once reserves a contiguous block: the returned value is the LAST LSN
        #    in the block, and the block is `last_lsn-N+1 .. last_lsn`.
        cur.execute(
            """
INSERT INTO recall_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, %s)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = recall_lsn_seq.last_lsn + %s
RETURNING last_lsn
            """,
            (tenant_id, dataset, n, n),
        )
        last_lsn = cur.fetchone()[0]
        first_lsn = last_lsn - n + 1
        # 2. SINGLE multi-row UPSERT. Each winner is stamped with its LSN from
        #    the block in input order. Last-write-wins on (tenant, dataset, id):
        #    a re-sent id overwrites embedding/metadata/lsn and clears any prior
        #    tombstone (deleted -> false), so an upserted id is live.
        rows = [
            (
                tenant_id,
                dataset,
                rec["id"],
                _to_pgvector_literal(rec["values"]),
                json.dumps(rec.get("metadata") or {}),
                first_lsn + offset,
            )
            for offset, rec in enumerate(winners)
        ]
        execute_values(
            cur,
            """
INSERT INTO recall_vectors (tenant_id, dataset, id, embedding, metadata, lsn, deleted)
VALUES %s
ON CONFLICT (tenant_id, dataset, id)
DO UPDATE SET
  embedding = EXCLUDED.embedding,
  metadata  = EXCLUDED.metadata,
  lsn       = EXCLUDED.lsn,
  deleted   = FALSE
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, FALSE)",
        )
    return n


# --- Recall-tier read path (brute-force exact search) ----------------------
#
# The query union (PR4): when the recall tier is on, `POST /v1/query` searches
# BOTH the consolidated (cold) shard AND the recall tier, then merges. This is
# the recall half — a brute-force exact L2 scan over `recall_vectors`, scoped to
# one `(tenant, dataset)` partition and to the rows ABOVE the resolved shard's
# watermark (`lsn > consolidated_lsn`). It returns BOTH live rows and tombstones
# (`deleted` rows above the watermark) so the merge in `v1_query` can suppress a
# consolidated id whose recall version is a tombstone. The metric is aligned to
# FAISS by SQUARING pgvector's plain-L2 `<->` (both tiers store the same
# un-normalised vectors). See docs/architecture/recall-consolidate.md, "Read
# path — the union" and invariants I1/I3.


def recall_search(
    tenant_id: str,
    dataset: str,
    vector: List[float],
    top_k: int,
    watermark: int,
    flt: Optional[dict] = None,
) -> Tuple[set, List[dict]]:
    """Brute-force exact L2 search of the recall tier above `watermark`.

    Returns `(suppress_ids, matches)`:

      - `suppress_ids` — the set of EVERY recall id above the watermark (every
        live row AND every tombstone), regardless of the filter and regardless of
        `top_k`. Recall is authoritative for any id with `lsn > watermark`, so the
        merge uses this set to drop the stale cold copy of that id unconditionally
        (live-but-filtered-out, ranked-past-top_k, and tombstoned ids all
        suppress their cold twin). Keying suppression on "any recall row for this
        id" — not "a recall row that became a match" — is the fix for the leak
        where a re-upsert that fails the filter (or ranks past `top_k`) let a
        stale cold copy survive.
      - `matches` — up to `top_k` filter-passing LIVE rows as dicts
        `{"id", "score", "metadata", "deleted"}`, ascending by `score`, the
        candidate recall MATCHES. Tombstones are NEVER matches; a live row that
        fails the filter is NOT a match (it only suppresses, via `suppress_ids`).

    `score` is the FAISS-aligned **L2-squared** distance — pgvector's `<->` (plain
    Euclidean L2) SQUARED, so it can be merged directly with the cold shard's
    FAISS L2² distances. Both tiers store the same UN-normalised vectors, so
    squaring is the only alignment needed (the most likely silent ranking bug —
    it has a dedicated test).

    Scope (I1 partition + I3 watermark pairing):
      WHERE tenant_id=? AND dataset=? AND lsn > :watermark
    The recall tier owns `lsn > consolidated_lsn`; the cold shard owns `<=` — so
    the union is complete with no double-count. `watermark` MUST be the
    `consolidated_lsn` of the shard the cold search ACTUALLY resolved (the caller
    pairs them), never a watermark read independently.

    The metadata `filter` (AND-of-equals, same semantics as the cold path's
    `metadata_matches_filter`) is applied to LIVE rows in Python after the scan,
    NOT pushed into SQL — the recall set is small by construction (bounded by
    consolidation cadence, not data size) so an exact scan + Python filter is
    sub-millisecond and keeps the equality semantics byte-identical to the cold
    path (exact type+value, no JSONB coercion surprises). The filter gates only
    whether a live row becomes a MATCH; it never removes an id from
    `suppress_ids`.

    SINGLE-SNAPSHOT SCAN — the suppress/match split comes from ONE SQL statement
    (task #17, b1 root cause). `suppress_ids` and `matches` are NOT two separate
    scans: they are derived in Python from the rows of a SINGLE SELECT over
    `lsn > watermark`. This is load-bearing for correctness, not a style choice:

      - The old shape ran TWO statements on a default READ COMMITTED recall
        connection — (a) a MATCH scan over live rows, then (b) a SUPPRESS scan
        over all ids. Under READ COMMITTED EACH `cur.execute` takes a FRESH MVCC
        snapshot, so a re-UPSERT that COMMITS BETWEEN scan (a) and scan (b) makes
        (b) observe ids that (a) never saw. The result: `suppress_ids ⊋
        match_ids`, the union over-suppresses, and a query that should return N
        live rows transiently returns 0 (benchmark case b1; root cause in
        bench-lab/analysis/b1-rootcause.md). The write/consolidate side
        (`recall_snapshot_for_consolidation`) was already hardened to a single
        statement for exactly this reason — this read path now matches it.
      - A SINGLE SELECT is evaluated against ONE MVCC snapshot EVEN UNDER READ
        COMMITTED (a statement-level snapshot is taken once, at statement start,
        and is stable for the whole statement). So `suppress_ids` (every returned
        id) and `matches` (the live subset) are split from the SAME consistent
        row set — `suppress_ids ⊇ match_ids` ALWAYS holds, no concurrent commit
        can wedge between them. This is precisely what eliminates the b1 race.

    The scan returns `(id, deleted, score, metadata)` so the tombstone/live split
    is done in Python BY THE EXPLICIT `deleted` FLAG, not by a `... AND NOT
    deleted` clause and not by SQL row ordering. That preserves the "no `LIMIT`
    pushdown can crowd out a real match with a tombstone" property: tombstones are
    excluded from `matches` by `if deleted` in Python (they only ever land in
    `suppress_ids`), so even a tombstone's zero-vector placeholder embedding
    (distance 0 to any query) can never displace a real live match. There is no
    SQL `LIMIT`; the live-row `top_k` truncation happens in Python over the
    live-only subset, and the recall set is small by construction (bounded by
    consolidation cadence, not data size — §Recall search).

    Only ever called when `recall_enabled()` is True (the query path gates it),
    so it never runs — and never opens a recall connection — with the flag off.
    """
    qlit = _to_pgvector_literal(vector)
    # ONE statement, ONE MVCC snapshot. Select every row above the watermark with
    # its `deleted` flag and FAISS-aligned squared-L2 `score`, and split it in
    # Python: `suppress_ids` = every id, `matches` = the live, filter-passing,
    # top_k-closest subset. DO NOT split this back into two scans — two
    # `cur.execute`s are two READ COMMITTED snapshots and a between-scan commit
    # re-opens the b1 over-suppression race (see the docstring).
    #
    # `power(embedding <-> %s, 2)` squares pgvector's plain-L2 `<->` so the score
    # matches FAISS L2². The query vector is bound TWICE — once for the score
    # expression, once for the ORDER BY — so the same literal drives both. The
    # `ORDER BY` ranks ALL rows (live and tombstoned) by raw distance, but the
    # Python split filters tombstones out by the `deleted` flag, so ordering never
    # lets a tombstone become a match. No SQL `LIMIT`: the `top_k` truncation
    # happens in Python over the live-only subset (the recall set is small by
    # construction — bounded by consolidation cadence, not data size).
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT id,
       deleted,
       power(embedding <-> %s, 2) AS score,
       metadata
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND lsn > %s
ORDER BY embedding <-> %s ASC
            """,
            (qlit, tenant_id, dataset, watermark, qlit),
        )
        rows = cur.fetchall()

    flt = flt or {}
    # `suppress_ids`: EVERY id the single scan returned (live AND tombstoned).
    # Recall is authoritative for `lsn > watermark`, so the stale cold copy of any
    # such id is always dropped — whether the recall row is a tombstone, fails the
    # filter, or ranks past `top_k`. The cold id's FAISS distance is over the
    # CONSOLIDATED embedding, which may differ from the (stale) recall embedding,
    # so we cannot assume the cold match ranks beyond `top_k` just because the
    # recall row does. Because suppress and matches come from the SAME snapshot's
    # rows, `suppress_ids ⊇ match_ids` always holds (the b1 fix).
    suppress_ids: set = set()
    matches: List[dict] = []
    live_count = 0
    for rid, deleted, score, metadata in rows:
        # Every returned id suppresses its cold twin, ALWAYS — added before any
        # match/filter/top_k decision so a tombstone, a filter-failing row, or a
        # past-top_k row still drops its stale cold copy.
        suppress_ids.add(rid)
        # Tombstone: NEVER a match. It only suppresses (already added above).
        # Splitting on the explicit `deleted` flag — not a `NOT deleted` WHERE
        # clause — is what keeps the no-leak property safe even though the scan
        # has no SQL `LIMIT` and orders all rows together by raw distance.
        if deleted:
            continue
        # Live row past the top_k closest live survivors: it cannot be a MATCH
        # (the `top_k` closer live rows already beat it). Its id is already in
        # `suppress_ids`, so skip it as a candidate and keep scanning (so its
        # later id is still suppressed, even though it is not a match).
        if live_count >= top_k:
            continue
        # `metadata` is JSONB → psycopg2 decodes it to a Python dict (or None for
        # a SQL NULL); coalesce None to `{}`.
        meta = metadata or {}
        # Live row: apply the AND-of-equals filter to decide whether it is a
        # MATCH. A live row that fails the filter is NOT a match — but its id is
        # ALREADY in `suppress_ids`, so the stale cold copy is still dropped (the
        # P1 fix: an authoritative live re-upsert that fails the query filter must
        # not let a stale, filter-matching cold copy leak). The predicate must
        # match `metadata_matches_filter` exactly (type+value, no coercion).
        if flt and not _metadata_matches_filter(meta, flt):
            continue
        matches.append(
            {"id": rid, "score": float(score), "metadata": meta, "deleted": False}
        )
        live_count += 1
    return suppress_ids, matches


def _metadata_matches_filter(metadata: dict, flt: dict) -> bool:
    """AND-of-equals predicate — the recall-side mirror of the cold path's
    `services.query_api.v1_query.metadata_matches_filter`.

    Duplicated (kept tiny and in sync) rather than imported so the state adapter
    does not take a dependency on the query service package — the same
    classifier-duplication rationale the query path already uses for its
    ephemeral-runner twin. Semantics are identical: every filter key must be
    present with an EXACTLY-equal value (same `type()` and `==`); a `null` filter
    value never matches; an empty filter matches everything.
    """
    for key, want in flt.items():
        if want is None:
            return False
        if key not in metadata:
            return False
        got = metadata[key]
        if type(got) is not type(want):
            return False
        if got != want:
            return False
    return True


# --- Recall-tier CRUD union helpers (get / list / delete) -----------------
#
# The get/list/delete-by-id union (PR6): when the recall tier is on, the
# consolidated-tier CRUD endpoints (`services.source_registry.main`) union the
# immutable shard sidecar with the recall tier, with RECALL AUTHORITATIVE for any
# id above the resolved shard's watermark — the same recall-wins / tombstone-
# suppress rule the QUERY union uses (`recall_search` + `_merge_recall_and_cold`).
# These point-lookup / partition-scan helpers back that union; they ONLY ever run
# when the endpoint reaches the `recall_enabled()`-gated path, so a flag-off
# deploy never opens a recall connection through here. See
# docs/architecture/recall-consolidate.md, "Read path — the union" / PR6.


def recall_get_vector(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict]]:
    """Point-look up one id in the recall tier above `watermark`. Recall-authoritative.

    Returns a `(status, metadata)` pair the CRUD get-by-id union consumes:

      - `("live", metadata)` — a LIVE recall row with `lsn > watermark`: recall
        wins, the endpoint returns this `{id, metadata}` (its version is newer
        than any consolidated copy).
      - `("tombstone", None)` — a recall TOMBSTONE with `lsn > watermark`: the id
        was deleted in recall; the endpoint returns `404 not_found` and MUST NOT
        fall back to the (stale) cold sidecar.
      - `(None, None)` — NO recall row above the watermark for this id: the
        endpoint falls back to the cold sidecar lookup (the id, if it exists, is
        consolidated).

    Scope (I1 partition + I3 watermark pairing): `WHERE tenant_id=? AND dataset=?
    AND id=? AND lsn > :watermark`. The recall tier owns `lsn > consolidated_lsn`;
    a row at or below the watermark is already consolidated into the cold shard
    and is harmlessly ignored here (the cold sidecar is then authoritative for it).
    `watermark` MUST be the `consolidated_lsn` of the shard the caller resolved.

    Tenant-scoped via the partition key; a cross-tenant id is simply absent from
    this partition → `(None, None)` → the caller's cold lookup (also tenant-
    scoped) yields the 404. Only ever called when `recall_enabled()` is True.
    """
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT metadata, deleted
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND id = %s AND lsn > %s
            """,
            (tenant_id, dataset, vector_id, watermark),
        )
        row = cur.fetchone()
    if row is None:
        return None, None
    metadata, deleted = row
    if deleted:
        return "tombstone", None
    return "live", (metadata or {})


def recall_get_vector_with_embedding(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict], Optional[List[float]]]:
    """Like :func:`recall_get_vector`, but ALSO return the stored embedding.

    Backs the `?include_values=true` recall path for get-by-id — the cheap
    recall-resident case (no FAISS): the embedding is a plain `vector` COLUMN on
    `recall_vectors`, so this is one SELECT over the same partition/watermark
    scope as `recall_get_vector`, plus a text→`list[float]` parse of the column.

    Returns a `(status, metadata, embedding)` triple:

      - `("live", metadata, [float, ...])` — a LIVE recall row above the
        watermark: its real stored embedding is returned alongside the metadata.
      - `("tombstone", None, None)` — a recall TOMBSTONE above the watermark.
      - `(None, None, None)` — NO recall row above the watermark for this id; the
        caller falls back to the cold tier (which, for `include_values`, cannot
        currently reconstruct the embedding — a deferred FAISS `reconstruct`).

    Scope is byte-identical to `recall_get_vector` (I1 partition + I3 watermark
    pairing). Only ever called when `recall_enabled()` is True.
    """
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT metadata, deleted, embedding
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND id = %s AND lsn > %s
            """,
            (tenant_id, dataset, vector_id, watermark),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None
    metadata, deleted, embedding = row
    if deleted:
        return "tombstone", None, None
    return "live", (metadata or {}), _pgvector_literal_to_list(embedding)


def recall_list_rows(
    tenant_id: str, dataset: str, watermark: int
) -> Tuple[List[dict], set]:
    """List the recall partition above `watermark` for the CRUD list union.

    Returns `(live_rows, suppress_ids)`, mirroring `recall_search`'s split so the
    list union dedups identically to the query union:

      - `live_rows` — `[{"id", "metadata"}, ...]` for every LIVE (not-deleted)
        recall row with `lsn > watermark`. These are unioned with the cold sidecar
        entries, recall-wins on the same id (recall metadata overrides cold).
      - `suppress_ids` — the FULL set of EVERY recall id above the watermark (live
        AND tombstoned). Recall is authoritative for `lsn > watermark`, so the
        stale cold copy of any such id is dropped: a tombstone HIDES the cold id
        from the list (the mirror of the query union's tombstone suppression), and
        a live recall row replaces the cold copy with the recall version.

    No metadata filter is applied here — the endpoint applies the AND-of-equals
    filter to the merged result (matching the cold list, which filters after the
    sidecar read), so the filter semantics stay byte-identical across tiers.

    Scope (I1 + I3): `WHERE tenant_id=? AND dataset=? AND lsn > :watermark`. The
    recall set is small by construction (consolidation-bounded), so materialising
    the whole partition is cheap. Only ever called when `recall_enabled()` is True.
    """
    # TWO SEPARATE SCANS, mirroring `recall_search`'s split — DO NOT collapse them
    # into one "all rows then skip `deleted` in Python" scan:
    #   (a) the LIVE scan — `... AND NOT deleted ...` — yields only the rows the
    #       list union surfaces, so a tombstone's placeholder embedding/metadata
    #       can never leak into `live_rows` as a property of the QUERY, not of a
    #       Python-side skip.
    #   (b) the SUPPRESS scan — every id above the watermark (live AND tombstoned)
    #       — preserves tombstone-suppress EXACTLY: a tombstone HIDES the cold id
    #       from the list, a live recall row replaces the cold copy.
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        # (a) LIVE scan: not-deleted rows only.
        cur.execute(
            """
SELECT id, metadata
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND NOT deleted AND lsn > %s
            """,
            (tenant_id, dataset, watermark),
        )
        live_db_rows = cur.fetchall()
        # (b) SUPPRESS scan: EVERY id above the watermark (live AND tombstoned).
        cur.execute(
            """
SELECT id
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND lsn > %s
            """,
            (tenant_id, dataset, watermark),
        )
        suppress_rows = cur.fetchall()
    # Every recall id above the watermark suppresses its stale cold twin
    # (tombstone OR live), exactly as `recall_search` builds `suppress_ids`.
    suppress_ids: set = {rid for (rid,) in suppress_rows}
    live_rows: List[dict] = [
        {"id": rid, "metadata": metadata or {}} for rid, metadata in live_db_rows
    ]
    return live_rows, suppress_ids


def recall_delete_vector(
    tenant_id: str, dataset: str, vector_id: str, dimension: int
) -> int:
    """Write an ABOVE-watermark tombstone for one id; return the allocated lsn.

    The recall-delete write path (PR6). DELETE is a last-write-wins UPSERT of a
    TOMBSTONE row (`deleted = true`) keyed on `(tenant, dataset, id)`, stamped
    with a FRESH lsn allocated from `recall_lsn_seq` — the SAME atomic upsert-
    increment the recall UPSERT uses. Returns that lsn.

    HARD CONTRACT — the lsn MUST be strictly ABOVE the current watermark (it is,
    because every allocated lsn is `> max(lsn) >= consolidated_lsn`). We
    deliberately ALLOCATE A NEW LSN rather than flip `deleted=true` in place /
    reuse the row's old lsn: a tombstone at or below the watermark would be
    EXCLUDED from every union (`lsn > consolidated_lsn` is false → the id never
    deletes) AND would be trim-eligible-but-unapplied (the consolidation could
    GC the row before folding the delete → the id resurrects). Allocating fresh,
    above the max, guarantees the tombstone is in the recall partition the union
    scans and the next consolidation applies. See
    docs/architecture/recall-consolidate.md, "Write path" (Delete) + invariants
    I1/I2 + the failure-mode "Delete then immediate query".

    `dimension` is the dataset's embedding dimension (the caller passes
    `dataset.dimension`). A brand-new tombstone — deleting an id present only in
    the COLD tier, with no prior recall row — still needs an `embedding` for the
    `NOT NULL` column, and we write a zero-vector of the dataset dimension so it
    matches the partition's dimension. NOTE: `recall_search` now EXCLUDES
    tombstones from its MATCH scan in SQL (`... AND NOT deleted ...`), so the
    placeholder is never ranked against a query vector and can no longer trigger
    pgvector's "different vector dimensions" error or crowd out a real match — the
    placeholder's dimension is now a belt-and-suspenders match for the column, not
    a correctness dependency of the search ranking. It is never folded into a
    shard either (consolidation only `_remove_ids`'s a tombstoned id). On a
    conflict the existing embedding is left untouched.

    Synchronous + read-your-deletes: the row is committed before this returns, so
    an immediate GET sees the tombstone and a `POST /query` no longer returns the
    id. Only ever called when `recall_enabled()` is True (the endpoint gates it),
    so it never opens a recall connection with the flag off.
    """
    # Zero-vector placeholder of the dataset dimension (see docstring): same
    # dimension as every live row in the partition, so the search scan's `<->`
    # never hits a dimension mismatch on a cold-only delete's fresh tombstone.
    placeholder = _to_pgvector_literal([0.0] * max(1, int(dimension)))
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        # 1. Allocate a SINGLE fresh lsn from the per-(tenant, dataset) sequence —
        #    the same atomic upsert-increment the write path uses, so the tombstone
        #    is stamped strictly above every prior lsn (and thus above any
        #    watermark, which can only be <= max(lsn)). This is the contract that
        #    keeps the tombstone inside the union's scan window.
        cur.execute(
            """
INSERT INTO recall_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, 1)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = recall_lsn_seq.last_lsn + 1
RETURNING last_lsn
            """,
            (tenant_id, dataset),
        )
        lsn = cur.fetchone()[0]
        # 2. UPSERT the tombstone (last-write-wins on the partition key). An
        #    existing live/deleted row for this id is overwritten with deleted=true
        #    and the fresh higher lsn; a brand-new tombstone gets the dimension-
        #    matched zero placeholder. On a conflict the embedding is untouched.
        cur.execute(
            """
INSERT INTO recall_vectors (tenant_id, dataset, id, embedding, metadata, lsn, deleted)
VALUES (%s, %s, %s, %s, '{}'::jsonb, %s, TRUE)
ON CONFLICT (tenant_id, dataset, id)
DO UPDATE SET
  lsn     = EXCLUDED.lsn,
  deleted = TRUE
            """,
            (tenant_id, dataset, vector_id, placeholder, lsn),
        )
    return int(lsn)


# --- Recall-tier consolidation (flush) helpers ----------------------------
#
# The recall→consolidated flush (PR5): the index builder snapshots a
# (tenant, dataset) recall partition up to the current max LSN, folds the LIVE
# rows into a new Consolidated shard, applies tombstones, commits the catalog
# row with that LSN as the watermark, and THEN trims the consolidated recall
# rows — grace-bounded and idempotent (I2 + I4). The four read/finder helpers
# plus the trim below back that operation; they ONLY ever run when the builder
# reaches the consolidation path, which is gated on `recall_enabled()`, so a
# flag-off deploy never opens a recall connection through here. See
# docs/architecture/recall-consolidate.md, "Consolidation / flush".


def recall_snapshot_for_consolidation(
    tenant_id: str, dataset: str
) -> Tuple[int, List[dict]]:
    """Snapshot a recall partition for a consolidation: `(max_lsn, rows)`.

    Reads EVERY row for `(tenant, dataset)` — live rows AND tombstones — up to
    the partition's `max(lsn) = N`, in ONE statement so the bound `N` and the
    rows it selects come from a SINGLE MVCC snapshot. The bound is derived in a
    scalar sub-SELECT (`lsn <= (SELECT COALESCE(MAX(lsn),0) FROM recall_vectors
    WHERE ...)`), so the whole `SELECT` is one statement and the rows are exactly
    those at or below the max LSN as of that statement's snapshot — self-
    consistent independent of writer internals (no inter-statement gap, no
    reliance on the recall-write seq-lock / commit-order==LSN-order coupling).
    A concurrent write that commits during the snapshot has a higher LSN than the
    max captured by the sub-SELECT, so it is excluded by the `lsn <=` bound and
    left in recall — not partly-included and not lost.

    The watermark `N` is then derived from the max LSN of the RETURNED rows (`0`
    if empty), so it matches the row set exactly and the caller commits the
    catalog row with `consolidated_lsn = N`. The builder folds the live rows into
    the new shard, applies the tombstones (`deleted=true` ids are removed and
    never added), and stamps that watermark.

    Returns:
      - `max_lsn` — the highest LSN among the returned rows, the watermark the
        consolidation will stamp (`0` when the partition is empty — nothing to
        consolidate).
      - `rows` — `[{"id", "values", "metadata", "lsn", "deleted"}, ...]` for
        every row with `lsn <= max_lsn`, ascending by LSN. `values` is the
        parsed float list (pgvector's text `vector` literal → `list[float]`);
        `metadata` is the JSONB dict (coalesced to `{}`).

    A write that lands AFTER the snapshot (a higher LSN) is left in recall
    untouched: it is simply not in this consolidation's set, stays queryable via
    the union (`lsn > consolidated_lsn`), and the next consolidation folds it.
    This is what makes read-your-writes hold THROUGH a consolidation (I1 + I2).
    """
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        # ONE statement: derive the bound N inside the same query via a scalar
        # sub-SELECT, so N and the selected rows share a single MVCC snapshot. A
        # concurrent write (higher LSN) is excluded by the `lsn <=` bound, so the
        # snapshot is internally consistent without depending on transaction
        # isolation level or on the recall writer's seq-lock allocation order.
        cur.execute(
            """
SELECT id, embedding, metadata, lsn, deleted
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s
  AND lsn <= (
    SELECT COALESCE(MAX(lsn), 0) FROM recall_vectors
    WHERE tenant_id = %s AND dataset = %s
  )
ORDER BY lsn ASC
            """,
            (tenant_id, dataset, tenant_id, dataset),
        )
        rows: List[dict] = []
        for rid, embedding, metadata, lsn, deleted in cur.fetchall():
            rows.append(
                {
                    "id": rid,
                    "values": _pgvector_literal_to_list(embedding),
                    "metadata": metadata or {},
                    "lsn": int(lsn),
                    "deleted": bool(deleted),
                }
            )
    # Derive the watermark from the returned rows so it matches the set exactly
    # (the rows are ascending by LSN). Empty partition -> 0, nothing to do.
    max_lsn = rows[-1]["lsn"] if rows else 0
    return max_lsn, rows


def _pgvector_literal_to_list(literal) -> List[float]:
    """Parse pgvector's text `vector` literal (`[1,2,3]`) to a float list.

    psycopg2 reads an unparameterised pgvector `vector` column back as its text
    representation (a bracketed, comma-separated string) since no typecaster is
    registered. Invert `_to_pgvector_literal`. A `list`/`tuple` (already parsed
    by a future typecaster) is passed through; an empty/blank literal yields
    `[]`.
    """
    if isinstance(literal, (list, tuple)):
        return [float(v) for v in literal]
    s = str(literal).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return []
    return [float(part) for part in s.split(",")]


def recall_partition_count(tenant_id: str, dataset: str) -> int:
    """Return the number of recall rows for `(tenant, dataset)` (live + tombstones).

    Backs the per-tenant cap (`RB_RECALL_MAX_ROWS`): the write path calls this
    after a recall UPSERT and enqueues `CONSOLIDATE` when the count exceeds the
    cap. Counts every row in the partition (a tombstone still occupies a row and
    still costs the brute-force scan), so the cap genuinely bounds the recall
    set the union scans.
    """
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM recall_vectors "
            "WHERE tenant_id = %s AND dataset = %s",
            (tenant_id, dataset),
        )
        return int(cur.fetchone()[0] or 0)


def recall_trim(tenant_id: str, dataset: str, grace_watermark: int) -> int:
    """Hard-delete consolidated recall rows up to `grace_watermark`; return the count.

    The FINAL step of a consolidation, run STRICTLY AFTER the catalog commit
    (I2): `DELETE FROM recall_vectors WHERE lsn <= grace_watermark`. Idempotent
    and re-runnable — a crash between commit and trim leaves rows that the union
    harmlessly excludes (`lsn > consolidated_lsn`) and the next consolidation's
    trim GCs (cross-DB crash safety).

    `grace_watermark` is the GRACE-BOUNDED watermark (I4): the caller passes the
    `consolidated_lsn` of the shard that is now the **2nd-newest** (NOT the
    newest just committed), so an in-flight query that resolved an older shard
    still finds its recall rows. `grace_watermark <= 0` is a no-op (nothing has
    aged into the grace window yet — e.g. the very first consolidation, whose
    new shard is the only one).
    """
    if grace_watermark <= 0:
        return 0
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM recall_vectors "
            "WHERE tenant_id = %s AND dataset = %s AND lsn <= %s",
            (tenant_id, dataset, grace_watermark),
        )
        return cur.rowcount


def recall_idle_partitions(idle_seconds: float) -> List[Tuple[str, str]]:
    """Return `(tenant_id, dataset)` partitions whose newest recall write is idle.

    Backs consolidate-on-idle (`RB_RECALL_IDLE_S`): the builder's idle-tick
    sweep calls this, then enqueues `CONSOLIDATE` for each returned partition so
    it drains to ZERO recall rows → idle queries skip pgvector entirely
    (scale-to-zero preserved). "Idle" = the partition's most recent write
    (`max(created_at)`) is older than `idle_seconds` ago AND the partition still
    has rows to drain. Returns an empty list when nothing is idle.

    Grouped per `(tenant, dataset)` so a single SQL round-trip finds every idle
    partition across the whole recall instance in one pass.
    """
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=idle_seconds)
    with recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT tenant_id, dataset
FROM recall_vectors
GROUP BY tenant_id, dataset
HAVING MAX(created_at) <= %s
            """,
            (cutoff,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


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
    consolidated_lsn: int = 0,
) -> int:
    """Insert a new shard record and return its ID.

    Two columns support incremental indexing:
      - `build_type`: `'full'` (trained-from-scratch), `'incremental'`
        (existing index loaded, only new vectors `index.add()`-ed),
        `'delete'` (existing index loaded, one vector removed by id — a
        delete-driven rebuild, labelled distinctly so deletes are not
        miscounted as ingests in `build_type`-keyed metrics), or
        `'consolidate'` (recall→consolidated flush: the recall partition up to
        `consolidated_lsn` is folded into a new shard — see
        docs/architecture/recall-consolidate.md, "Consolidation / flush").
      - `indexed_landing_uris`: the manifest of landing parquet part URIs
        already folded into this shard. The index builder reads the *newest*
        shard's manifest to decide which landing parts are new, so a
        subsequent ingest never re-reads previously indexed uploads.

    `consolidated_lsn` (migration 008) is the recall-tier watermark: the highest
    recall LSN folded into any shard of this dataset so far (a per-dataset high-
    water mark). It partitions every vector into exactly one tier —
    `lsn <= consolidated_lsn` lives in the cold shard, `lsn >` lives in recall
    (I1). A consolidation advances it to the snapshot's `max(lsn)`; every other
    build (ingest/incremental/delete) carries the prior newest shard's value
    forward so the watermark stays monotonic — a non-consolidate fold only
    touches recall-owned rows (`lsn > watermark`) and must NOT regress it (a
    regression stalls the grace-trim and re-unions already-consolidated rows).
    The default `0` is correct only for a dataset's very first shard (no
    consolidated predecessor) and, with the flag off, for every shard. The value
    is set here at every build commit, never per recall write (the seam lives
    across two databases — I2's commit-then-trim keeps it safe).
    """
    uris = list(indexed_landing_uris or [])
    consolidated_lsn = int(consolidated_lsn or 0)
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
            "consolidated_lsn": consolidated_lsn,
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
  build_type, indexed_landing_uris, consolidated_lsn)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                tenant_id, dataset_name, shard_uri, checksum, vector_count,
                index_type, build_type, uris, consolidated_lsn,
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
    """Return shards for a `(tenant_id, dataset_name)` sorted newest-first.

    The ordering must be a TOTAL order: `_grace_watermark` selects `shards[1]`
    (the 2nd-newest) and `get_latest_shard` selects `shards[0]`, so a tie would
    make those selections nondeterministic and could mis-bound the grace-trim.
    Memory mode orders by the monotonic insertion `id` (already total). The
    Postgres path orders by `created_at DESC` but `created_at` is
    `TIMESTAMPTZ DEFAULT now()` (transaction-start time), so two shards built in
    the same transaction window could share it; the `id DESC` tiebreaker (the
    serial PK, strictly increasing) makes the order provably total and matches
    memory mode's `id`-desc semantics.
    """
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
ORDER BY created_at DESC, id DESC
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
