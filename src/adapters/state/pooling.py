from __future__ import annotations

"""Connection pooling + request-scoped connection lifecycle for the state adapter.

Extracted from `adapters.state.state` (behaviour-preserving). This module holds:

  * the dedicated, non-pooled `_conn()` factory (used by the migration runner and
    `dataset_build_lock` — callers with a connection-scope dependency the pool
    cannot honour);
  * the process-wide control-plane `ThreadedConnectionPool` factory
    (`_get_pool` / `pooled_conn`) — ONE pool per process, built lazily;
  * the block-with-timeout checkout helper (`_getconn_with_timeout`) and the two
    knobs that drive it (`_pool_max_size`, `_pool_checkout_timeout_s`); and
  * the request-scoped connection lifecycle (`_REQUEST_CONN`, `bind_request_conn`,
    `unbind_request_conn`, `finish_request_conn`, `_checkout_request_conn`,
    `request_scoped_connection`) driven by `adapters/state/conn_middleware.py`.

Mutable process-wide state — `_MEMORY_MODE`, `_POOL`, `_POOL_LOCK`, the
`_REQUEST_CONN` contextvar — is OWNED by `adapters.state.state` and accessed here
through the `_state` reference at CALL time (never at import time), so that:

  * `monkeypatch.setattr(state, "_MEMORY_MODE", False)` / `..., "_POOL", fake)`
    is observed by these functions verbatim (the test suite relies on this); and
  * `importlib.reload(state)` (which recreates those globals fresh) is observed
    too — the `state` module object identity is stable across a reload, so a
    held `_state` reference always sees the current globals.

This is the same seam `conn_middleware.py` already uses; see its module docstring.
The `PoolCheckoutTimeout` raised here is the SINGLE class from `adapters.errors`,
so every `except PoolCheckoutTimeout` frame (here and in callers) keeps matching.
"""

import contextlib
import contextvars
import time
from typing import Iterator, Optional

import psycopg2
import psycopg2.pool

from adapters import config
from adapters.errors import PoolCheckoutTimeout
from adapters.observability.tracing import state_connect_span

# The state module owns the mutable process-wide globals (`_MEMORY_MODE`,
# `_POOL`, `_POOL_LOCK`, `_REQUEST_CONN`, …). Reference them through `_state.X`
# at call time so monkeypatches and `importlib.reload(state)` are both honoured.
# Imported here, but every access is deferred to call time (no import-time use),
# so the partial-init of `state` during its own import of this module is safe.
from adapters.state._lazy_state import state as _state  # lazy proxy: resolves the facade at call time (breaks the import cycle)


_DEFAULT_PG_POOL_MAX = 10  # per-process ceiling; override with RB_PG_POOL_MAX
_PG_POOL_MIN = 1  # connections kept warm even when idle

_DEFAULT_POOL_CHECKOUT_TIMEOUT_S = 2.5  # total deadline; RB_PG_POOL_CHECKOUT_TIMEOUT_S
_POOL_CHECKOUT_POLL_S = 0.015  # sleep between fail-fast getconn() retries (~15ms)


def _dsn() -> str:
    """Return the Postgres DSN from the environment (Postgres mode only)."""
    return config.database_url_dsn()


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
    if _state._MEMORY_MODE:
        raise RuntimeError("memory state has no SQL connection")
    # Unannotated `state.connect` span — the legacy dedicated/unpooled path. A
    # genuine connect every time; the annotated reuse/open variants are emitted
    # by `pooled_conn()`.
    with state_connect_span():
        return psycopg2.connect(_dsn())


def _pool_max_size() -> int:
    """Return the per-process pool ceiling.

    Defaults to `_DEFAULT_PG_POOL_MAX` (10). `RB_PG_POOL_MAX`, when set to a
    positive integer, overrides it — useful to fit the pool under a managed
    database's connection cap when many worker processes run side by side. A
    missing or malformed value falls back to the default rather than crashing.
    """
    return config.pg_pool_max()


def _pool_checkout_timeout_s() -> float:
    """Return the total block-with-timeout deadline for a pool checkout.

    Defaults to `_DEFAULT_POOL_CHECKOUT_TIMEOUT_S` (2.5s). `RB_PG_POOL_CHECKOUT_TIMEOUT_S`,
    when set to a positive number, overrides it — so an operator can tune how
    long a request waits for a free connection before the pool checkout gives
    up with a 503. Read live (per call) so a test can retune it without a
    module reload. A missing or malformed value falls back to the default.
    """
    return config.pg_pool_checkout_timeout_s()


def _get_pool(maxconn_override: Optional[int] = None) -> psycopg2.pool.ThreadedConnectionPool:
    """Return the process-wide connection pool, building it lazily on first use.

    `ThreadedConnectionPool` (not `SimpleConnectionPool`) because the services
    run threaded under uvicorn — checkout/return must be thread-safe.

    `maxconn_override` forces the pool's max size; it is a test hook (a tiny
    max-1 pool makes a connection leak immediately fatal). In production the
    size comes from `_pool_max_size()`.
    """
    if _state._MEMORY_MODE:
        raise RuntimeError("memory state has no SQL connection pool")
    if _state._POOL is None:
        with _state._POOL_LOCK:
            # Re-check under the lock — another thread may have built it while
            # this one waited.
            if _state._POOL is None:
                maxconn = maxconn_override or _pool_max_size()
                _state._POOL = psycopg2.pool.ThreadedConnectionPool(
                    minconn=_PG_POOL_MIN,
                    maxconn=max(_PG_POOL_MIN, maxconn),
                    dsn=_dsn(),
                )
    return _state._POOL


def _close_pool() -> None:
    """Close every connection in the pool and discard it.

    Mainly a test teardown hook so a test that rebinds `DATABASE_URL` does not
    leave a pool pinned to a stopped container. Safe to call when no pool
    exists. Production processes are long-lived and let the pool live for the
    process lifetime.
    """
    with _state._POOL_LOCK:
        if _state._POOL is not None:
            _state._POOL.closeall()
            _state._POOL = None


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
    bound = None if standalone else _state._REQUEST_CONN.get()
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
        # Return the connection on EVERY exit — even if commit() itself raises
        # (broken backend / TLS reset / statement_timeout on COMMIT). Otherwise a
        # commit failure leaks the conn from the pool's accounting, permanently
        # shrinking the pool. psycopg2._putconn inspects transaction_status and
        # rolls back / closes a non-idle or server-lost conn before re-pooling,
        # so a post-commit-failure conn is returned clean, not poisoned. (Mirrors
        # the corrected pattern in `recall_pooled_conn()`.)
        try:
            conn.commit()
        finally:
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
    if _state._MEMORY_MODE:
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
    return _state._REQUEST_CONN.set(conn)


def unbind_request_conn(token: "contextvars.Token") -> None:
    """Reset `_REQUEST_CONN` to its pre-request value.

    Must run in the SAME context `bind_request_conn` ran in — the ASGI
    middleware calls both in its coroutine. A no-op-safe pair: if `set`
    bound `None` (memory mode) this still cleanly unwinds.
    """
    _state._REQUEST_CONN.reset(token)


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
    if _state._MEMORY_MODE:
        # No SQL pool in memory mode — nothing to bind. The middleware calls
        # this unconditionally; here it is a pure no-op.
        yield
        return

    pool = _get_pool(maxconn_override=maxconn_override)
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        conn = _getconn_with_timeout(pool, _pool_checkout_timeout_s())
    token = _state._REQUEST_CONN.set(conn)
    try:
        yield
    except BaseException:
        try:
            conn.rollback()
        finally:
            _state._REQUEST_CONN.reset(token)
            pool.putconn(conn)
        raise
    else:
        try:
            conn.commit()
        finally:
            _state._REQUEST_CONN.reset(token)
            pool.putconn(conn)
