from __future__ import annotations

"""Postgres LISTEN/NOTIFY consumer for `catalog_updates`.

Activated when `RB_CATALOG_LISTEN=true`. Spawns a single long-lived daemon
thread that holds a dedicated psycopg2 connection (the LISTEN session
cannot share the request-time pool — a checked-in connection would lose
its subscriptions to the next borrower). On each notify, the thread
dispatches a parsed dict payload to every subscriber registered via
`subscribe(cb)`. The DP's per-`(tenant, dataset)` `list_shards` cache is
the only intended subscriber today; future readers (telemetry, audit)
can subscribe without growing this module.

Reliability contract:

  - Best-effort delivery (the LISTEN/NOTIFY contract from Postgres). A
    notify fired while the connection is dropped is LOST — the caller's
    TTL safety net (`RB_CATALOG_FRESHNESS_S` in the cache wrapper) is
    the correctness mechanism; this listener is the latency optimisation.
  - Reconnect on `psycopg2.OperationalError` (drop, server restart, idle
    eviction) with bounded exponential backoff to avoid a tight reconnect
    loop against a hard-down Postgres.
  - A raising subscriber callback is logged-and-skipped so one bad
    subscriber cannot starve the others.
  - `stop()` is idempotent and joins the worker with a short timeout —
    safe to call from a shutdown handler that ALSO closes other
    resources.

The default-off rollback contract is enforced by the CALLER (the DP):
this module's `subscribe()` does nothing dangerous on its own, but the
listener thread only starts inside `subscribe()`. A deployment that
never imports or never calls `subscribe()` runs with this file as dead
code — no connection opened, no thread spawned.
"""

import json
import logging
import select
import threading
import time
from typing import Callable, List, Optional

import psycopg2
import psycopg2.extensions

from adapters import config


_LOG = logging.getLogger(__name__)

# Channel name MUST match the `pg_notify(...)` call in
# `adapters/state/state.py:add_shard`. Hardcoded here rather than
# imported because state.py shouldn't depend on the listener.
_CHANNEL = "catalog_updates"

# Exponential backoff bounds. Initial doubles each failed reconnect up to
# the cap; counter resets on a successful LISTEN. The tests monkeypatch
# these to sub-second values to keep the reconnect-on-drop assertion
# fast — they intentionally live at module scope so a deployment can
# bump them via a small monkey-patch in a startup hook without forking
# the file.
_BACKOFF_INITIAL_S = 0.5
_BACKOFF_CAP_S = 30.0

# Wait-for-notify poll interval. `select(timeout=)` lets us wake on the
# stop event without busy-spinning. Bounded so a stop() never waits more
# than one interval after being signalled.
_SELECT_TIMEOUT_S = 1.0


# --- internal state -------------------------------------------------------


_SUBS_LOCK = threading.Lock()
_SUBS: List[Callable[[dict], None]] = []

_THREAD_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()


# --- swap points (patched by tests / overridable by callers) --------------


def _pg_connect(dsn: str):
    """Open a dedicated psycopg2 connection for the LISTEN session.

    Indirected through a module-level function so the listener tests can
    inject a fake connection without a real Postgres.
    """
    return psycopg2.connect(dsn)


def _select(rlist, wlist, xlist, timeout):
    """Thin wrapper around `select.select` for test injection.

    Tests intercept this to avoid blocking on a fake file descriptor;
    production uses the real `select.select`.
    """
    return select.select(rlist, wlist, xlist, timeout)


# --- public API -----------------------------------------------------------


def subscribe(callback: Callable[[dict], None]) -> Callable[[dict], None]:
    """Register `callback` for every parsed notify payload.

    Spawns the listener thread on the FIRST subscriber so a module that
    imports this file but never subscribes pays nothing. Returns the
    callback so a caller can pass it straight to `unsubscribe(token)`
    without keeping a separate handle.
    """
    with _SUBS_LOCK:
        _SUBS.append(callback)
    _start_if_needed()
    return callback


def unsubscribe(callback: Callable[[dict], None]) -> bool:
    """Remove `callback` from the subscriber list. Returns True if removed.

    Does NOT stop the listener thread when the subscriber list empties —
    the thread is cheap to leave running, and stop+restart on subscribe
    churn would create a window where a notify is lost. Operator-driven
    shutdown uses `stop()`.
    """
    with _SUBS_LOCK:
        try:
            _SUBS.remove(callback)
            return True
        except ValueError:
            return False


def stop(join_timeout_s: float = 2.0) -> None:
    """Signal the listener thread to exit and join with a short timeout.

    Idempotent: calling stop() with no live thread is a no-op. The join
    timeout exists so a stuck thread (e.g. wedged on a long network
    blip mid-select) cannot stall a shutdown indefinitely — a daemon
    thread is left to die with the process if the join times out.
    """
    _STOP_EVENT.set()
    with _THREAD_LOCK:
        t = _THREAD
    if t is not None and t.is_alive():
        t.join(timeout=join_timeout_s)


def _is_running() -> bool:
    """Test helper — returns whether the listener thread is alive."""
    with _THREAD_LOCK:
        t = _THREAD
    return bool(t and t.is_alive())


# --- internals ------------------------------------------------------------


def _dsn() -> str:
    """Read the DSN at start time so a flip-flopping env across pod
    restarts cleanly toggles the listener target."""
    return config.database_url_dsn()


def _start_if_needed() -> None:
    """Spawn the worker thread on the first subscriber. Idempotent."""
    global _THREAD
    with _THREAD_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        # Fresh start — clear any prior stop signal so a stop()/restart
        # cycle (e.g. tests with module reloads) works.
        _STOP_EVENT.clear()
        _THREAD = threading.Thread(
            target=_run, name="catalog-listener", daemon=True,
        )
        _THREAD.start()


def _dispatch(payload_str: str) -> None:
    """Parse + fan a single notify payload out to subscribers.

    Parse failures and subscriber failures are isolated: a malformed
    payload skips dispatch entirely (logged), and a raising subscriber
    is logged-and-skipped so the others still fire.
    """
    try:
        payload = json.loads(payload_str)
    except (ValueError, TypeError):
        _LOG.warning(
            "catalog_updates payload is not valid JSON; skipping: %r",
            payload_str,
        )
        return
    if not isinstance(payload, dict):
        _LOG.warning(
            "catalog_updates payload is not a JSON object; skipping: %r",
            payload,
        )
        return
    with _SUBS_LOCK:
        snapshot = list(_SUBS)
    for cb in snapshot:
        try:
            cb(payload)
        except Exception:  # noqa: BLE001 - isolate one bad subscriber
            _LOG.warning(
                "catalog_updates subscriber raised; continuing", exc_info=True,
            )


def _run() -> None:
    """Worker loop: connect, LISTEN, dispatch notifies, reconnect on drop.

    Outer loop owns the connection lifecycle (open + close). Inner loop
    owns the poll/dispatch cycle on a live connection. A failure in the
    inner loop falls back to the outer loop, which closes the dead
    connection and applies exponential backoff before retrying. Stop
    breaks both loops promptly via the `_STOP_EVENT`.
    """
    backoff = _BACKOFF_INITIAL_S
    while not _STOP_EVENT.is_set():
        conn = None
        try:
            conn = _pg_connect(_dsn())
            # AUTOCOMMIT is required for `LISTEN` to subscribe immediately
            # without an explicit `BEGIN; COMMIT`. The listener never
            # writes; the connection's only role is to receive notifies.
            conn.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
            )
            with conn.cursor() as cur:
                cur.execute(f"LISTEN {_CHANNEL}")
            _LOG.info("catalog listener: LISTEN %s established", _CHANNEL)
            # Successful subscribe — reset the backoff so the next drop
            # starts a fresh exponential ramp from the initial.
            backoff = _BACKOFF_INITIAL_S
            _poll_loop(conn)
        except psycopg2.OperationalError as exc:
            _LOG.warning(
                "catalog listener: connection error: %s; reconnect in %.1fs",
                exc, backoff,
            )
        except Exception:  # noqa: BLE001 - log and try to recover
            _LOG.exception(
                "catalog listener: unexpected error; reconnect in %.1fs",
                backoff,
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - best-effort close
                    pass
        # Bounded backoff. `wait(timeout)` returns True if `_STOP_EVENT`
        # is set during the wait, so a stop() during backoff exits
        # promptly instead of sleeping out the full interval.
        if _STOP_EVENT.wait(timeout=backoff):
            break
        backoff = min(backoff * 2.0, _BACKOFF_CAP_S)


def _poll_loop(conn) -> None:
    """Block on the connection's fd; dispatch each notify; surface drops.

    Raises `psycopg2.OperationalError` on a connection drop so the outer
    `_run` reconnect/backoff path engages. Returns cleanly when the stop
    event fires.
    """
    while not _STOP_EVENT.is_set():
        # `select` wakes on incoming data OR on the timeout — the
        # timeout is how we re-check `_STOP_EVENT` without polling.
        try:
            rlist, _, _ = _select([conn], [], [], _SELECT_TIMEOUT_S)
        except (OSError, ValueError):
            # `select` on a closed fd raises — treat as a drop.
            raise psycopg2.OperationalError("select on listener fd failed")
        if not rlist:
            continue
        # `poll()` consumes whatever the server sent and appends Notify
        # objects to `conn.notifies`. A drop raises OperationalError
        # here, which propagates to `_run` for reconnect.
        conn.poll()
        while conn.notifies:
            n = conn.notifies.pop(0)
            if n.channel != _CHANNEL:
                # Defensive: a future module on the same connection
                # might LISTEN to another channel. Skip non-target
                # notifies so we never dispatch them to subscribers
                # who expect catalog payloads.
                continue
            _dispatch(n.payload)
