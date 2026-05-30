"""`services/_common/catalog_listener.py` — LISTEN/NOTIFY catalog consumer.

Pins the subscribe / unsubscribe / reconnect / clean-shutdown contracts of
the Postgres LISTEN/NOTIFY consumer in isolation. No real Postgres: a fake
connection records `set_isolation_level`, `cursor().execute(...)` LISTEN
statements, and a queue of `Notify` events that `poll()` drains into
`notifies`. This is the standard psycopg2 LISTEN/NOTIFY pattern documented
upstream (`connection.notifies` is appended to by `connection.poll()`).

Why these tests, in this order:

  - Subscribe -> first subscriber spawns the listener thread; the fake
    connection's LISTEN was executed; a fed Notify is delivered to the
    subscribed callback. This is the happy path the DP relies on.
  - Multiple subscribers all receive the same notification (the catalog
    cache lives in two services that both subscribe in the same process
    when colocated under the dev harness).
  - Unsubscribe removes the callback; further notifies do not invoke it.
  - Reconnect on connection drop: the listener retries with bounded
    exponential backoff; a new fake connection is opened, LISTEN is
    re-executed, post-reconnect notifies are delivered. The bound is
    important — an unbounded retry loop would silently busy-loop a process
    on an offline Postgres.
  - `stop()` joins the worker thread quickly; subsequent notifies are
    NOT delivered (the listener owns no other resources after stop).
"""
from __future__ import annotations

import importlib
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit


# --- fake psycopg2 connection ---------------------------------------------


class _FakeNotify:
    """Mimics `psycopg2.extensions.Notify`: `.channel`, `.payload`."""

    def __init__(self, channel: str, payload: str):
        self.channel = channel
        self.payload = payload


class _FakeCursor:
    def __init__(self, conn: "_FakeConn"):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, *args, **kwargs):
        self._conn.executed.append(sql)

    def close(self):
        pass


class _FakeConn:
    """Records the LISTEN SQL and serves queued Notify events via `poll()`.

    `poll()` drains `_pending` into `notifies` and optionally raises a
    pre-armed exception so reconnect-on-drop can be exercised. `closed`
    flips when `close()` runs so the listener can detect a dead connection.
    """

    def __init__(self):
        self.executed: List[str] = []
        self.notifies: List[_FakeNotify] = []
        self._pending: List[_FakeNotify] = []
        self._poll_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self.closed = 0
        self.isolation_level_set: Optional[int] = None
        # `fileno()` is used by `select.select`; any small int works as long
        # as it does not clash with stdin/stdout/stderr. The listener under
        # test uses a select-with-timeout that the fake intercepts.
        self._fd = -1  # sentinel: select() is monkey-patched in tests

    def set_isolation_level(self, level: int) -> None:
        self.isolation_level_set = level

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def poll(self) -> None:
        with self._lock:
            if self._poll_error is not None:
                err = self._poll_error
                self._poll_error = None
                raise err
            self.notifies.extend(self._pending)
            self._pending.clear()

    def fileno(self) -> int:
        return self._fd

    def close(self) -> None:
        self.closed += 1

    # Test helpers ---------------------------------------------------------

    def feed(self, channel: str, payload: str) -> None:
        with self._lock:
            self._pending.append(_FakeNotify(channel, payload))

    def arm_poll_error(self, exc: BaseException) -> None:
        with self._lock:
            self._poll_error = exc


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def listener_mod(monkeypatch):
    """Reload `services._common.catalog_listener` for a clean per-test state.

    Module-level state (the singleton listener, subscriber list, backoff
    counter) must be reset between tests so a sequencing bug in one does
    not leak into the next.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub/forced-pg-mode")
    import services._common.catalog_listener as mod

    importlib.reload(mod)
    yield mod
    # Best-effort shutdown — a test that did not stop its own listener must
    # not leak a daemon thread into the next test.
    try:
        mod.stop()
    except Exception:  # noqa: BLE001 - test cleanup
        pass


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    """Poll `predicate` until True or `timeout_s` elapses.

    Used in place of `sleep` so a fast box does not pad the test runtime
    and a slow box does not false-fail. Returns whether the predicate was
    satisfied within the deadline.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def _install_fake_psycopg(monkeypatch, mod, conns: list[_FakeConn]):
    """Replace `psycopg2.connect` so the listener gets `_FakeConn`s.

    `conns` is a list the test owns; the listener pops the next fake on
    each connect (re-connect path uses successive entries). The `select`
    used inside the listener loop is also stubbed to a tiny sleep so the
    test does not block on a real fd.
    """
    iterator = iter(conns)

    def _fake_connect(dsn):  # noqa: ARG001 - match psycopg2.connect signature
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError("test ran out of fake connections") from exc

    monkeypatch.setattr(mod, "_pg_connect", _fake_connect)

    def _fake_select(rlist, wlist, xlist, timeout):  # noqa: ARG001
        # Brief sleep stands in for a real wait on the connection fd. Long
        # enough that the listener loop does not spin; short enough that
        # the test's `_wait_until` resolves quickly.
        time.sleep(0.01)
        return ([rlist[0]] if rlist else [], [], [])

    monkeypatch.setattr(mod, "_select", _fake_select)


# --- tests ----------------------------------------------------------------


def test_subscribe_delivers_notify_to_callback(listener_mod, monkeypatch):
    """A subscribed callback receives the channel + payload on notify.

    Pins the happy path: subscribe -> listener thread starts -> the
    LISTEN statement runs against the (fake) connection -> a fed Notify
    is delivered to the registered callback.
    """
    conn = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn])

    received: list[tuple[str, str]] = []

    def cb(payload: dict) -> None:
        # Listener parses the JSON payload before dispatching so a
        # subscriber gets a dict (not a raw string) — that is the
        # contract under test.
        received.append((payload.get("tenant", ""), payload.get("dataset", "")))

    listener_mod.subscribe(cb)
    # The listener must execute `LISTEN catalog_updates` against its
    # dedicated connection before delivering notifies.
    assert _wait_until(lambda: any("LISTEN" in s for s in conn.executed)), (
        "listener did not issue LISTEN within the deadline"
    )
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "ds"}')
    assert _wait_until(lambda: received == [("t1", "ds")]), (
        f"callback did not observe the notify in time; received={received}"
    )


def test_unsubscribe_stops_callback_delivery(listener_mod, monkeypatch):
    """An unsubscribed callback does NOT receive subsequent notifies.

    The cache-invalidation use case requires precise lifetime control: a
    DP that swaps its listener subscription (e.g. on a config reload)
    must not keep firing the old callback against the new cache.
    """
    conn = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn])

    seen: list[dict] = []

    def cb(payload: dict) -> None:
        seen.append(payload)

    token = listener_mod.subscribe(cb)
    assert _wait_until(lambda: any("LISTEN" in s for s in conn.executed))
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "ds"}')
    assert _wait_until(lambda: len(seen) == 1)

    listener_mod.unsubscribe(token)
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "after-unsub"}')
    # A brief wait so a buggy implementation that still dispatches has
    # time to do so; the assertion is that the count did NOT grow.
    time.sleep(0.1)
    assert len(seen) == 1, f"unsubscribed callback still fired; seen={seen}"


def test_multiple_subscribers_all_receive(listener_mod, monkeypatch):
    """Two subscribers in one process both observe every notify.

    The query_api and ephemeral_runner can both subscribe in the dev
    harness (they share a process); the listener must not deliver only
    to one of them.
    """
    conn = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn])

    a: list[dict] = []
    b: list[dict] = []
    listener_mod.subscribe(a.append)
    listener_mod.subscribe(b.append)

    assert _wait_until(lambda: any("LISTEN" in s for s in conn.executed))
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "ds"}')
    assert _wait_until(lambda: len(a) == 1 and len(b) == 1)


def test_reconnect_on_connection_drop(listener_mod, monkeypatch):
    """A `psycopg2.OperationalError` from `poll()` triggers a reconnect.

    A real-world LISTEN connection drops on network blips, PG restarts,
    and idle eviction. The listener MUST re-`connect()` and re-`LISTEN`
    on its own — silently going dark would leave the per-dataset cache
    relying entirely on the TTL safety net.
    """
    conn1 = _FakeConn()
    conn2 = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn1, conn2])

    # Tiny backoff so the test does not wait the default exponential
    # ramp; we just need to prove the loop reconnects.
    monkeypatch.setattr(listener_mod, "_BACKOFF_INITIAL_S", 0.01)
    monkeypatch.setattr(listener_mod, "_BACKOFF_CAP_S", 0.05)

    seen: list[dict] = []
    listener_mod.subscribe(seen.append)
    assert _wait_until(lambda: any("LISTEN" in s for s in conn1.executed))

    # Arm conn1.poll() to raise on the next loop iteration — the listener
    # must catch, close, sleep, and reconnect (which gets conn2).
    import psycopg2

    conn1.arm_poll_error(psycopg2.OperationalError("server closed the connection"))
    assert _wait_until(lambda: any("LISTEN" in s for s in conn2.executed), 3.0), (
        "listener did not re-LISTEN against the replacement connection"
    )
    assert conn1.closed >= 1, "dropped connection was not closed"

    # Post-reconnect notifies delivered via the NEW connection.
    conn2.feed("catalog_updates", '{"tenant": "t1", "dataset": "after-reconnect"}')
    assert _wait_until(
        lambda: any(p.get("dataset") == "after-reconnect" for p in seen)
    ), f"post-reconnect notify was not delivered; seen={seen}"


def test_stop_terminates_listener_thread(listener_mod, monkeypatch):
    """`stop()` joins the worker quickly and stops delivering notifies.

    Clean shutdown matters: a service restart that leaks a daemon thread
    blocks Python's exit on the next deploy cycle. The listener must
    exit its loop on `stop()`, close the connection, and the thread
    must not be alive after a brief join.
    """
    conn = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn])

    seen: list[dict] = []
    listener_mod.subscribe(seen.append)
    assert _wait_until(lambda: any("LISTEN" in s for s in conn.executed))

    listener_mod.stop()
    # The thread should have exited; a slow box gets a 2 s grace.
    assert _wait_until(
        lambda: not listener_mod._is_running(), 2.0
    ), "listener thread did not exit after stop()"
    # A post-stop feed is moot but exercised for the contract: nothing
    # is delivered because the loop is gone.
    before = len(seen)
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "post-stop"}')
    time.sleep(0.05)
    assert len(seen) == before, "callback fired after stop()"


def test_malformed_payload_does_not_kill_listener(listener_mod, monkeypatch):
    """A bad JSON payload is logged-and-skipped; the listener stays up.

    NOTIFY payloads come from the database; a future producer (operator
    SQL, accidental cross-schema collision on the channel name) could
    send a non-JSON string. The listener MUST NOT crash the worker
    thread on a parse failure — otherwise one rogue payload halts all
    cache invalidation.
    """
    conn = _FakeConn()
    _install_fake_psycopg(monkeypatch, listener_mod, [conn])

    seen: list[dict] = []
    listener_mod.subscribe(seen.append)
    assert _wait_until(lambda: any("LISTEN" in s for s in conn.executed))

    conn.feed("catalog_updates", "not-json")
    # Good payload follows: the listener must still deliver it.
    conn.feed("catalog_updates", '{"tenant": "t1", "dataset": "ok"}')
    assert _wait_until(lambda: any(p.get("dataset") == "ok" for p in seen)), (
        f"listener stopped delivering after a malformed payload; seen={seen}"
    )
