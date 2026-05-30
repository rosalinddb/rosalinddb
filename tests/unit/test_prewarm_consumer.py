"""`services/_common/prewarm_consumer.py` — PREWARM_SHARD queue consumer.

The PREWARM_SHARD consumer is a long-lived daemon thread that consumes
messages off the shipping queue and dispatches each to
`shard_tier.prewarm(shard_uri)`. On success the message is acked; on
`CacheCapacityExceeded` (or any other consumer-side exception) the
message is nacked so the queue's bounded retries handle the back-off
(eventually dead-lettering after `QUEUE_MAX_ATTEMPTS`).

Why these tests (in order):

  - subscribe / start_if_needed spawns exactly one worker thread; a second
    call is idempotent (the daemon's lifecycle mirrors catalog_listener).
  - On a delivered message the consumer calls `shard_tier.prewarm` with the
    `shard_uri` from the payload, then `ack`s the message — the standard
    happy path.
  - On `CacheCapacityExceeded` from `prewarm`, the consumer nacks (the
    queue's retry/DLQ scaffolding does the right thing from there) and the
    worker keeps running for the next message.
  - On a generic exception, the consumer also nacks and survives — one bad
    message must not kill the worker.
  - `stop()` is idempotent and joins quickly.
"""
from __future__ import annotations

import importlib
import threading
import time
from typing import List

import pytest


pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    """Poll `predicate` until True or `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def consumer_mod(monkeypatch, tmp_path):
    """Reload `services._common.prewarm_consumer` for clean per-test state.

    Module state (the worker thread, the stop event) must reset between
    tests so a sequencing bug in one does not bleed into the next.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", str(1024 * 1024))
    import services._common.prewarm_consumer as mod

    importlib.reload(mod)
    yield mod
    try:
        mod.stop()
    except Exception:  # noqa: BLE001 - test cleanup
        pass


class _FakeMessage(dict):
    """Stand-in for `adapters.queue.queue.Message`."""

    __slots__ = ("topic", "msg_id", "raw", "acked", "nacked")

    def __init__(self, payload: dict, topic: str = "PREWARM_SHARD"):
        super().__init__(payload)
        self.topic = topic
        self.msg_id = "fake-msg-id"
        self.raw = "fake-raw"
        self.acked = False
        self.nacked = False


def _install_fake_queue(monkeypatch, mod, messages: List[_FakeMessage]):
    """Replace consumer's `consume/ack/nack` with a fake that drains `messages`.

    `messages` is a list the test owns; the fake pops the next entry per
    `consume()` call and records ack/nack on the entry itself.
    """
    delivered: List[_FakeMessage] = []
    deliver_lock = threading.Lock()

    def _fake_consume(topic, block=True, timeout=None):  # noqa: ARG001
        with deliver_lock:
            if messages:
                m = messages.pop(0)
                delivered.append(m)
                return m
        # When the queue is empty, sleep briefly so the worker loop does
        # not spin against a real Queue.Empty equivalent.
        time.sleep(0.01)
        return None

    def _fake_ack(message):
        if message is not None:
            message.acked = True

    def _fake_nack(message, requeue=True):  # noqa: ARG001
        if message is not None:
            message.nacked = True
        return False

    monkeypatch.setattr(mod, "consume", _fake_consume)
    monkeypatch.setattr(mod, "ack", _fake_ack)
    monkeypatch.setattr(mod, "nack", _fake_nack)
    return delivered


# --- tests ----------------------------------------------------------------


def test_start_spawns_worker_and_is_idempotent(consumer_mod, monkeypatch):
    """Calling `start_if_needed` twice spawns exactly one worker thread.

    The lifecycle mirrors `catalog_listener._start_if_needed` — a second
    subscribe must observe the existing worker instead of doubling it.
    """
    _install_fake_queue(monkeypatch, consumer_mod, [])

    # Patch the dispatch target so a stray message cannot escape into a
    # real prewarm path during the lifecycle assertion.
    monkeypatch.setattr(consumer_mod, "_dispatch_one", lambda msg: None)

    consumer_mod.start_if_needed()
    assert _wait_until(consumer_mod._is_running), "worker thread did not start"

    consumer_mod.start_if_needed()  # idempotent — no second thread
    # Both calls should reference the same thread.
    assert consumer_mod._is_running()


def test_consumer_dispatches_to_prewarm_and_acks(consumer_mod, monkeypatch):
    """A delivered PREWARM_SHARD message calls `prewarm(uri)` and acks.

    Pins the happy path: the consumer extracts `shard_uri` from the
    payload, hands it to `shard_tier.prewarm`, and on success acks the
    message so the queue does not redeliver.
    """
    prewarm_calls: List[str] = []

    def _fake_prewarm(uri):
        prewarm_calls.append(uri)
        return "/tmp/fake/path"

    monkeypatch.setattr(consumer_mod.shard_tier, "prewarm", _fake_prewarm)

    msg = _FakeMessage({
        "tenant": "t1",
        "dataset": "ds1",
        "shard_uri": "memory://bucket/shard-prewarm.bin",
    })
    _install_fake_queue(monkeypatch, consumer_mod, [msg])

    consumer_mod.start_if_needed()
    assert _wait_until(lambda: msg.acked), (
        f"consumer did not ack within deadline; "
        f"prewarm_calls={prewarm_calls} acked={msg.acked}"
    )
    assert prewarm_calls == ["memory://bucket/shard-prewarm.bin"]
    assert msg.nacked is False


def test_consumer_nacks_on_cache_capacity_exceeded(consumer_mod, monkeypatch):
    """`CacheCapacityExceeded` from `prewarm` triggers a nack, not a crash.

    The queue's bounded-retry scaffolding handles back-off; eventually the
    message dead-letters after `QUEUE_MAX_ATTEMPTS`. The consumer must
    keep running after the nack — one rejected admission must not kill
    the worker.
    """
    def _fail_prewarm(uri):  # noqa: ARG001
        raise consumer_mod.shard_tier.CacheCapacityExceeded(
            "every candidate too young"
        )

    monkeypatch.setattr(consumer_mod.shard_tier, "prewarm", _fail_prewarm)

    msg = _FakeMessage({
        "tenant": "t1",
        "dataset": "ds1",
        "shard_uri": "memory://bucket/full-tier.bin",
    })
    _install_fake_queue(monkeypatch, consumer_mod, [msg])

    consumer_mod.start_if_needed()
    assert _wait_until(lambda: msg.nacked), (
        f"consumer did not nack CacheCapacityExceeded within deadline; "
        f"acked={msg.acked} nacked={msg.nacked}"
    )
    assert msg.acked is False
    # Worker must still be running after the rejection.
    assert consumer_mod._is_running()


def test_consumer_nacks_on_generic_exception(consumer_mod, monkeypatch):
    """A generic exception from `prewarm` is logged, nacked, and survived.

    Sibling of the CacheCapacityExceeded case. A storage outage that
    surfaces as `FileNotFoundError`, or any other unexpected exception,
    must not kill the daemon — the queue's retry contract handles it.
    """
    def _flaky_prewarm(uri):  # noqa: ARG001
        raise FileNotFoundError("transient storage failure")

    monkeypatch.setattr(consumer_mod.shard_tier, "prewarm", _flaky_prewarm)

    msg = _FakeMessage({
        "tenant": "t1",
        "dataset": "ds1",
        "shard_uri": "memory://bucket/missing.bin",
    })
    _install_fake_queue(monkeypatch, consumer_mod, [msg])

    consumer_mod.start_if_needed()
    assert _wait_until(lambda: msg.nacked)
    assert consumer_mod._is_running()


def test_consumer_skips_messages_without_shard_uri(consumer_mod, monkeypatch):
    """A malformed message (no `shard_uri`) is nacked, never calls prewarm.

    Defensive: a future producer that sends the wrong envelope shape must
    not silently re-trigger downloads — the consumer should treat it as a
    handler error and let the queue route it to the DLQ after retries.
    """
    prewarm_calls: List[str] = []

    def _spy_prewarm(uri):
        prewarm_calls.append(uri)
        return "/tmp/fake/path"

    monkeypatch.setattr(consumer_mod.shard_tier, "prewarm", _spy_prewarm)

    msg = _FakeMessage({"tenant": "t1", "dataset": "ds1"})  # no shard_uri
    _install_fake_queue(monkeypatch, consumer_mod, [msg])

    consumer_mod.start_if_needed()
    assert _wait_until(lambda: msg.nacked or msg.acked)
    assert prewarm_calls == [], (
        f"prewarm must not be called on a malformed message; "
        f"calls={prewarm_calls}"
    )


def test_stop_joins_worker_quickly(consumer_mod, monkeypatch):
    """`stop()` is idempotent and the worker exits within the join timeout.

    A stuck worker must not stall a shutdown handler — the daemon-thread
    pattern leaves it to die with the process if the join times out.
    """
    monkeypatch.setattr(consumer_mod, "_dispatch_one", lambda msg: None)
    _install_fake_queue(monkeypatch, consumer_mod, [])

    consumer_mod.start_if_needed()
    assert _wait_until(consumer_mod._is_running)

    consumer_mod.stop(join_timeout_s=2.0)
    assert _wait_until(lambda: not consumer_mod._is_running())

    # Idempotent — a second stop on a dead worker is a no-op.
    consumer_mod.stop(join_timeout_s=0.1)
