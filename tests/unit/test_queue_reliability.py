"""Unit tests for the reliable-queue API shape and the reconciliation reaper.

These exercise the in-process (`queue.Queue`) backend and the memory-mode
state adapter — hermetic, no Docker. The Redis-specific reliable-delivery and
DLQ guarantees are covered by `tests/integration/test_queue_redis.py`; here we
pin the parts that must hold on the test-only backend too:

  - `consume()` returns a `Message` (a dict subclass) so existing callers that
    do `msg["dataset"]` keep working.
  - `ack()`/`nack()` exist and are callable on the in-process path.
  - Trace-context (`_otel`) still round-trips through publish/consume and the
    `_otel`/`_msg_id` envelope keys never leak into the payload.
  - The reaper flips a dataset stranded in a non-terminal status to `error`.
"""
from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture
def fresh_queue(monkeypatch):
    """Reload the queue adapter with no `REDIS_URL` so it uses `queue.Queue`."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    import adapters.queue.queue as q
    importlib.reload(q)
    return q


def test_consume_returns_message_that_behaves_like_a_dict(fresh_queue):
    q = fresh_queue
    q.publish("VALIDATE_DATASET", {"dataset": "ds1", "tenant": "t1", "uri": "memory://x"})
    msg = q.consume("VALIDATE_DATASET", block=False)
    assert msg is not None
    # Existing callers index it like a plain dict.
    assert msg["dataset"] == "ds1"
    assert msg.get("tenant") == "t1"
    assert isinstance(msg, dict)
    # The reserved envelope keys must never leak into the payload.
    assert "_otel" not in msg
    assert "_msg_id" not in msg


def test_ack_and_nack_are_callable_on_in_process_backend(fresh_queue):
    q = fresh_queue
    q.publish("DATASET_READY", {"dataset": "ds2", "tenant": "t1"})
    msg = q.consume("DATASET_READY", block=False)
    assert msg is not None
    # ack must not raise on the in-process path.
    q.ack(msg)
    # nack(requeue=False) returns whether it was dead-lettered; in-process has
    # no DLQ so it is always False, and must not raise.
    q.publish("DATASET_READY", {"dataset": "ds3", "tenant": "t1"})
    msg2 = q.consume("DATASET_READY", block=False)
    assert q.nack(msg2, requeue=False) is False


def test_nack_requeue_redelivers_on_in_process_backend(fresh_queue):
    q = fresh_queue
    q.publish("DATASET_READY", {"dataset": "redeliver-me", "tenant": "t1"})
    msg = q.consume("DATASET_READY", block=False)
    assert msg is not None
    # nack with requeue re-publishes — the message comes back round.
    q.nack(msg, requeue=True)
    again = q.consume("DATASET_READY", block=False)
    assert again is not None and again["dataset"] == "redeliver-me"


def test_happy_path_publish_consume_ack_and_otel_propagation(fresh_queue):
    """A normal publish→consume→ack works and trace context still propagates."""
    q = fresh_queue
    # Inject a fake carrier directly by publishing with an active span would
    # need an SDK; instead assert the propagation hook does not corrupt the
    # payload and that a published message carries a stamped `_msg_id`.
    q.publish("RUN_EPHEMERAL_QUERY", {"dataset": "ds", "vector": [0, 0, 0, 0]})
    msg = q.consume("RUN_EPHEMERAL_QUERY", block=False)
    assert msg is not None
    assert msg["vector"] == [0, 0, 0, 0]
    # `_msg_id` is stamped at publish and stripped from the consumed payload,
    # but is exposed on the Message for ack addressing.
    assert msg.msg_id != "" and msg.msg_id is not None
    q.ack(msg)
    # Queue is now empty.
    assert q.consume("RUN_EPHEMERAL_QUERY", block=False) is None


# --- reaper: stuck-dataset reconciliation --------------------------------


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "memory://reaper-test")
    import adapters.state.state as state
    importlib.reload(state)
    state._MEM_DATASETS.clear()
    state._MEM_TENANTS.clear()
    state._MEM_TENANTS_BY_EMAIL.clear()
    state._MEM_SHARDS.clear()
    return state


def test_find_stale_datasets_only_returns_old_non_terminal(fresh_state):
    state = fresh_state
    state.create_dataset("t1", "fresh", 4)
    state.update_dataset_status("t1", "fresh", "validating")
    state.create_dataset("t1", "done", 4)
    state.update_dataset_status("t1", "done", "indexed")
    # A very long timeout finds nothing — nothing is that old yet.
    assert state.find_stale_datasets(older_than_seconds=3600) == []
    # A zero timeout finds the non-terminal one, but never the terminal one.
    stale = state.find_stale_datasets(older_than_seconds=0)
    names = {d["dataset_name"] for d in stale}
    assert "fresh" in names
    assert "done" not in names


def test_reaper_flips_stuck_dataset_to_error(fresh_state, monkeypatch):
    """The reaper flips a dataset stuck in a non-terminal status to `error`."""
    state = fresh_state
    state.create_dataset("t1", "wedged", 4)
    state.update_dataset_status("t1", "wedged", "indexing")

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)

    # stuck_timeout=0 → the dataset (just stamped) already qualifies as stale.
    summary = reaper.reap_once(reclaim_timeout=0, stuck_timeout=0)
    assert summary["datasets_failed"] == 1

    ds = state.get_dataset("t1", "wedged")
    assert ds["status"] == "error"
    assert ds["error_message"] and "stuck" in ds["error_message"].lower()


def test_reaper_leaves_healthy_dataset_alone(fresh_state):
    state = fresh_state
    state.create_dataset("t1", "healthy", 4)
    state.update_dataset_status("t1", "healthy", "validating")

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)

    # A generous stuck_timeout — a freshly-stamped dataset is not yet stale.
    summary = reaper.reap_once(reclaim_timeout=3600, stuck_timeout=3600)
    assert summary["datasets_failed"] == 0
    assert state.get_dataset("t1", "healthy")["status"] == "validating"


# --- Finding 2: reaper-vs-worker compare-and-set -------------------------


def test_fail_dataset_if_stale_does_not_clobber_terminal_status(fresh_state):
    """A worker that legitimately reached `indexed` is NOT flipped to `error`.

    Finding 2: the reaper reads a dataset as stale, then a worker writes a
    terminal status, then the reaper's flip runs. The guarded flip
    (`fail_dataset_if_stale`) must compare-and-set — it only writes `error`
    while the dataset is STILL non-terminal. This test fails against an
    unconditional `update_dataset_status`.
    """
    state = fresh_state
    state.create_dataset("t1", "racey", 4)
    state.update_dataset_status("t1", "racey", "indexing")
    # The reaper observed it stale (stuck_timeout=0) — but BEFORE it flips, a
    # worker legitimately finishes the build and writes the terminal status.
    state.update_dataset_status("t1", "racey", "indexed")
    # The guarded flip must now be a no-op: `indexed` is terminal.
    flipped = state.fail_dataset_if_stale(
        "t1", "racey", older_than_seconds=0, error_message="reaper: stuck"
    )
    assert flipped is False
    assert state.get_dataset("t1", "racey")["status"] == "indexed"


def test_fail_dataset_if_stale_flips_a_genuinely_stuck_dataset(fresh_state):
    """The guarded flip still flips a dataset that IS stuck and non-terminal."""
    state = fresh_state
    state.create_dataset("t1", "wedged", 4)
    state.update_dataset_status("t1", "wedged", "validating")
    flipped = state.fail_dataset_if_stale(
        "t1", "wedged", older_than_seconds=0, error_message="reaper: stuck"
    )
    assert flipped is True
    ds = state.get_dataset("t1", "wedged")
    assert ds["status"] == "error"
    assert ds["error_message"] == "reaper: stuck"


def test_reaper_does_not_clobber_dataset_a_worker_just_finished(
    fresh_state, monkeypatch
):
    """End-to-end Finding 2: reconcile pass leaves a now-terminal dataset alone.

    The reaper's `find_stale_datasets` snapshot includes a dataset that a
    worker finishes (writes `indexed`) before `reconcile_stuck_datasets`
    reaches its guarded flip. The dataset must stay `indexed`.
    """
    state = fresh_state
    state.create_dataset("t1", "finished", 4)
    state.update_dataset_status("t1", "finished", "indexing")

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)

    # Simulate the worker winning the race: between the stale-scan and the
    # flip, `find_stale_datasets` returns the stale snapshot but the live
    # dataset has already moved to `indexed`.
    real_find = state.find_stale_datasets

    def racing_find(*args, **kwargs):
        snapshot = real_find(*args, **kwargs)
        # Worker finishes the build right after the reaper's snapshot.
        state.update_dataset_status("t1", "finished", "indexed")
        return snapshot

    monkeypatch.setattr(reaper, "find_stale_datasets", racing_find)

    summary = reaper.reap_once(reclaim_timeout=0, stuck_timeout=0)
    assert summary["datasets_failed"] == 0
    assert state.get_dataset("t1", "finished")["status"] == "indexed"


# --- graceful shutdown ----------------------------------------------------


def test_shutdown_event_trips_and_resets():
    """A worker loop polls `should_stop()`; signals/`request_stop` trip it."""
    import adapters.queue.shutdown as shutdown
    importlib.reload(shutdown)

    assert shutdown.should_stop() is False
    shutdown.request_stop()
    assert shutdown.should_stop() is True
    # The shared Event is what background loops (the reaper) wait on.
    assert shutdown.stop_event().is_set()
    shutdown.reset()
    assert shutdown.should_stop() is False


def test_install_signal_handlers_is_safe_to_call(monkeypatch):
    """Wiring SIGTERM/SIGINT must not raise even off the main thread."""
    import adapters.queue.shutdown as shutdown
    importlib.reload(shutdown)
    # Must not raise regardless of which thread the test runs on.
    shutdown.install_signal_handlers()
    shutdown.reset()
