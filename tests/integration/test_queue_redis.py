"""Integration tests for the reliable-queue Redis backend.

These spin up a real Redis container (testcontainers) and exercise the
deployment-resilience guarantees that only exist on the Redis path:

  - **At-least-once delivery.** A worker that consumes a message and dies
    without acking (the consumer is simply dropped) leaves the message
    reclaimable; the next consumer — after the reaper reclaims it — gets it.
  - **Max-retry → dead-letter.** A poison message `nack`-ed past
    `QUEUE_MAX_ATTEMPTS` lands in the dead-letter list.
  - **Reaper reclaim.** The reaper moves a message stuck on the processing
    list past the reclaim timeout back for redelivery.
  - **Happy path.** A normal publish→consume→ack removes the message for good
    and leaves the processing list empty; trace-context still propagates.

The queue adapter binds `REDIS_URL` at import time, so each test reloads
`adapters.queue.queue` with the container's URL set, then `flushdb`s for
isolation.
"""
from __future__ import annotations

import importlib

import pytest

try:
    from testcontainers.redis import RedisContainer
except ImportError as exc:  # pragma: no cover
    RedisContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def redis_url():
    """Start one Redis container for this module; yield its URL."""
    if RedisContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers[redis] is required for the Redis queue suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with RedisContainer("redis:7-alpine") as rc:
        host = rc.get_container_host_ip()
        port = rc.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture
def q(redis_url, monkeypatch):
    """Reload the queue adapter bound to the container Redis; flush per test.

    Teardown reloads the adapter back to its default in-process backend so a
    subsequent test in the session (which expects `queue.Queue`) is not left
    talking to a Redis the queue module bound at import time.
    """
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("QUEUE_MAX_ATTEMPTS", "3")
    import adapters.queue.queue as queue_mod
    importlib.reload(queue_mod)
    queue_mod._redis.flushdb()
    yield queue_mod
    queue_mod._redis.flushdb()
    # Restore the in-process backend for the rest of the session.
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("QUEUE_MAX_ATTEMPTS", raising=False)
    importlib.reload(queue_mod)


def test_happy_path_publish_consume_ack(q):
    """publish → consume → ack removes the message; processing list ends empty."""
    q.publish("VALIDATE_DATASET", {"dataset": "ds", "tenant": "t1", "uri": "x"})
    msg = q.consume("VALIDATE_DATASET", block=False)
    assert msg is not None and msg["dataset"] == "ds"
    # While unacked the message sits on the processing list.
    assert q.processing_size("VALIDATE_DATASET") == 1
    q.ack(msg)
    # Acked: gone from both the main list and the processing list.
    assert q.processing_size("VALIDATE_DATASET") == 0
    assert q.consume("VALIDATE_DATASET", block=False) is None
    assert q.dlq_size("VALIDATE_DATASET") == 0


def test_otel_envelope_keys_do_not_leak_into_payload(q):
    """Trace-context plumbing must not surface `_otel`/`_msg_id` to consumers."""
    q.publish("DATASET_READY", {"dataset": "ds", "tenant": "t1"})
    msg = q.consume("DATASET_READY", block=False)
    assert msg is not None
    assert "_otel" not in msg and "_msg_id" not in msg
    # The id is still available on the Message for ack addressing.
    assert msg.msg_id
    q.ack(msg)


def test_worker_dies_mid_job_message_is_redelivered(q):
    """A consumer that takes a message and dies without acking → redelivered.

    Simulates a deploy/restart killing a worker mid-job: consume the message,
    do NOT ack, drop the consumer reference. The message is stuck on the
    processing list; the reaper reclaims it and the next consumer gets it and
    completes the job.
    """
    q.publish("VALIDATE_DATASET", {"dataset": "survives", "tenant": "t1", "uri": "x"})

    # Worker 1 consumes, then "dies" — no ack.
    dead = q.consume("VALIDATE_DATASET", block=False)
    assert dead is not None and dead["dataset"] == "survives"
    del dead
    # The message is NOT lost — it is parked on the processing list.
    assert q.processing_size("VALIDATE_DATASET") == 1
    # Nothing is on the main list, so a fresh consume sees nothing yet.
    assert q.consume("VALIDATE_DATASET", block=False) is None

    # The reaper reclaims anything stuck in-processing past the timeout.
    import adapters.queue.reaper as reaper
    importlib.reload(reaper)
    reclaimed = reaper.reclaim_stuck_messages(reclaim_timeout=0)
    assert reclaimed == 1

    # Worker 2 now gets the redelivered message and completes the job.
    msg = q.consume("VALIDATE_DATASET", block=False)
    assert msg is not None and msg["dataset"] == "survives"
    q.ack(msg)
    assert q.processing_size("VALIDATE_DATASET") == 0
    assert q.dlq_size("VALIDATE_DATASET") == 0


def test_poison_message_dead_letters_after_max_attempts(q):
    """A message nack-ed past QUEUE_MAX_ATTEMPTS lands in the dead-letter list."""
    q.publish("VALIDATE_DATASET", {"dataset": "poison", "tenant": "t1", "uri": "x"})

    # QUEUE_MAX_ATTEMPTS is pinned to 3 by the fixture. Each consume bumps the
    # attempt counter; nack(requeue=True) re-queues until the cap is crossed.
    dead_lettered = False
    for _ in range(10):
        msg = q.consume("VALIDATE_DATASET", block=False)
        if msg is None:
            break
        dead_lettered = q.nack(msg, requeue=True)
        if dead_lettered:
            break
    assert dead_lettered, "poison message should have been dead-lettered"
    assert q.dlq_size("VALIDATE_DATASET") == 1
    assert q.consume("VALIDATE_DATASET", block=False) is None
    payloads = q.peek_dlq("VALIDATE_DATASET")
    assert payloads and payloads[0]["dataset"] == "poison"


def test_reaper_dead_letters_stuck_message_over_attempt_cap(q):
    """A stuck processing-list message already over the cap is dead-lettered.

    The reaper does not blindly redeliver: a message that has been delivered
    QUEUE_MAX_ATTEMPTS times and is THEN found stuck in-processing is routed to
    the DLQ instead of looping forever.
    """
    topic = "VALIDATE_DATASET"
    q.publish(topic, {"dataset": "stuck-poison", "tenant": "t1", "uri": "x"})
    # Drive the attempt counter to the cap via repeated consume+nack(requeue).
    for _ in range(2):
        m = q.consume(topic, block=False)
        assert m is not None
        q.nack(m, requeue=True)
    # The 3rd consume reaches the cap; this time do NOT nack — leave it stuck.
    m = q.consume(topic, block=False)
    assert m is not None
    assert q.processing_size(topic) == 1

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)
    reclaimed = reaper.reclaim_stuck_messages(reclaim_timeout=0)
    assert reclaimed == 1
    # Over the cap → dead-lettered, not requeued.
    assert q.dlq_size(topic) == 1
    assert q.consume(topic, block=False) is None


# --- Regression guard: a live worker's message is never reclaimed ---------


def test_legacy_message_without_msg_id_not_reclaimed_from_live_worker(q):
    """A processing-list entry with no `received` stamp is NOT reclaimed.

    A message lacking a `_msg_id` (a legacy/raw payload) gets no `received`
    stamp on consume. A bug caused `reclaim_stale_processing` to see
    `received_at is None` and treat the entry as immediately stale —
    reclaiming a healthy in-flight message out from under its live worker on
    EVERY reaper tick (an infinite-reclaim loop).

    Here a raw message with an empty `_msg_id` is placed on the processing
    list directly (simulating the legacy path) and consumed. A reaper tick
    with a generous timeout must leave it alone — and a second tick too.
    """
    import json as _json

    topic = "VALIDATE_DATASET"
    proc = q.processing_key(topic)
    # A live worker is holding this message: it is on the processing list with
    # an empty `_msg_id`, exactly as the legacy/no-envelope path would leave it.
    raw = _json.dumps({"dataset": "live", "tenant": "t1", "uri": "x", "_msg_id": ""})
    q._redis.rpush(proc, raw)
    assert q.processing_size(topic) == 1

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)

    # Generous reclaim timeout — a just-arrived message is NOT stale. The
    # buggy code reclaimed it anyway because it had no `received` stamp.
    assert reaper.reclaim_stuck_messages(reclaim_timeout=300) == 0
    assert q.processing_size(topic) == 1
    assert q.consume(topic, block=False) is None  # not requeued
    # A second tick must also leave it alone — no infinite-reclaim loop.
    assert reaper.reclaim_stuck_messages(reclaim_timeout=300) == 0
    assert q.processing_size(topic) == 1


def test_publish_always_stamps_a_unique_msg_id(q):
    """`publish()` stamps a `_msg_id` even when the caller supplies an empty one.

    No message may ever reach the queue without a tracking id.
    """
    q.publish("DATASET_READY", {"dataset": "d", "tenant": "t1", "_msg_id": ""})
    msg = q.consume("DATASET_READY", block=False)
    assert msg is not None
    assert msg.msg_id  # non-empty — publish overwrote the empty value
    q.ack(msg)


def test_just_consumed_message_is_not_reclaimed(q):
    """A message consumed a moment ago by a live worker is never reclaimed.

    The `received` stamp is written atomically with the attempt bump, so a
    reaper tick right after `consume()` sees a fresh stamp and a generous
    timeout leaves the in-flight message alone.
    """
    q.publish("VALIDATE_DATASET", {"dataset": "inflight", "tenant": "t1", "uri": "x"})
    msg = q.consume("VALIDATE_DATASET", block=False)
    assert msg is not None
    assert q.processing_size("VALIDATE_DATASET") == 1

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)
    assert reaper.reclaim_stuck_messages(reclaim_timeout=300) == 0
    assert q.processing_size("VALIDATE_DATASET") == 1
    q.ack(msg)


# --- Finding 4: nack after reclaim must not duplicate ---------------------


def test_nack_after_reclaim_does_not_duplicate_message(q):
    """A `nack` of a message a reaper already reclaimed does not re-queue it.

    Finding 4: `nack`'s `lrem` on the processing list returns 0 when the
    reaper already moved the message. The old code ignored that and requeued
    anyway → the message ended up duplicated on the queue. The fix returns
    early when `lrem` removes nothing.
    """
    topic = "VALIDATE_DATASET"
    q.publish(topic, {"dataset": "once-only", "tenant": "t1", "uri": "x"})

    # Worker consumes the message; it is now on the processing list.
    msg = q.consume(topic, block=False)
    assert msg is not None
    assert q.processing_size(topic) == 1

    # The reaper reclaims it (timeout 0) — it goes back onto the main list.
    import adapters.queue.reaper as reaper
    importlib.reload(reaper)
    assert reaper.reclaim_stuck_messages(reclaim_timeout=0) == 1
    assert q.processing_size(topic) == 0

    # The original worker, now late, nacks the SAME message. Its `lrem`
    # removes nothing — the message must NOT be requeued a second time.
    q.nack(msg, requeue=True)

    # Exactly one copy is deliverable. Consume it, ack it, queue is empty.
    first = q.consume(topic, block=False)
    assert first is not None and first["dataset"] == "once-only"
    q.ack(first)
    assert q.consume(topic, block=False) is None
    assert q.processing_size(topic) == 0
    assert q.dlq_size(topic) == 0


# --- Finding 3: single-reaper lock gate -----------------------------------


def test_concurrent_reapers_only_one_acts_per_tick(q):
    """With the Redis lock gate, only one reaper does work per concurrent tick.

    Finding 3: `index_builder` scales to N replicas, each hosting a reaper
    thread. When several reaper ticks fire at the same time the `reaper:lock`
    gate means exactly one `reap_once(gated=True)` runs the tick; the others
    see the lock held and skip. No double-reclaim, no double-flip.
    """
    import json as _json
    import threading as _threading

    topic = "VALIDATE_DATASET"
    proc = q.processing_key(topic)
    # One message stuck on the processing list (a dead worker's).
    raw = _json.dumps(
        {"dataset": "stuck", "tenant": "t1", "uri": "x", "_msg_id": "fixed-id"}
    )
    q._redis.rpush(proc, raw)
    # Stamp an old `received` so it qualifies as stale.
    q._redis.hset(f"{topic}:received", "fixed-id", 0)

    import adapters.queue.reaper as reaper
    importlib.reload(reaper)

    # Simulate N replicas: each reaper carries its own identity, and all fire
    # their tick concurrently against the shared Redis lock.
    results: list[dict] = []
    results_lock = _threading.Lock()
    barrier = _threading.Barrier(3)

    def run_replica(replica_id: str) -> None:
        # Each replica is a distinct process — its own reaper identity.
        reaper._REAPER_ID = replica_id
        barrier.wait()  # all three hit the lock at the same instant
        res = reaper.reap_once(reclaim_timeout=0, stuck_timeout=999999)
        with results_lock:
            results.append(res)

    threads = [
        _threading.Thread(target=run_replica, args=(f"replica-{i}",))
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    acted = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]
    assert len(acted) == 1, "exactly one reaper should run the concurrent tick"
    assert len(skipped) == 2
    # The single acting tick reclaimed the one stuck message — not three times.
    assert acted[0]["messages_reclaimed"] == 1
    # Exactly one copy was requeued — no double-reclaim duplicate.
    assert q.consume(topic, block=False) is not None
    assert q.consume(topic, block=False) is None


def test_reaper_lock_acquire_and_release(q):
    """`acquire_reaper_lock` is mutually exclusive; release frees it."""
    assert q.acquire_reaper_lock("reaper-a", ttl_seconds=60) is True
    # A second distinct reaper cannot acquire while A holds it.
    assert q.acquire_reaper_lock("reaper-b", ttl_seconds=60) is False
    q.release_reaper_lock("reaper-a")
    # Now B can take it.
    assert q.acquire_reaper_lock("reaper-b", ttl_seconds=60) is True
    # A's stale release must NOT delete B's lock.
    q.release_reaper_lock("reaper-a")
    assert q.acquire_reaper_lock("reaper-c", ttl_seconds=60) is False
    q.release_reaper_lock("reaper-b")
