from __future__ import annotations

"""Queue adapter abstraction.

Provides a publish/consume/ack API backed by in-process queues by default and
Redis when `REDIS_URL` is set. Topics used in this project include
`VALIDATE_DATASET`, `DATASET_READY`, and `RUN_EPHEMERAL_QUERY`.

**Delivery semantics.** The Redis path is *at-least-once*. `BRPOP` removed a
message from Redis the instant it was read, so a worker killed mid-job (a
deploy/restart) silently lost the in-flight message and stranded the dataset in
a non-terminal status forever. The new path is a **reliable processing-list**
pattern:

  - `publish()` does `LPUSH <topic>`.
  - `consume()` does `LMOVE <topic> <topic>:processing LEFT RIGHT` (atomically
    moving the message onto a per-topic processing list) and records the raw
    payload + an attempt count in a side hash `<topic>:attempts`.
  - `ack(msg)` removes the message from the processing list — called ONLY after
    the consumer finishes successfully.
  - `nack(msg)` either re-queues the message for another attempt or, once the
    per-message attempt count crosses `QUEUE_MAX_ATTEMPTS`, routes it to the
    dead-letter list `<topic>:dlq`.

A worker that dies between `consume()` and `ack()` leaves the message on the
processing list; the reaper (`reaper.py`) reclaims it after a timeout. Because
RosalindDB's workers are idempotent (a duplicate `DATASET_READY` is a no-op via
the shard `indexed_landing_uris` manifest; re-validation re-writes Parquet
harmlessly) at-least-once redelivery is safe.

**In-process backend.** The default `queue.Queue` backend is test-only. It
keeps at-most-once semantics — `get()` removes the message — and `ack`/`nack`
are best-effort no-ops (a `nack` re-publishes). The reliable-delivery
guarantees are a Redis-path concern; the unit suite stays green either way.

**Trace-context propagation.** `publish()` injects the current W3C trace
context into the message under the reserved `_otel` key; `consume()` extracts
it and starts a child span (`queue.consume <topic>`) linked to the producer's
trace. The `_otel` key is stripped from the `Message` that `consume()` returns,
so it never leaks into business/validation logic.
"""

import json
import os
import queue
import time
import uuid
from typing import Any, Dict, Optional

from opentelemetry import context as _otel_context
from opentelemetry import trace as _otel_trace
from opentelemetry.propagate import extract as _otel_extract
from opentelemetry.propagate import inject as _otel_inject

# Reserved message key carrying the serialised W3C trace context. Stripped
# before the payload is handed to consumers.
_OTEL_KEY = "_otel"
# Reserved envelope key carrying the per-message delivery id. Used to address
# the exact processing-list entry on ack/nack; stripped before the payload is
# handed to consumers.
_MSG_ID_KEY = "_msg_id"

# Max delivery attempts before a message is dead-lettered. A poison message
# that crashes the worker every time must not redeliver forever.
QUEUE_MAX_ATTEMPTS = int(os.getenv("QUEUE_MAX_ATTEMPTS", "5"))

_local_queues: dict[str, queue.Queue] = {
    "VALIDATE_DATASET": queue.Queue(),
    "DATASET_READY": queue.Queue(),
    "DELETE_VECTORS": queue.Queue(),
    # `CONSOLIDATE`: a (tenant, dataset) recall partition needs folding into a
    # new Consolidated shard + the watermark advanced (the recall→consolidated
    # flush). Enqueued by the per-tenant recall-row cap on the write path and by
    # the builder's consolidate-on-idle sweep; consumed by the index builder
    # (single-replica, per-dataset advisory lock). See
    # docs/architecture/recall-consolidate.md, "Consolidation / flush".
    "CONSOLIDATE": queue.Queue(),
    "SHARD_BUILT": queue.Queue(),
    "MERGE_READY": queue.Queue(),
    "RUN_EPHEMERAL_QUERY": queue.Queue(),
    "RESULT_READY": queue.Queue(),
}

_REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if _REDIS_URL:
    import redis  # type: ignore

    _redis = redis.from_url(_REDIS_URL)

_tracer = _otel_trace.get_tracer("rosalinddb.queue")


# --- key helpers ----------------------------------------------------------


def processing_key(topic: str) -> str:
    """Redis list holding messages a consumer has taken but not yet acked."""
    return f"{topic}:processing"


def dlq_key(topic: str) -> str:
    """Redis list holding messages that exhausted `QUEUE_MAX_ATTEMPTS`."""
    return f"{topic}:dlq"


def _attempts_key(topic: str) -> str:
    """Redis hash mapping a message id to its delivery attempt count."""
    return f"{topic}:attempts"


# --- Message --------------------------------------------------------------


class Message(dict):
    """A consumed message: the decoded payload plus delivery bookkeeping.

    Subclasses `dict` so existing callers can keep doing `msg["dataset"]`,
    `msg.get("tenant")`, truthiness checks, etc. The delivery bookkeeping
    (`raw`, `topic`, `msg_id`) lives on attributes, not dict keys, so it never
    collides with payload fields and is not visible to business logic.
    """

    __slots__ = ("topic", "msg_id", "raw")

    def __init__(self, payload: Dict[str, Any], topic: str, msg_id: str, raw: str):
        super().__init__(payload)
        self.topic = topic
        self.msg_id = msg_id
        self.raw = raw


# --- publish --------------------------------------------------------------


def publish(topic: str, message: Dict[str, Any]) -> None:
    """Publish a message to a topic.

    The current trace context is injected under the `_otel` key and a unique
    `_msg_id` is stamped so a later ack/nack can address the exact entry. A
    shallow copy is made first so the caller's dict is not mutated.

    The `_msg_id` is stamped UNCONDITIONALLY (overwriting any caller-supplied
    value) so that every message that ever reaches the queue carries a
    tracking identity. A message lacking a `_msg_id` cannot be stamped with a
    `received` timestamp on consume, which the reaper would then misread as
    "immediately stale" and reclaim out from under a live worker every tick.

    Args:
        topic: Logical channel name.
        message: JSON-serializable payload.
    """
    enveloped = dict(message)
    # Always stamp a fresh id — never `setdefault`. A caller-supplied or empty
    # `_msg_id` must not survive: a message with no tracking identity breaks
    # the reaper's staleness judgement (see `reclaim_stale_processing`).
    enveloped[_MSG_ID_KEY] = uuid.uuid4().hex
    carrier: Dict[str, str] = {}
    # `inject` writes the active span's W3C `traceparent` (and `tracestate`)
    # into the carrier. A no-op tracer / no active span → empty carrier.
    try:
        _otel_inject(carrier)
    except Exception:  # noqa: BLE001 — never let observability break publish
        carrier = {}
    if carrier:
        enveloped[_OTEL_KEY] = carrier

    raw = json.dumps(enveloped)
    if _redis is not None:
        _redis.lpush(topic, raw)
        return
    if topic not in _local_queues:
        _local_queues[topic] = queue.Queue()
    _local_queues[topic].put(raw)


# --- consume --------------------------------------------------------------


def consume(
    topic: str, block: bool = True, timeout: Optional[float] = None
) -> Optional[Message]:
    """Consume a message from a topic.

    Redis path: atomically `LMOVE`s the message onto `<topic>:processing` (so a
    worker death leaves it reclaimable) and bumps its attempt count. The caller
    MUST `ack()` it on success or `nack()` it on failure; an unacked message is
    redelivered by the reaper after `QUEUE_RECLAIM_TIMEOUT`.

    In-process path: `get()` removes the message (at-most-once, test-only); the
    returned `Message` still supports `ack`/`nack` as best-effort no-ops.

    Extracts any `_otel` trace context and starts a `queue.consume <topic>`
    span as a child of the originating trace. The `_otel` / `_msg_id` keys are
    removed from the returned `Message` so neither reaches business logic.

    Returns:
        A `Message` (a dict subclass carrying the payload) or None if none
        available in time.
    """
    if _redis is not None:
        raw = _redis_take(topic, block=block, timeout=timeout)
    else:
        if topic not in _local_queues:
            _local_queues[topic] = queue.Queue()
        try:
            raw = _local_queues[topic].get(block=block, timeout=timeout)
        except queue.Empty:
            raw = None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

    if raw is None:
        return None

    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        # Defensive: a non-dict payload cannot be addressed for ack; treat the
        # raw value as the message id so processing-list cleanup still works.
        msg_id = ""
    else:
        msg_id = decoded.get(_MSG_ID_KEY, "")

    if _redis is not None and msg_id:
        # Bump the attempt count and stamp the delivery time so the reaper can
        # age out this exact processing-list entry if the worker dies before
        # acking. Both writes are keyed by the stable `_msg_id` and are issued
        # together in a single MULTI/EXEC pipeline so the reaper can never
        # observe an `attempts` bump without its matching `received` stamp.
        pipe = _redis.pipeline(transaction=True)
        pipe.hincrby(_attempts_key(topic), msg_id, 1)
        pipe.hset(f"{topic}:received", msg_id, time.time())
        pipe.execute()

    payload = _continue_trace(topic, decoded)
    if not isinstance(payload, dict):
        payload = {}
    return Message(payload, topic=topic, msg_id=msg_id, raw=raw)


def _redis_take(
    topic: str, block: bool, timeout: Optional[float]
) -> Optional[str]:
    """Atomically move one message from `topic` onto its processing list.

    `LMOVE topic topic:processing LEFT RIGHT` (and its blocking sibling
    `BLMOVE`) is the at-least-once primitive: the message is never absent from
    Redis — it is on the main list OR the processing list — so a crash between
    the move and the ack cannot lose it.
    """
    proc = processing_key(topic)
    if block:
        # BLMOVE timeout 0 blocks forever; callers pass a finite timeout.
        res = _redis.blmove(topic, proc, float(timeout or 0), src="LEFT", dest="RIGHT")
    else:
        res = _redis.lmove(topic, proc, src="LEFT", dest="RIGHT")
    if res is None:
        return None
    return res.decode("utf-8") if isinstance(res, bytes) else res


# --- ack / nack -----------------------------------------------------------


def ack(message: Optional[Message]) -> None:
    """Acknowledge a message — remove it from the processing list for good.

    Call ONLY after the consumer has finished the job successfully. For the
    Redis path this `LREM`s the message from `<topic>:processing` and clears
    its attempt counter. For the in-process path it is a no-op (the message
    already left the `queue.Queue` on `get()`).
    """
    if message is None or _redis is None:
        return
    _redis.lrem(processing_key(message.topic), 1, message.raw)
    if message.msg_id:
        _redis.hdel(_attempts_key(message.topic), message.msg_id)
        _redis.hdel(f"{message.topic}:received", message.msg_id)


def nack(message: Optional[Message], requeue: bool = True) -> None:
    """Negatively-acknowledge a message after a handling failure.

    Removes it from the processing list, then either re-queues it for another
    delivery attempt or — once it has been delivered `QUEUE_MAX_ATTEMPTS`
    times — routes it to the dead-letter list `<topic>:dlq` and returns False
    via `dead_lettered`. For the in-process path `requeue` re-publishes the
    message; there is no DLQ tracking there.

    Returns:
        True if the message was dead-lettered (exhausted its retries), else
        False.
    """
    if message is None:
        return False
    if _redis is None:
        # In-process / test path: best-effort requeue, no attempt tracking.
        if requeue:
            try:
                publish(message.topic, dict(message))
            except Exception:  # noqa: BLE001
                pass
        return False

    topic = message.topic
    # If `lrem` removes nothing the message is no longer on the processing
    # list — a reaper already reclaimed it, or this is a double-nack. Either
    # way another path owns the message now; requeueing/dead-lettering here
    # would duplicate it on the queue. Return without touching it.
    if _redis.lrem(processing_key(topic), 1, message.raw) == 0:
        return False
    attempts = 0
    if message.msg_id:
        attempts = int(_redis.hget(_attempts_key(topic), message.msg_id) or 0)

    if not requeue or attempts >= QUEUE_MAX_ATTEMPTS:
        _redis.lpush(dlq_key(topic), message.raw)
        if message.msg_id:
            _redis.hdel(_attempts_key(topic), message.msg_id)
            _redis.hdel(f"{topic}:received", message.msg_id)
        return True
    # Re-queue for another attempt. The attempt counter persists so the next
    # consume()'s HINCRBY keeps counting toward the cap; the received-time
    # stamp is cleared so it is re-stamped fresh on the next delivery.
    if message.msg_id:
        _redis.hdel(f"{topic}:received", message.msg_id)
    _redis.rpush(topic, message.raw)
    return False


# --- introspection (used by the reaper + tests) ---------------------------


def dlq_size(topic: str) -> int:
    """Number of messages currently in `topic`'s dead-letter list."""
    if _redis is None:
        return 0
    return int(_redis.llen(dlq_key(topic)))


def processing_size(topic: str) -> int:
    """Number of in-flight (consumed-but-unacked) messages for `topic`."""
    if _redis is None:
        return 0
    return int(_redis.llen(processing_key(topic)))


def peek_dlq(topic: str) -> list[Dict[str, Any]]:
    """Return the decoded payloads currently on `topic`'s dead-letter list."""
    if _redis is None:
        return []
    out: list[Dict[str, Any]] = []
    for raw in _redis.lrange(dlq_key(topic), 0, -1):
        try:
            decoded = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(decoded, dict):
            decoded.pop(_OTEL_KEY, None)
            decoded.pop(_MSG_ID_KEY, None)
        out.append(decoded)
    return out


def acquire_reaper_lock(lock_id: str, ttl_seconds: float) -> bool:
    """Try to acquire the short-lived single-reaper lock for one tick.

    The reaper is hosted as a daemon thread inside `index_builder`, which is
    planned to scale to multiple replicas — so N reapers would otherwise run
    concurrently, wastefully widening every race window. This lock gates each
    tick: `SET reaper:lock <id> NX EX <ttl>` succeeds for exactly one caller;
    the others see it already held and skip the tick.

    `ttl` should be a little longer than the reaper interval so the lock
    naturally expires before the next tick even if the holder dies without
    releasing it (no stuck-lock starvation).

    In `memory://` / in-process mode there is no Redis and only one process,
    so there is nothing to gate — the reaper runs unconditionally and this
    returns True.

    Returns True if the caller holds the lock and should run the tick.
    """
    if _redis is None:
        return True
    return bool(
        _redis.set("reaper:lock", lock_id, nx=True, ex=max(1, int(ttl_seconds)))
    )


def release_reaper_lock(lock_id: str) -> None:
    """Release the reaper lock iff this caller still owns it.

    A compare-and-delete (Lua) so a tick that overran its `ttl` — letting the
    lock expire and another reaper acquire it — does not delete the new
    holder's lock. A no-op in in-process mode.
    """
    if _redis is None:
        return
    try:
        _redis.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end",
            1,
            "reaper:lock",
            lock_id,
        )
    except Exception:  # noqa: BLE001 — releasing is best-effort; ttl backs it up
        pass


def reclaim_stale_processing(topic: str, older_than_seconds: float) -> int:
    """Reclaim messages stuck on `topic`'s processing list (reaper primitive).

    A message that has been on the processing list longer than
    `older_than_seconds` belongs to a worker that took it and then died (or
    hung) without acking. Each such message is moved back: re-queued for
    another delivery attempt, or dead-lettered if it is already over the
    attempt cap. The per-message `<topic>:received` hash records when each
    delivery happened so "stale" can be judged without a wall-clock scan.

    A processing-list entry whose `received` stamp is MISSING (an empty/legacy
    `_msg_id`, or the tiny window between `BLMOVE` and the stamp pipeline) is
    NOT treated as stale: a just-arrived message must never be reclaimed out
    from under a live worker. Instead the reaper stamps it `now` and skips it
    this tick, so it ages from this moment forward like any other message.

    Returns the number of messages reclaimed.
    """
    if _redis is None:
        return 0
    proc = processing_key(topic)
    received_key = f"{topic}:received"
    now = time.time()
    reclaimed = 0
    for raw in _redis.lrange(proc, 0, -1):
        raw_s = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            decoded = json.loads(raw_s)
        except Exception:  # noqa: BLE001
            decoded = {}
        msg_id = decoded.get(_MSG_ID_KEY, "") if isinstance(decoded, dict) else ""
        received_at = _redis.hget(received_key, msg_id) if msg_id else None
        if received_at is None:
            # No delivery stamp — either a legacy/empty `_msg_id`, or this
            # message arrived between its `BLMOVE` and its stamp pipeline.
            # Treat it as just-arrived: stamp it `now` so it starts aging,
            # and do NOT reclaim it this tick. Never reclaim an unstamped
            # entry — that is the infinite-reclaim-of-a-live-worker bug.
            if msg_id:
                _redis.hset(received_key, msg_id, now)
            continue
        if (now - float(received_at)) < older_than_seconds:
            continue
        # Reclaim: drop from processing, then requeue or dead-letter.
        if _redis.lrem(proc, 1, raw_s) == 0:
            continue  # another reaper / an ack got there first
        attempts = int(_redis.hget(_attempts_key(topic), msg_id) or 0) if msg_id else 0
        if attempts >= QUEUE_MAX_ATTEMPTS:
            _redis.lpush(dlq_key(topic), raw_s)
            if msg_id:
                _redis.hdel(_attempts_key(topic), msg_id)
                _redis.hdel(received_key, msg_id)
        else:
            _redis.rpush(topic, raw_s)
            if msg_id:
                _redis.hdel(received_key, msg_id)
        reclaimed += 1
    return reclaimed


# --- trace propagation ----------------------------------------------------


def _continue_trace(topic: str, decoded: Any) -> Any:
    """Strip `_otel`/`_msg_id`, link the consumer's work to the producer trace.

    Starts a short `queue.consume <topic>` span as a child of the extracted
    context and activates that context for the current thread so any spans the
    consumer opens next (e.g. `validate_dataset`) nest under the originating
    trace. Observability failures are swallowed — delivery must never break.
    """
    if not isinstance(decoded, dict):
        return decoded
    decoded.pop(_MSG_ID_KEY, None)
    carrier = decoded.pop(_OTEL_KEY, None)
    if not carrier:
        return decoded

    try:
        parent_ctx = _otel_extract(carrier)
        with _tracer.start_as_current_span(
            f"queue.consume {topic}", context=parent_ctx
        ) as sp:
            sp.set_attribute("messaging.system", "rosalinddb-queue")
            sp.set_attribute("messaging.destination.name", topic)
            sp.set_attribute("messaging.operation", "receive")
        # Attach the parent context for the rest of this thread's handling so
        # the consumer's pipeline spans link back to the originating trace.
        _otel_context.attach(parent_ctx)
    except Exception:  # noqa: BLE001
        pass
    return decoded
