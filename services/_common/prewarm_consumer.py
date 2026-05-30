from __future__ import annotations

"""Long-lived consumer for the `PREWARM_SHARD` queue topic.

Activated when `RB_PREWARM_CONSUMER=true`. Spawns a single daemon thread
that consumes `PREWARM_SHARD` messages off the shipping queue and
dispatches each to `shard_tier.prewarm(shard_uri)`. The admission floor
lives inside `prewarm()`; this consumer's only job is delivery + ack/nack +
survive.

Lifecycle (modelled on `services/_common/catalog_listener.py`):

  - `start_if_needed()` spawns the worker on first call; idempotent.
  - `stop(join_timeout_s)` is idempotent and joins with a short timeout
    so a stuck worker cannot stall shutdown (daemon threads are left to
    die with the process).
  - Reconnect-on-fail is implicit in the queue adapter: `consume()` /
    `ack()` / `nack()` raise on a backend failure, which the worker
    catches with bounded exponential backoff before retrying.

Delivery semantics:

  - Success: `shard_tier.prewarm(uri)` returns a local path -> ack.
  - `CacheCapacityExceeded`: log + nack. The queue's reliable-delivery
    scaffolding handles back-off; after `QUEUE_MAX_ATTEMPTS` the message
    dead-letters.
  - Generic exception (storage outage, malformed payload, etc.): log +
    nack. Same back-off / DLQ path.

Default-off rollback contract: a deployment that never sets
`RB_PREWARM_CONSUMER` runs this file as dead code — `start_if_needed()`
is never called and no thread spawns. The caller (`dp_app.py`) gates the
start under that env var.
"""

import logging
import os
import threading
import time
from typing import Optional

from adapters.queue.queue import ack, consume, nack
from adapters.storage import shard_tier


_LOG = logging.getLogger(__name__)

# Topic name. Hardcoded here rather than imported because the producer
# (`services/index_builder/run.py`) and consumer (this file) are independent
# services; the constant is part of the wire contract.
_TOPIC = "PREWARM_SHARD"

# Backoff bounds for a queue-backend failure. Tests monkeypatch these to
# sub-second values; module scope means an operator can also bump them via
# a startup hook without forking the file.
_BACKOFF_INITIAL_S = 0.5
_BACKOFF_CAP_S = 30.0

# Consume poll timeout. Long enough that the worker is not spinning on an
# idle queue; short enough that `stop()` is observed quickly. Mirrors the
# select-timeout in `catalog_listener`.
_CONSUME_TIMEOUT_S = 1.0


# --- internal state -------------------------------------------------------


_THREAD_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()


# --- public API -----------------------------------------------------------


def start_if_needed() -> None:
    """Spawn the consumer thread on first call; idempotent.

    A second call observes the existing thread and returns; a call after
    `stop()` clears the prior stop signal and spawns a fresh thread (so
    tests with module reloads round-trip cleanly).
    """
    global _THREAD
    with _THREAD_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        _STOP_EVENT.clear()
        _THREAD = threading.Thread(
            target=_run, name="prewarm-consumer", daemon=True,
        )
        _THREAD.start()


def stop(join_timeout_s: float = 2.0) -> None:
    """Signal the consumer thread to exit; join with a short timeout.

    Idempotent: calling stop() with no live thread is a no-op. The join
    timeout prevents a stuck worker from blocking shutdown indefinitely;
    a daemon thread is left to die with the process if the join times out.
    """
    _STOP_EVENT.set()
    with _THREAD_LOCK:
        t = _THREAD
    if t is not None and t.is_alive():
        t.join(timeout=join_timeout_s)


def _is_running() -> bool:
    """Test helper — returns whether the consumer thread is alive."""
    with _THREAD_LOCK:
        t = _THREAD
    return bool(t and t.is_alive())


# --- internals ------------------------------------------------------------


def _dispatch_one(message) -> None:
    """Hand a single PREWARM_SHARD message to `shard_tier.prewarm` and ack/nack.

    Isolation contract:
      - A successful `prewarm` -> ack.
      - `CacheCapacityExceeded` -> log + nack. The queue's bounded retries
        handle back-off; eventually the message dead-letters.
      - Any other exception -> log + nack. One bad message must NOT kill
        the worker — the daemon-thread pattern is the substrate for the
        whole consumer's resilience.
      - A malformed message (no `shard_uri`) is nacked without calling
        prewarm: skipping silently would mask a producer bug.
    """
    shard_uri = message.get("shard_uri") if message is not None else None
    if not shard_uri:
        _LOG.warning(
            "prewarm_consumer: message missing shard_uri; nacking: %r",
            dict(message) if message is not None else None,
        )
        try:
            nack(message)
        except Exception:  # noqa: BLE001 - nack failures are best-effort
            _LOG.exception("prewarm_consumer: nack failed on malformed message")
        return

    try:
        shard_tier.prewarm(shard_uri)
    except shard_tier.CacheCapacityExceeded as exc:
        # Speculative arrival rejected by the admission floor — the
        # operator's signal to raise `RB_SHARD_TIER_BYTES`. The queue's
        # retry scaffolding will redeliver; if every retry hits the same
        # rejection the message dead-letters after QUEUE_MAX_ATTEMPTS.
        _LOG.info(
            "prewarm_consumer: capacity rejection for %s; nacking: %s",
            shard_uri, exc,
        )
        try:
            nack(message)
        except Exception:  # noqa: BLE001
            _LOG.exception("prewarm_consumer: nack failed after capacity rejection")
        return
    except Exception as exc:  # noqa: BLE001 - isolate one bad message
        _LOG.warning(
            "prewarm_consumer: dispatch failed for %s (%s); nacking",
            shard_uri, type(exc).__name__,
            exc_info=True,
        )
        try:
            nack(message)
        except Exception:  # noqa: BLE001
            _LOG.exception("prewarm_consumer: nack failed after dispatch error")
        return

    try:
        ack(message)
    except Exception:  # noqa: BLE001 - ack failures get logged but the
        # prewarm already happened; redelivery would be a no-op (warm hit).
        _LOG.exception("prewarm_consumer: ack failed for %s", shard_uri)


def _run() -> None:
    """Worker loop: consume / dispatch / repeat, with bounded backoff on error.

    Outer try/except wraps each consume() so a transient queue-backend
    failure (Redis blip, connection drop) triggers a bounded retry rather
    than killing the worker. The `_STOP_EVENT.wait(backoff)` returns True
    if stop() fires during backoff, so shutdown is prompt.
    """
    backoff = _BACKOFF_INITIAL_S
    while not _STOP_EVENT.is_set():
        try:
            msg = consume(_TOPIC, block=True, timeout=_CONSUME_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            _LOG.exception(
                "prewarm_consumer: consume failed; backoff %.1fs", backoff,
            )
            if _STOP_EVENT.wait(timeout=backoff):
                return
            backoff = min(backoff * 2.0, _BACKOFF_CAP_S)
            continue
        if msg is None:
            # Idle queue — reset backoff so a transient failure that
            # cleared is not penalised on the next real error.
            backoff = _BACKOFF_INITIAL_S
            continue
        _dispatch_one(msg)
        backoff = _BACKOFF_INITIAL_S
