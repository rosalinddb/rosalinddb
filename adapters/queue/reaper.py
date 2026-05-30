from __future__ import annotations

"""Queue + catalog reconciliation reaper.

The reliable-queue processing-list pattern (`adapters/queue/queue.py`) makes a
worker death mid-job *recoverable* — the message stays on `<topic>:processing`
instead of being lost — but something has to actually move it back. The reaper
is that something. It runs as a periodic task *inside an existing worker
process* (the index builder hosts it; see `index_builder/run.py`) rather than
as a brand-new service, because RosalindDB's deploy runs each service as its
own process group and we want to minimise process types.

Two reconciliation passes run on each tick:

  1. **Processing-list reclaim.** For every known topic, any message that has
     been on `<topic>:processing` longer than `QUEUE_RECLAIM_TIMEOUT` belongs
     to a worker that took it and then died (or hung) without acking. Such a
     message is requeued for another delivery attempt, or dead-lettered if it
     is already over `QUEUE_MAX_ATTEMPTS`.

  2. **Stuck-dataset reconcile.** Any dataset stranded in a non-terminal status
     (`validating`/`indexing`) for longer than `DATASET_STUCK_TIMEOUT` is
     flipped to `error` with a clear `error_message`. This is the backstop for
     a worker *hang* — the queue reclaim handles a worker *death*, but a worker
     that is alive-but-wedged never releases its message, so the catalog-side
     timeout is the guarantee that a customer's `GET /v1/datasets/{name}` can
     never report a silently-stuck dataset forever.

Both timeouts are env-configurable. The reaper's host (`index_builder`) is
planned to scale to multiple replicas, so several reaper threads can exist at
once. Two layers keep that safe:

  - **Single-reaper lock.** Each tick first acquires a short-lived Redis lock
    (`SET reaper:lock <id> NX EX <ttl>`); only the holder runs the tick, the
    rest skip. So exactly one reaper acts per tick regardless of replica count.
  - **Idempotent primitives** as a backstop: `reclaim_stale_processing` uses
    `LREM` (the loser of a race removes nothing and skips), and the stuck-
    dataset flip is a compare-and-set (`fail_dataset_if_stale`) that never
    clobbers a terminal status a worker just wrote.
"""

import os
import threading
import time
import uuid

from adapters.queue.queue import (
    acquire_reaper_lock,
    reclaim_stale_processing,
    release_reaper_lock,
)
from adapters.state.state import fail_dataset_if_stale, find_stale_datasets

# Topics whose processing lists the reaper reclaims. These are the durable
# pipeline topics; `RESULT_READY` is deliberately excluded — an ephemeral query
# result is request-scoped and a lost one simply times the request out.
RECLAIMED_TOPICS = (
    "VALIDATE_DATASET",
    "DATASET_READY",
    "RUN_EPHEMERAL_QUERY",
)

# How long a message may sit on a processing list before the reaper assumes
# the worker that took it has died/hung and reclaims it. Generous by default so
# a slow-but-healthy job is never reclaimed out from under a live worker.
QUEUE_RECLAIM_TIMEOUT = float(os.getenv("QUEUE_RECLAIM_TIMEOUT", "300"))

# How long a dataset may sit in `validating`/`indexing` before the reaper flips
# it to `error`. The backstop for a worker hang.
DATASET_STUCK_TIMEOUT = float(os.getenv("DATASET_STUCK_TIMEOUT", "900"))

# Seconds between reaper ticks when run as a background loop.
REAPER_INTERVAL = float(os.getenv("REAPER_INTERVAL", "30"))

# TTL of the single-reaper lock. A little longer than the tick interval so the
# lock outlives one tick even under clock skew, but still expires on its own if
# the holder dies — no stuck-lock starvation.
REAPER_LOCK_TTL = float(os.getenv("REAPER_LOCK_TTL", str(REAPER_INTERVAL + 30)))

# Per-process identity stamped into the reaper lock so `release_reaper_lock`
# only deletes a lock this process still owns.
_REAPER_ID = uuid.uuid4().hex


def reclaim_stuck_messages(reclaim_timeout: float = QUEUE_RECLAIM_TIMEOUT) -> int:
    """Reclaim messages stranded on any pipeline topic's processing list.

    Returns the total number of messages reclaimed across all topics.
    """
    total = 0
    for topic in RECLAIMED_TOPICS:
        total += reclaim_stale_processing(topic, reclaim_timeout)
    return total


def reconcile_stuck_datasets(
    stuck_timeout: float = DATASET_STUCK_TIMEOUT,
) -> int:
    """Flip datasets stranded in a non-terminal status to `error`.

    A dataset whose `status` has been `validating`/`indexing` for longer than
    `stuck_timeout` is flipped to `error` with an explanatory message.

    The flip is a compare-and-set (`fail_dataset_if_stale`), NOT an
    unconditional `update_dataset_status`: between `find_stale_datasets`
    reading a dataset as stale and this flip running, a worker may legitimately
    finish the job and write `indexed`. The guarded flip only writes `error` if
    the dataset is STILL non-terminal and stale, so a terminal status a worker
    just wrote is never clobbered. Returns the number actually flipped.
    """
    stale = find_stale_datasets(stuck_timeout)
    flipped = 0
    for ds in stale:
        if fail_dataset_if_stale(
            ds["tenant_id"],
            ds["dataset_name"],
            stuck_timeout,
            error_message=(
                f"reaper: dataset stuck in '{ds['status']}' for over "
                f"{int(stuck_timeout)}s — the worker processing it likely "
                "crashed or hung; re-ingest to retry"
            ),
        ):
            flipped += 1
    return flipped


def reap_once(
    reclaim_timeout: float = QUEUE_RECLAIM_TIMEOUT,
    stuck_timeout: float = DATASET_STUCK_TIMEOUT,
    gated: bool = True,
) -> dict:
    """Run one full reconciliation pass.

    When `gated` is True (the default, used by `run_reaper_loop`) the tick is
    guarded by the single-reaper Redis lock: with several `index_builder`
    replicas each hosting a reaper thread, exactly one acquires the lock and
    runs the tick — the rest skip. On the in-process backend there is nothing
    to gate and the tick always runs.

    `gated=False` bypasses the lock — used by tests that drive a single
    deterministic tick directly.

    Returns a `{"messages_reclaimed", "datasets_failed", "skipped"}` summary so
    a caller (or a test) can assert what the tick did. `skipped` is True when
    the lock was held by another reaper and this tick did no work.
    """
    if gated and not acquire_reaper_lock(_REAPER_ID, REAPER_LOCK_TTL):
        return {"messages_reclaimed": 0, "datasets_failed": 0, "skipped": True}
    try:
        reclaimed = reclaim_stuck_messages(reclaim_timeout)
        failed = reconcile_stuck_datasets(stuck_timeout)
    finally:
        if gated:
            release_reaper_lock(_REAPER_ID)
    return {
        "messages_reclaimed": reclaimed,
        "datasets_failed": failed,
        "skipped": False,
    }


def run_reaper_loop(stop_event: threading.Event | None = None) -> None:
    """Run the reaper on a fixed interval until `stop_event` is set.

    Designed to be the target of a daemon thread started by a worker process
    (`start_reaper_thread`). Each tick's failures are swallowed and logged so a
    transient Redis/Postgres blip never kills the loop.
    """
    while stop_event is None or not stop_event.is_set():
        try:
            summary = reap_once()
            if summary["messages_reclaimed"] or summary["datasets_failed"]:
                print(
                    f"reaper: reclaimed {summary['messages_reclaimed']} message(s), "
                    f"failed {summary['datasets_failed']} stuck dataset(s)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"reaper: tick failed: {exc}")
        if stop_event is not None:
            if stop_event.wait(REAPER_INTERVAL):
                break
        else:
            time.sleep(REAPER_INTERVAL)


def start_reaper_thread(stop_event: threading.Event | None = None) -> threading.Thread:
    """Start the reaper loop in a daemon thread and return it.

    Called once on worker startup. The thread is a daemon so it never blocks
    process exit; graceful shutdown additionally signals `stop_event` so the
    loop returns promptly rather than mid-`sleep`.
    """
    t = threading.Thread(
        target=run_reaper_loop, args=(stop_event,), name="queue-reaper", daemon=True
    )
    t.start()
    print(
        f"reaper: started (reclaim_timeout={QUEUE_RECLAIM_TIMEOUT}s, "
        f"dataset_stuck_timeout={DATASET_STUCK_TIMEOUT}s, interval={REAPER_INTERVAL}s)"
    )
    return t
