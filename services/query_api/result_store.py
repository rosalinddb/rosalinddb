from __future__ import annotations

"""Shared ephemeral-query result store for `query_api`.

`POST /v1/query` against a not-yet-indexed dataset enqueues the work on the
ephemeral runner and returns a `job_id`. The runner publishes the answer to the
`RESULT_READY` queue topic; `query_api`'s `RESULT_READY` consumer stashes it
here, and `GET /v1/query/status/{job_id}` reads it back.

Multi-worker safety. With `query_api` running multiple uvicorn workers /
replicas, the `RESULT_READY` consumer in replica A and the status poll in
replica B are different processes. An in-process dict therefore loses the
result across replicas — A stores it, B's poll sees "not found". This store
puts the result in **Redis** (shared across every replica) under
`query_result:{job_id}` with a short TTL: ephemeral results are transient, so
an hour is ample for a client to poll and well short of unbounded growth.

When `REDIS_URL` is unset (the unit-test / single-process mode) there is no
Redis to share and only one process, so an in-process dict fallback is correct
and keeps the unit suite hermetic. The fallback dict is exposed as
`_RESULTS` so existing tests that reset it (`v1_query._RESULTS.clear()`) keep
working.
"""

import json
import os
import threading
from typing import Dict, Optional

# Reuse the queue adapter's Redis connection mechanism: it already binds a
# client from `REDIS_URL` at import time. Sharing it means one connection
# pool and one source of truth for "are we in Redis mode".
from adapters.queue.queue import _redis as _queue_redis

# Key prefix for an ephemeral result in Redis.
_KEY_PREFIX = "query_result:"

# Ephemeral results are short-lived — a client polls `GET /v1/query/status`
# for a few seconds. One hour is generous headroom for a slow poller while
# still bounding Redis memory (the key self-expires; no sweeper needed).
RESULT_TTL_SECONDS = int(os.getenv("RB_QUERY_RESULT_TTL", "3600"))

# In-process fallback used only when there is no `REDIS_URL` (unit tests /
# single-process mode). Guarded by a lock so the RESULT_READY consumer thread
# and a status poll never race. Exposed for test reset.
_RESULTS: Dict[str, dict] = {}
_RESULTS_LOCK = threading.Lock()


def _key(job_id: str) -> str:
    """Return the Redis key for a job's ephemeral result."""
    return f"{_KEY_PREFIX}{job_id}"


def store_result(job_id: str, result: dict) -> None:
    """Persist an ephemeral query result for later polling.

    Redis mode: `SET query_result:{job_id} <json> EX RESULT_TTL_SECONDS` —
    every `query_api` replica can read it back, and it self-expires. In-process
    mode: stash it in the `_RESULTS` dict under the lock.
    """
    if _queue_redis is not None:
        _queue_redis.set(_key(job_id), json.dumps(result), ex=RESULT_TTL_SECONDS)
        return
    with _RESULTS_LOCK:
        _RESULTS[job_id] = dict(result)


def get_result(job_id: str) -> Optional[dict]:
    """Return a stored ephemeral result, or None if absent / expired.

    Redis mode reads `query_result:{job_id}`; in-process mode reads `_RESULTS`.
    A missing key — unknown or not-yet-ready job, or a TTL-expired result —
    returns None, which the status endpoint maps to `{"ready": false}`.
    """
    if _queue_redis is not None:
        raw = _queue_redis.get(_key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            decoded = json.loads(raw)
        except Exception:  # noqa: BLE001 - defensive: a corrupt value is "absent"
            return None
        return decoded if isinstance(decoded, dict) else None
    with _RESULTS_LOCK:
        res = _RESULTS.get(job_id)
        return dict(res) if res is not None else None


def clear() -> None:
    """Drop every stored result — test helper / process reset.

    Redis mode deletes every `query_result:*` key; in-process mode empties the
    fallback dict.
    """
    if _queue_redis is not None:
        keys = list(_queue_redis.scan_iter(match=f"{_KEY_PREFIX}*"))
        if keys:
            _queue_redis.delete(*keys)
        return
    with _RESULTS_LOCK:
        _RESULTS.clear()
