"""Local shard fetch with optional per-URI download coalescing.

Single home for the ``_ensure_cached`` logic that
``services.query_api.v1_query`` (coalesced) and
``services.ephemeral_runner.run`` (non-coalesced) previously duplicated.

FAISS's ``read_index`` only accepts a filesystem path, so an object-store shard
(``s3://`` or ``memory://``) is fetched once into ``cache_dir``. There is no
``file://`` branch: RosalindDB is object-storage-first and ``memory://`` is the
unit-test backend.

Coalescing is a strict superset of the plain path, controlled by the
``coalescing`` flag:

  * ``coalescing=True`` (hot path): concurrent callers for the same URI are
    coalesced into a single download via the caller-supplied ``inflight``
    registry. Without it, N concurrent queries against a cold shard issue N
    parallel GETs, the object store throttles, and the cold warm-up fails for
    everyone. With it, exactly one thread downloads and the rest block on a
    per-URI ``Event`` until the rename publishes the file.
  * ``coalescing=False`` (ephemeral path): the plain SSD-tier-gated, atomic
    temp+rename body, with no single-flight coordination. The ephemeral path
    runs at much lower concurrency, where the download stampede has not been
    observed.

State (``cache_dir``, the in-flight registry + lock, the coalesce deadline and
the timeout class) is passed IN so each caller keeps owning its own
module-level constants — tests reset / monkeypatch them by module attribute and
those changes must take effect at call time.

The SSD-tier ``fetch`` and ``download_to`` are imported locally (lazily, inside
the function) so they pick up monkeypatches against
``adapters.storage.storage`` / ``adapters.storage.shard_tier`` exactly as the
two inline copies did.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Dict, Optional, Type


def ensure_cached(
    shard_uri: str,
    *,
    cache_dir: str,
    coalescing: bool,
    inflight: Optional[Dict[str, threading.Event]] = None,
    inflight_lock: Optional[threading.Lock] = None,
    coalesce_wait_s: float = 0.0,
    timeout_exc: Optional[Type[BaseException]] = None,
) -> str:
    """Ensure a shard is present in the local cache and return its path.

    SSD-tier activation gate. When ``RB_SHARD_TIER_BYTES`` is set in the
    environment, delegation runs through
    ``adapters.storage.shard_tier.fetch(shard_uri)`` — the same single-flight
    contract, but the local file is owned by the tier (which runs its own
    byte-budgeted LRU eviction). The env check happens here rather than at
    import time so a flip-flop env var across pod restarts cleanly toggles the
    path without needing a code redeploy. When the env is unset, the legacy
    body below runs unchanged — that is the rollback contract.

    With ``coalescing=True`` the caller must supply ``inflight``,
    ``inflight_lock``, ``coalesce_wait_s`` and ``timeout_exc``: concurrent
    callers for the same URI are coalesced into a single download — see the
    module docstring for the why.
    """
    # SSD-tier import. Imported lazily so a monkeypatch against
    # `adapters.storage.shard_tier` (and the env gate) is honoured at call time.
    from adapters.storage import shard_tier

    if os.getenv("RB_SHARD_TIER_BYTES"):
        # Tier handles its own directory creation, single-flight, and
        # eviction. `ShardTierTimeout` and `FileNotFoundError` are the two
        # raise paths the caller's classifier already maps to 503; let them
        # propagate untouched.
        return shard_tier.fetch(shard_uri)

    os.makedirs(cache_dir, exist_ok=True)
    if not (shard_uri.startswith("s3://") or shard_uri.startswith("memory://")):
        raise ValueError("Unsupported shard uri")

    cache_key = shard_uri.split("://", 1)[1].replace("/", "_")
    path = os.path.join(cache_dir, cache_key)

    if not coalescing:
        # Plain path (ephemeral runner): atomic temp+rename, no single-flight.
        # `download_to` streams the GET to disk without buffering the whole
        # object in RAM (matches the hot path; both use this pattern to avoid
        # the multi-GB-shard OOM of the prior `f.write(read_bytes(uri))`).
        from adapters.storage.storage import download_to

        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                download_to(shard_uri, tmp)
                # Atomic publish — concurrent readers never see a partial file.
                os.replace(tmp, path)
            except BaseException:
                # On any failure before the rename, remove the leftover temp
                # file so a crash mid-write does not leak `.tmp` files into
                # CACHE_DIR. Best-effort: the file may already be gone.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        return path

    # Coalesced path (hot path). The caller-owned single-flight state must be
    # present.
    assert inflight is not None and inflight_lock is not None
    assert timeout_exc is not None

    # Fast path: file already on disk. No coordination needed — the file is
    # already atomic (written via temp + rename) so a concurrent reader sees
    # either nothing or the complete bytes. Skip the lock/event dance
    # entirely so the warm-cache case stays byte-for-byte unchanged.
    if os.path.exists(path):
        return path

    # Single-flight registration. Atomically decide whether this thread is
    # the *initiator* (creates the entry, will do the GET) or a *waiter*
    # (found a pre-existing entry, will block on its Event).
    is_initiator = False
    with inflight_lock:
        # A second `exists` check inside the lock closes a race where the
        # initiator finished the download AND cleared the in-flight entry
        # between our outer `exists` check and acquiring the lock. Without
        # this, we'd needlessly become a fresh initiator for an already-
        # cached shard.
        if os.path.exists(path):
            return path
        event = inflight.get(shard_uri)
        if event is None:
            event = threading.Event()
            inflight[shard_uri] = event
            is_initiator = True

    if not is_initiator:
        # Waiter path. Block on the initiator's event with a bounded timeout
        # so a wedged initiator cannot stall callers indefinitely. On a clean
        # set(), re-check that the file actually materialised — the initiator
        # may have failed, in which case the event fires but the file is
        # absent and we surface that to the caller via the classifier as
        # storage_unavailable (503), the right retry hint for "another caller
        # tried, you should try again."
        logging.getLogger(__name__).debug(
            "coalesced waiter on %s", shard_uri,
        )
        completed = event.wait(coalesce_wait_s)
        if not completed:
            logging.getLogger(__name__).warning(
                "download coalescing timeout after %.1fs on %s",
                coalesce_wait_s, shard_uri,
            )
            raise timeout_exc(
                f"timed out after {coalesce_wait_s}s waiting for "
                f"an in-flight download of {shard_uri}"
            )
        if not os.path.exists(path):
            # Initiator's download must have failed; the file is not there.
            # Raise FileNotFoundError (not bare RuntimeError) so the existing
            # classifier branch routes this to storage_unavailable / 503
            # instead of the catch-all ephemeral_error / 500. Customers
            # retrying on 503 get the right transient-failure semantics.
            raise FileNotFoundError(
                f"coalesced download of {shard_uri} did not produce a local file "
                "(initiator likely failed); caller should retry"
            )
        return path

    # Initiator path. Do the actual download, then publish the file via an
    # atomic rename. Cleanup of the in-flight entry runs in `finally` so a
    # failure does NOT leave a stale entry that wedges every future caller
    # as a perpetual waiter — the next caller becomes a fresh initiator on
    # a fresh event and can retry the download.
    #
    # `download_to` streams the GET to `tmp` without buffering the whole
    # object in RAM. The previous `f.write(read_bytes(shard_uri))` pattern
    # OOMed the DP container on a multi-GB shard (a 6 GB Python bytes
    # object inside a 2 GB cgroup limit).
    from adapters.storage.storage import download_to

    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        try:
            download_to(shard_uri, tmp)
            os.replace(tmp, path)
        except BaseException:
            # On any failure before the rename, remove the leftover temp
            # file so a crash mid-write does not leak `.tmp` files into
            # CACHE_DIR. Best-effort: the file may already be gone.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path
    finally:
        # Order matters: clear the registry FIRST so a thread that wakes from
        # `event.wait()` and re-enters this function sees no in-flight entry
        # and becomes a fresh initiator (if the file is still missing).
        # Then set the event so existing waiters unblock and re-check the
        # filesystem. Both steps run on both success and failure paths.
        with inflight_lock:
            inflight.pop(shard_uri, None)
        event.set()
