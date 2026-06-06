"""Per-`(tenant, dataset)` catalog (`list_shards`) cache.

Single home for the TTL-bounded, LRU-capped, generation-checked cache that
`services.query_api.v1_query` and `services.ephemeral_runner.run` previously
duplicated verbatim. Each service owns its own :class:`CatalogCache` instance
(state is NOT shared between services — the two inline copies were independent
and tests rely on that isolation).

Behaviour is byte-for-byte the legacy logic:

  * TTL: an entry is a hit while ``(now - stored_at) < ttl``; expiry re-fetches.
  * Generation counter: bumped on every invalidate; the writer captures the
    generation under the lock before the (lock-free) fetch and refuses to
    install rows if a concurrent invalidate moved the counter — closing the
    "race against concurrent invalidate" install window.
  * LRU bound: hits ``move_to_end``; inserts evict the oldest entries until
    ``len <= max_entries`` (the just-inserted entry is the MRU, never evicted in
    the same call).
  * Single-flight is NOT a contract: N concurrent callers on a cold miss may
    each call the source. ``list_shards`` is cheap and the TTL is short.

The ``list_shards_fn`` and ``now_fn`` are passed PER CALL so each caller's
thin wrapper can resolve those names from its own module namespace at call
time (preserving ``monkeypatch.setattr(module, "list_shards"/"_now", ...)``).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Dict


class CatalogCache:
    """A TTL + LRU + generation-checked cache of `list_shards` results.

    One instance per service. The cache key is ``(tenant, dataset)`` and the
    stored value is ``(stored_at, rows)`` where ``rows`` is returned BY
    REFERENCE — callers must not mutate it.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        # Hardcoded LRU bound. `max_entries` datasets is generous — even a
        # self-host with 10k distinct tenants under one dataset each fits.
        self.max_entries = max_entries
        self._cache: "OrderedDict[tuple[str, str], tuple[float, list]]" = OrderedDict()
        self._lock = threading.Lock()
        # Per-key generation counter. Bumped on every invalidation; the cache
        # writer compares pre-fetch and post-fetch generations under the lock
        # and refuses to install rows older than a concurrent invalidate.
        self._gen: "Dict[tuple[str, str], int]" = {}

    def cached_list_shards(
        self,
        tenant: str,
        dataset: str,
        list_shards_fn: Callable[[str, str], list],
        now_fn: Callable[[], float],
        ttl: float,
        active: bool,
    ) -> list:
        """Return `list_shards_fn(tenant, dataset)`, cached for `ttl` seconds.

        When ``active`` is False (the cache is inactive — SSD tier off or
        TTL=0) every call defers to ``list_shards_fn``. When active, returns a
        fresh-enough cached list or re-fetches on expiry. The cached list is
        returned BY REFERENCE — callers must not mutate it.

        Single-flight is NOT a contract: N concurrent callers on a cold miss
        can each call the source.
        """
        if not active:
            return list_shards_fn(tenant, dataset)
        key = (tenant, dataset)
        now = now_fn()
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and (now - entry[0]) < ttl:
                # LRU bump — move the warm entry to the most-recent end so
                # the bounded-cap eviction prefers cold entries.
                self._cache.move_to_end(key)
                return entry[1]
            # Capture generation under the lock so a concurrent invalidate
            # AFTER this snapshot bumps the counter and our install loses.
            gen_pre = self._gen.get(key, 0)
        # Miss / expired — fetch outside the lock so a slow Postgres does
        # not serialise other readers.
        rows = list_shards_fn(tenant, dataset)
        with self._lock:
            # Generation check: if an invalidate fired during the fetch, the
            # counter has moved past our snapshot. Skip the install — the
            # rows we just fetched are no fresher than the invalidate, and a
            # future caller will re-fetch.
            if self._gen.get(key, 0) != gen_pre:
                return rows
            self._cache[key] = (now, rows)
            self._cache.move_to_end(key)
            # Bounded-cap eviction: drop the oldest entries until we are
            # under the cap. The just-inserted entry is the MRU so it is
            # never the one evicted in the same call.
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return rows

    def invalidate(self, tenant: str, dataset: str) -> bool:
        """Drop the cached entry for `(tenant, dataset)`. Idempotent.

        Bumps the per-key generation counter so a `cached_list_shards` call
        mid-flight (already past the miss-check, fetching rows from the
        source) refuses to install the now-stale rows. Returns True iff an
        entry was actually removed.
        """
        key = (tenant, dataset)
        with self._lock:
            self._gen[key] = self._gen.get(key, 0) + 1
            return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every cached `(tenant, dataset)` entry (test/reset helper)."""
        with self._lock:
            self._cache.clear()
            self._gen.clear()
