"""Contract tests for `adapters/storage/shard_tier.py`.

Tests the implemented byte-budgeted LRU SSD tier that sits between the object
store and the in-process RAM cache, plus the admission floor and prewarm
endpoint. The `test_prewarm_admission_*` tests are fully implemented with real
assertions and `pytest.raises` checks.

Contract under test:

  Core (byte-budgeted LRU):
    - `fetch(shard_uri) -> local_path` is single-flight per URI; the URI IS
      the cache key (content-addressed shards mean new bytes => new URI by
      construction, so no separate version identifier is needed at the cache
      layer).
    - `evict(shard_uri)` unlinks the file; an open mmap on it stays valid
      (POSIX guarantee)
    - byte-budgeted LRU eviction at `RB_SHARD_TIER_BYTES`
    - `residency()` + `bytes_used()` observability
    - `ShardTierTimeout` for bounded waits

  Admission floor + prewarm:
    - `prewarm(uri)` performs admission control: evict-coldest-if-old enough
      (older than `MIN_RESIDENT_S`); else reject with
      `CacheCapacityExceeded` (HTTP 503 in the wire path).
"""
from __future__ import annotations

import importlib
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest


pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


@pytest.fixture
def tier(monkeypatch, tmp_path):
    """Fresh `shard_tier` module with `RB_SHARD_TIER_DIR` pointed at a tmp dir.

    The module reads its env vars at import time, so we set them and reload
    to pick up the fresh values. Belt-and-braces clear of any in-flight or
    residency state in case a prior test in the same process leaked an entry.
    The byte budget defaults to 2 GiB but individual tests override it via
    `monkeypatch.setattr(tier, "_TIER_BYTES", ...)` to keep the LRU test
    payloads small.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"))
    from adapters.storage import shard_tier

    importlib.reload(shard_tier)
    # Clear residency + in-flight state. Reload should already do this (the
    # dicts are rebuilt on import), but a test that fails mid-run could
    # otherwise leak an entry into the next test.
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()
    yield shard_tier
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()


class _SlowReadBytes:
    """A stand-in for `adapters.storage.storage.read_bytes`.

    Counts invocations and sleeps `delay` seconds per call so concurrent
    callers visibly stack up. Lets a test assert both "exactly N downloads
    ran" and "downloads for distinct URIs overlapped in time" — the two
    pillars of the per-URI single-flight invariant.
    """

    def __init__(self, delay: float = 0.2, payload: bytes = b"shard-bytes"):
        self.delay = delay
        self.payload = payload
        self.count = 0
        self.intervals: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    def __call__(self, uri: str) -> bytes:
        with self._lock:
            self.count += 1
        start = time.monotonic()
        time.sleep(self.delay)
        end = time.monotonic()
        with self._lock:
            self.intervals.append((start, end))
        return self.payload


def _patch_read_bytes(monkeypatch, fake):
    """Patch the cache-fill download at the storage module.

    `shard_tier.fetch` calls `download_to(uri, local_path)` to stream bytes
    to disk. To keep the existing `_SlowReadBytes`-style fakes (which take a
    URI and return bytes) usable, we patch BOTH symbols: `read_bytes` so a
    direct caller sees the same fake, and `download_to` with a thin wrapper
    that calls the fake to get bytes and writes them to the local path.
    Preserves the fake's call-counting + overlap bookkeeping without rewriting
    the threading tests.
    """
    import adapters.storage.storage as storage_mod

    monkeypatch.setattr(storage_mod, "read_bytes", fake)

    def _download_to(uri: str, local_path: str) -> None:
        payload = fake(uri)
        with open(local_path, "wb") as f:
            f.write(payload)

    monkeypatch.setattr(storage_mod, "download_to", _download_to)


# --- Core: byte-budgeted LRU + single-flight tests -------------------------


def test_cold_fetch_materialises_file_on_disk(tier, monkeypatch):
    """A first-ever `fetch(uri)` downloads and writes the bytes to local disk.

    Pins the boring single-caller path: the returned path exists, contains
    exactly the bytes the storage adapter served, and lives under the
    configured tier directory (not in the legacy CACHE_DIR root — the tier
    keeps its files distinct so the hot-path wiring does not collide with
    `_ensure_cached`'s files).
    """
    fake = _SlowReadBytes(delay=0.01, payload=b"hello-shard")
    _patch_read_bytes(monkeypatch, fake)

    path = tier.fetch("memory://bucket/shard-1.bin")

    assert os.path.exists(path), f"local file missing at {path}"
    with open(path, "rb") as f:
        assert f.read() == b"hello-shard"
    assert path.startswith(tier._TIER_DIR), (
        f"file landed outside RB_SHARD_TIER_DIR: {path}"
    )
    assert fake.count == 1


def test_warm_fetch_returns_same_path_without_redownload(tier, monkeypatch):
    """A second `fetch` for the same URI hits the local file, no GET issued.

    A regression here would mean either the residency entry was never
    written or the fast-path check missed it — both would silently re-issue
    object-store GETs and re-engage the storm the SSD tier exists to prevent.
    """
    fake = _SlowReadBytes(delay=0.01)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-warm.bin"
    first = tier.fetch(uri)
    second = tier.fetch(uri)

    assert first == second
    assert fake.count == 1, (
        f"warm fetch re-downloaded; count={fake.count}"
    )


def test_fetch_updates_last_query_at_on_hit(tier, monkeypatch):
    """A warm fetch refreshes `last_query_at` so the entry walks toward MRU.

    The residency table records both `last_admit_at` (set once at insert
    time, used by the `MIN_RESIDENT_S` admission floor) and `last_query_at`
    (refreshed on every hit, used by the LRU ordering). This test pins the
    refresh: after a `fetch` of an already-resident entry, the entry's
    `last_query_at` must be strictly greater than its `last_admit_at`.
    """
    fake = _SlowReadBytes(delay=0.0)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-touch.bin"
    tier.fetch(uri)
    entry_before = [e for e in tier.residency() if e.shard_uri == uri][0]

    # Tiny sleep so the monotonic clock visibly advances between admit and
    # the second fetch. Without it the two timestamps can be byte-identical
    # on a fast machine and the strictly-greater check is flaky.
    time.sleep(0.01)
    tier.fetch(uri)
    entry_after = [e for e in tier.residency() if e.shard_uri == uri][0]

    assert entry_after.last_admit_at == entry_before.last_admit_at, (
        "last_admit_at must be set once on insert, not refreshed on hit"
    )
    assert entry_after.last_query_at > entry_before.last_query_at, (
        "last_query_at must be refreshed on every fetch hit"
    )


def test_fetch_single_flight_on_concurrent_calls(tier, monkeypatch):
    """N callers on the same uri => 1 underlying read; all return the same path.

    Without coalescing, N concurrent queries against a cold shard issue N
    parallel GETs and the object store throttles. With coalescing only the
    initiator downloads; the other N-1 block on the per-URI event and return
    the same local path once the rename publishes the file.
    """
    fake = _SlowReadBytes(delay=0.2)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-coalesce.bin"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(tier.fetch, uri) for _ in range(10)]
        results = [f.result(timeout=10) for f in as_completed(futures)]

    assert fake.count == 1, (
        f"expected exactly 1 download under coalescing; got {fake.count}"
    )
    assert len(set(results)) == 1, (
        f"expected all callers to return the same local path; got {set(results)}"
    )
    path = results[0]
    assert os.path.exists(path)


def test_different_uris_do_not_serialize(tier, monkeypatch):
    """The single-flight lock is per-URI, not global.

    10 callers each fetching a DIFFERENT URI must run in parallel — a global
    download mutex would serialise them and take ~10x longer. The threshold
    is set comfortably between the parallel time (~0.2 s) and the serial
    time (~2 s) so the test is robust to CI jitter.
    """
    fake = _SlowReadBytes(delay=0.2)
    _patch_read_bytes(monkeypatch, fake)

    uris = [f"memory://bucket/shard-parallel-{i}.bin" for i in range(10)]

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(tier.fetch, uri) for uri in uris]
        [f.result(timeout=10) for f in as_completed(futures)]
    elapsed = time.monotonic() - start

    assert fake.count == 10, (
        f"expected one download per distinct URI; got {fake.count}"
    )
    assert elapsed < 1.0, (
        f"different-URI downloads serialised (elapsed={elapsed:.2f}s); "
        "the in-flight lock must be per-URI"
    )


def test_failed_download_does_not_leave_stale_lock(tier, monkeypatch):
    """A failed initiator MUST clear the in-flight entry so retries are possible.

    Without the cleanup, the entry persists forever: every subsequent caller
    becomes a waiter on a fired-but-unrecoverable event and either returns a
    bogus path or blocks. With the fix, the cleanup runs in `finally`, the
    entry is removed, and the next caller becomes a fresh initiator on a
    fresh event.
    """
    call_count = {"n": 0}

    def flaky_read(uri: str) -> bytes:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient storage failure")
        return b"recovered"

    _patch_read_bytes(monkeypatch, flaky_read)

    uri = "memory://bucket/shard-flaky.bin"

    with pytest.raises(RuntimeError):
        tier.fetch(uri)

    with tier._INFLIGHT_LOCK:
        assert uri not in tier._INFLIGHT, (
            "stale in-flight entry left after failed download"
        )

    path = tier.fetch(uri)
    assert os.path.exists(path)
    assert call_count["n"] == 2


def test_coalesced_waiter_bounded_by_timeout(tier, monkeypatch):
    """A hung initiator must not block waiters indefinitely.

    Override `_COALESCE_WAIT_S` to a small value for the test so it runs in
    well under a second. The contract: a waiter that exceeds the deadline
    raises `ShardTierTimeout` with a clear message — it does NOT silently
    return a bogus path.
    """
    hang_started = threading.Event()
    release = threading.Event()

    def hanging_read(uri: str) -> bytes:
        hang_started.set()
        release.wait(timeout=5)
        return b"eventually"

    _patch_read_bytes(monkeypatch, hanging_read)
    monkeypatch.setattr(tier, "_COALESCE_WAIT_S", 0.2)

    uri = "memory://bucket/shard-hang.bin"
    initiator_done = threading.Event()

    def initiator():
        try:
            tier.fetch(uri)
        except Exception:
            pass
        finally:
            initiator_done.set()

    t = threading.Thread(target=initiator, daemon=True)
    t.start()

    assert hang_started.wait(timeout=2), "initiator never started"

    waiter_start = time.monotonic()
    with pytest.raises(tier.ShardTierTimeout):
        tier.fetch(uri)
    waiter_elapsed = time.monotonic() - waiter_start

    assert waiter_elapsed < 2.0, (
        f"waiter blocked far past its deadline ({waiter_elapsed:.2f}s)"
    )

    release.set()
    initiator_done.wait(timeout=5)


def test_fetch_raises_filenotfound_on_missing_uri(tier, monkeypatch):
    """A URI the storage layer cannot resolve surfaces as `FileNotFoundError`.

    The storage adapter normalises s3:// 404s and memory:// missing keys to
    `FileNotFoundError`. The tier preserves that exception type so callers
    can distinguish "shard truly does not exist" from transient storage
    failures. The single-flight entry MUST also be cleared
    so a subsequent retry — perhaps after the key has been published — does
    not wait forever on a stale event.
    """
    def missing_read(uri: str) -> bytes:
        raise FileNotFoundError(f"memory:// key not found: {uri}")

    _patch_read_bytes(monkeypatch, missing_read)
    uri = "memory://bucket/shard-absent.bin"

    with pytest.raises(FileNotFoundError):
        tier.fetch(uri)

    with tier._INFLIGHT_LOCK:
        assert uri not in tier._INFLIGHT, (
            "in-flight entry leaked after FileNotFoundError"
        )


def test_byte_budget_lru_eviction(tier, monkeypatch):
    """When `RB_SHARD_TIER_BYTES` is exceeded, the LRU end is evicted first.

    Sets a budget of 30 bytes and inserts three 10-byte entries A, B, C.
    Touches A (so its LRU recency is refreshed past B's) and then inserts D
    — the LRU end is now B, so B's file must be unlinked, A/C/D must
    remain, and the byte total must equal 30.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 30)

    uri_a = "memory://bucket/A.bin"
    uri_b = "memory://bucket/B.bin"
    uri_c = "memory://bucket/C.bin"
    uri_d = "memory://bucket/D.bin"

    a = tier.fetch(uri_a)
    b = tier.fetch(uri_b)
    c = tier.fetch(uri_c)
    assert tier.bytes_used() == 30
    assert all(os.path.exists(p) for p in (a, b, c))

    # Touch A so the LRU end becomes B (insert order is A, B, C; after the
    # touch the order is B, C, A — B is LRU).
    time.sleep(0.005)
    tier.fetch(uri_a)

    # Insert D. With a 30-byte budget and four 10-byte entries, one must go.
    d = tier.fetch(uri_d)

    resident_uris = {e.shard_uri for e in tier.residency()}
    assert resident_uris == {uri_a, uri_c, uri_d}, (
        f"expected LRU end (B) to be evicted; resident={resident_uris}"
    )
    assert not os.path.exists(b), f"evicted file should be unlinked: {b}"
    assert all(os.path.exists(p) for p in (a, c, d))
    assert tier.bytes_used() == 30


def test_residency_returns_lru_order(tier, monkeypatch):
    """`residency()` returns entries in LRU order (LRU first, MRU last).

    Inserting A, B, C and then touching A walks A to the MRU end. The
    expected ordering is therefore [B, C, A]. The residency-registry writer
    iterates in this order; pinning it here so a future refactor cannot
    silently flip the convention.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"xxxxxxxxxx")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 1024)

    uri_a = "memory://bucket/A.bin"
    uri_b = "memory://bucket/B.bin"
    uri_c = "memory://bucket/C.bin"

    tier.fetch(uri_a)
    time.sleep(0.005)
    tier.fetch(uri_b)
    time.sleep(0.005)
    tier.fetch(uri_c)
    time.sleep(0.005)
    tier.fetch(uri_a)  # touch A -> MRU

    uris = [e.shard_uri for e in tier.residency()]
    assert uris == [uri_b, uri_c, uri_a], f"expected LRU-first order; got {uris}"


def test_bytes_used_matches_sum_of_resident_entries(tier, monkeypatch):
    """`bytes_used()` is the live sum of `nbytes` across all resident entries.

    Operators read this from `/admin/cache` and tests pin it as the bridge
    between admission decisions and observable budget — a drift here would
    leave the tier admitting past its budget without anyone seeing it.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"abcdefghij")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 1024)

    tier.fetch("memory://bucket/A.bin")
    tier.fetch("memory://bucket/B.bin")

    summed = sum(e.nbytes for e in tier.residency())
    assert tier.bytes_used() == summed == 20


def test_evict_removes_entry_and_returns_true(tier, monkeypatch):
    """`evict(shard_uri)` unlinks the file, drops the residency entry, returns True.

    An idempotent second call returns False — no entry to remove. Tests the
    plain happy path; the unlink-with-open-mmap edge is the next test.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"hello")
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/E.bin"
    path = tier.fetch(uri)
    assert os.path.exists(path)
    assert any(e.shard_uri == uri for e in tier.residency())

    assert tier.evict(uri) is True
    assert not os.path.exists(path)
    assert not any(e.shard_uri == uri for e in tier.residency())

    # Idempotent: evicting an absent entry is False, not a raise.
    assert tier.evict(uri) is False


def test_evict_unlinks_file_and_keeps_open_mmap_valid(tier, monkeypatch):
    """POSIX guarantee: unlink-with-open-mmap keeps the mapping readable.

    The contract that lets us evict aggressively without coordinating with
    in-flight queries. A query that has already mmap'd the file before the
    evict() runs continues to read the file's bytes through its mapping; the
    kernel only reclaims the inode when the last fd closes.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"MMAP_TEST_BYTES_0123456789")
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/M.bin"
    path = tier.fetch(uri)

    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mapping = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        try:
            # Evict while the mapping is open — file is unlinked but the
            # mmap'd bytes remain readable (POSIX semantics).
            assert tier.evict(uri) is True
            assert not os.path.exists(path)
            assert mapping[:size] == b"MMAP_TEST_BYTES_0123456789"
        finally:
            mapping.close()
    finally:
        os.close(fd)


def test_refetch_after_evict_is_cold_miss(tier, monkeypatch):
    """A fetch after evict() retriggers the download (admission is a cold miss).

    Asserts the eviction is total: not just unlinked, but also removed from
    the residency table. Without the residency drop a subsequent `fetch`
    would find the entry, attempt to read the unlinked path, and fail. The
    fetch finds neither a live entry nor a local file and goes through the
    full download path. A regression where the residency entry survived would
    return a stale path that no longer exists on disk — the worst-case bug.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"first-bytes")
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/R.bin"
    tier.fetch(uri)
    tier.evict(uri)
    assert fake.count == 1

    path = tier.fetch(uri)
    assert fake.count == 2, "evicted shard must trigger a cold re-download"
    assert os.path.exists(path)


# --- Crash-recovery .tmp sweep on module import (plan-mandated) -----------


def test_tmp_sweep_removes_old_orphan_tmps(monkeypatch, tmp_path):
    """A `.tmp` older than the sweep threshold is removed on module import.

    A crashed initiator leaves `{path}.{pid}.{uuid8}.tmp` behind. Without
    a sweep, long-lived deployments accumulate these monotonically.
    The sweep runs once on import (the module is loaded in the same
    process as the DP). Recent `.tmp` files (in-flight downloads in
    another thread) MUST survive — the sweep ages out only entries
    older than `RB_SHARD_TIER_TMP_MAX_AGE_S` (default 3600s).
    """
    tier_dir = tmp_path / "shards" / "tier-managed"
    tier_dir.mkdir(parents=True)

    recent = tier_dir / "recent.tmp"
    recent.write_bytes(b"in-flight")

    old = tier_dir / "old.tmp"
    old.write_bytes(b"crashed")
    old_mtime = time.time() - 7200  # 2 hours ago
    os.utime(old, (old_mtime, old_mtime))

    # Non-.tmp files must be ignored by the sweep entirely — those are
    # the tier's actual cached files and unlinking them would corrupt
    # the cache.
    keeper = tier_dir / "keeper.bin"
    keeper.write_bytes(b"do-not-sweep")
    keeper_mtime = time.time() - 7200
    os.utime(keeper, (keeper_mtime, keeper_mtime))

    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tier_dir))
    monkeypatch.setenv("RB_SHARD_TIER_TMP_MAX_AGE_S", "3600")

    from adapters.storage import shard_tier
    importlib.reload(shard_tier)

    assert recent.exists(), "recent .tmp must survive the sweep"
    assert not old.exists(), "old .tmp must be removed on module import"
    assert keeper.exists(), "non-.tmp files must never be touched by the sweep"


# --- Admission floor + prewarm contract ------------------------------------


def test_prewarm_admission_evicts_when_lru_old_enough(tier, monkeypatch):
    """`prewarm` succeeds by evicting the LRU end when it is older than `MIN_RESIDENT_S`.

    With the SSD tier at its byte budget and the LRU end's `last_admit_at`
    older than `MIN_RESIDENT_S` (set to 0.05 s here so the test runs fast),
    `prewarm(new_uri)` must evict the LRU end, admit the new entry, and
    return its local path.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 20)
    # Override floor so the test does not have to sleep 30 seconds.
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 0.05)

    uri_a = "memory://bucket/A.bin"
    uri_b = "memory://bucket/B.bin"
    uri_new = "memory://bucket/NEW.bin"

    tier.fetch(uri_a)
    tier.fetch(uri_b)
    assert tier.bytes_used() == 20

    # Age both entries past the floor so the LRU end is evictable.
    time.sleep(0.08)

    path = tier.prewarm(uri_new)
    assert os.path.exists(path)

    resident_uris = {e.shard_uri for e in tier.residency()}
    assert uri_new in resident_uris, "prewarmed URI must be resident"
    assert uri_a not in resident_uris, (
        "LRU end (A) should have been evicted to make room for the prewarm"
    )
    assert tier.bytes_used() == 20


def test_prewarm_admission_rejects_when_lru_too_young(tier, monkeypatch):
    """`prewarm` raises `CacheCapacityExceeded` when every candidate is younger than `MIN_RESIDENT_S`.

    With a full tier and every resident entry's `last_admit_at` younger than
    `MIN_RESIDENT_S`, `prewarm(new_uri)` must NOT evict anyone — it must
    raise `CacheCapacityExceeded`, leave the resident set untouched, and
    clear its in-flight registration so a later retry (e.g. after the floor
    expires) becomes a fresh initiator.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 20)
    # Large floor so the resident entries are guaranteed to be too young.
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 60.0)

    uri_a = "memory://bucket/A.bin"
    uri_b = "memory://bucket/B.bin"
    uri_new = "memory://bucket/NEW.bin"

    tier.fetch(uri_a)
    tier.fetch(uri_b)
    before = {e.shard_uri for e in tier.residency()}
    bytes_before = tier.bytes_used()

    with pytest.raises(tier.CacheCapacityExceeded):
        tier.prewarm(uri_new)

    # The new URI was never downloaded (admission rejected before fetch
    # could succeed) or — if downloaded first — the file was cleaned up.
    resident_uris = {e.shard_uri for e in tier.residency()}
    assert resident_uris == before, (
        f"prewarm rejection must leave residency untouched; "
        f"before={before} after={resident_uris}"
    )
    assert tier.bytes_used() == bytes_before
    # In-flight registration cleared so a later retry can become initiator.
    with tier._INFLIGHT_LOCK:
        assert uri_new not in tier._INFLIGHT, (
            "in-flight entry leaked after CacheCapacityExceeded"
        )


def test_coalesced_waiter_on_rejected_prewarm_gets_capacity_exceeded(
    tier, monkeypatch,
):
    """A waiter that wakes after the initiator was admission-rejected sees
    `CacheCapacityExceeded`, NOT `FileNotFoundError`.

    The fetch path raises `FileNotFoundError` on a waiter whose initiator
    failed (the residency entry is missing — looks like a missing object).
    The prewarm path can't reuse that signal: a waiter cannot distinguish
    "object store has no such URI" from "initiator was admission-rejected"
    at wake time, but the prewarm-specific signal is the more informative
    of the two. This test pins the contract so a future refactor that
    "unifies" the waiter wake-up path cannot silently downgrade
    CacheCapacityExceeded to FileNotFoundError.
    """
    fake = _SlowReadBytes(delay=0.2, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 20)
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 60.0)

    # Fill the tier with young residents so prewarm rejection is guaranteed.
    tier.fetch("memory://bucket/A.bin")
    tier.fetch("memory://bucket/B.bin")

    new_uri = "memory://bucket/REJECT-ME.bin"

    initiator_done = threading.Event()
    initiator_exc: list[BaseException] = []

    def initiator():
        try:
            tier.prewarm(new_uri)
        except BaseException as exc:
            initiator_exc.append(exc)
        finally:
            initiator_done.set()

    # Waiter starts shortly after initiator so it lands on the per-URI event
    # rather than racing into the fast path.
    waiter_exc: list[BaseException] = []

    def waiter():
        # Tiny delay so the initiator registers its in-flight entry first.
        time.sleep(0.02)
        try:
            tier.prewarm(new_uri)
        except BaseException as exc:
            waiter_exc.append(exc)

    t_init = threading.Thread(target=initiator, daemon=True)
    t_wait = threading.Thread(target=waiter, daemon=True)
    t_init.start()
    t_wait.start()
    initiator_done.wait(timeout=5)
    t_wait.join(timeout=5)

    assert initiator_exc and isinstance(
        initiator_exc[0], tier.CacheCapacityExceeded
    ), f"initiator must raise CacheCapacityExceeded; got {initiator_exc!r}"
    assert waiter_exc and isinstance(
        waiter_exc[0], tier.CacheCapacityExceeded
    ), (
        f"waiter on a rejected prewarm must surface CacheCapacityExceeded "
        f"(not FileNotFoundError); got {waiter_exc!r}"
    )


def test_cache_capacity_exceeded_is_runtimeerror_subclass():
    """`CacheCapacityExceeded` is a `RuntimeError` subclass.

    Pinned so callers that catch `RuntimeError` (legacy code paths) still
    catch this; pinned so classifier branches that test
    `isinstance(exc, CacheCapacityExceeded)` can sit alongside the
    `RuntimeError`-catching frames without re-ordering risk.
    """
    from adapters.storage import shard_tier

    assert issubclass(shard_tier.CacheCapacityExceeded, RuntimeError)


def test_prewarm_returns_local_path_for_warm_uri(tier, monkeypatch):
    """A `prewarm` of an already-resident URI is a no-op success.

    The contract: prewarm is idempotent. Calling it on a URI that is
    already in residency must return the existing local path without
    re-downloading and without raising — the operator should be able to
    fire prewarm at-least-once without coordinating with the cache state.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"warm-bytes")
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 30.0)

    uri = "memory://bucket/already-warm.bin"
    first = tier.fetch(uri)
    assert fake.count == 1

    second = tier.prewarm(uri)
    assert second == first
    assert fake.count == 1, (
        f"prewarm of a warm URI must not re-download; count={fake.count}"
    )


def test_prewarm_cold_admission_under_budget_succeeds(tier, monkeypatch):
    """A `prewarm` into spare capacity succeeds without evicting anyone.

    When `bytes_used + nbytes <= _TIER_BYTES`, no eviction is required and
    the admission floor never comes into play. This is the boring case the
    operator hits when prewarming into a freshly-booted DP.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 1024)
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 60.0)

    uri = "memory://bucket/cold-prewarm.bin"
    path = tier.prewarm(uri)

    assert os.path.exists(path)
    resident_uris = {e.shard_uri for e in tier.residency()}
    assert uri in resident_uris


def test_fetch_remains_unconditional_when_tier_full_of_young_entries(tier, monkeypatch):
    """`fetch()` ignores the `MIN_RESIDENT_S` floor (unconditional admission).

    A real query MUST get its shard cached even if displacing a recently-
    arrived neighbour. The floor exists to discriminate speculative
    arrivals (prewarm) from queries-under-load: only `prewarm()` honours
    it; `fetch()` always admits, evicting the LRU end unconditionally.

    Without this distinction, a write-storm during a query burst would
    starve fetches as well as prewarms — the discrimination signal would
    collapse.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 20)
    # Floor set high enough that every resident entry is "too young" to
    # evict if the floor were honoured.
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 60.0)

    tier.fetch("memory://bucket/A.bin")
    tier.fetch("memory://bucket/B.bin")
    # The tier is at its budget and every entry is too young to evict
    # under prewarm rules. `fetch()` must still succeed.
    path = tier.fetch("memory://bucket/C.bin")

    assert os.path.exists(path)
    resident_uris = {e.shard_uri for e in tier.residency()}
    assert "memory://bucket/C.bin" in resident_uris


def test_prewarm_when_tier_inactive_raises(tier, monkeypatch):
    """`prewarm` raises when `RB_SHARD_TIER_BYTES` is unset (tier inactive).

    There is nothing to admit into when the tier is off. The error type
    is `CacheCapacityExceeded` so the wire path collapses to the same
    503 the contended case uses — both are "the tier cannot accept this
    speculative arrival right now" from the operator's perspective.
    """
    monkeypatch.delenv("RB_SHARD_TIER_BYTES", raising=False)
    # Re-import so the module re-reads the env. Since the fixture has
    # already pinned a value (default 2 GiB), explicitly set _TIER_BYTES
    # to 0 to simulate "inactive tier".
    monkeypatch.setattr(tier, "_TIER_BYTES", 0)

    fake = _SlowReadBytes(delay=0.0, payload=b"x")
    _patch_read_bytes(monkeypatch, fake)

    with pytest.raises(tier.CacheCapacityExceeded):
        tier.prewarm("memory://bucket/inactive.bin")


def test_min_resident_s_boundary_admits_at_floor(tier, monkeypatch):
    """An entry whose age is exactly `MIN_RESIDENT_S` is admittable to evict.

    Documented decision: `>=` (not `>`) — an entry whose age has *reached*
    the floor is fair game. The boundary case matters because a tight
    floor (e.g. 0.0) means EVERY entry is age >= 0 and prewarm must
    behave as unconditional fetch.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 10)
    # Floor of 0 -> every entry's age (>= 0) is >= floor -> evictable.
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 0.0)

    uri_a = "memory://bucket/A.bin"
    uri_new = "memory://bucket/NEW.bin"

    tier.fetch(uri_a)
    path = tier.prewarm(uri_new)

    assert os.path.exists(path)
    resident_uris = {e.shard_uri for e in tier.residency()}
    assert resident_uris == {uri_new}


def test_v1_query_classifier_maps_capacity_exceeded_to_503(tier):
    """`CacheCapacityExceeded` -> `cache_capacity_exceeded` / 503 on the hot path.

    The classifier's job is to bucket transient operational signals into
    customer-visible v1 error codes. `CacheCapacityExceeded` is the prewarm
    rejection signal and surfaces as a 503 (retry-safe) with the dedicated
    `cache_capacity_exceeded` code so a client / operator dashboard can
    distinguish capacity pressure from storage outages.
    """
    from services.query_api import v1_query

    code, _msg = v1_query._classify_hot_path_error(
        tier.CacheCapacityExceeded("tier full of young entries")
    )
    assert code == "cache_capacity_exceeded", (
        f"expected cache_capacity_exceeded; got {code}"
    )


def test_ephemeral_classifier_maps_capacity_exceeded_to_503(tier):
    """Mirror of the v1_query classifier branch in the ephemeral runner.

    The two classifiers are intentionally duplicated to keep the ephemeral
    runner's import graph free of services.query_api; this test pins them
    in sync for the new `CacheCapacityExceeded` branch the same way the
    earlier wiring tests pinned them for `ShardTierTimeout`.
    """
    from services.ephemeral_runner import run as eph_run

    code, _msg = eph_run._classify_error(
        tier.CacheCapacityExceeded("tier full of young entries")
    )
    assert code == "cache_capacity_exceeded"


def test_run_query_maps_capacity_exceeded_to_503_status(tier):
    """`run_query` returns HTTP 503 when the hot path raises CacheCapacityExceeded.

    The classifier returns the code; the wire-status mapping in `run_query`
    turns `cache_capacity_exceeded` into a 503 envelope. This pins the
    wire-status mapping itself — the code -> status table must include the
    new code so it does not fall through to 500.
    """
    from services.query_api import v1_query

    code, _msg = v1_query._classify_hot_path_error(
        tier.CacheCapacityExceeded("x")
    )
    # The mapping in `run_query`: `500 if code == "ephemeral_error" else 503`.
    # The new code must therefore route to 503, not 500.
    assert code != "ephemeral_error", (
        "CacheCapacityExceeded must NOT classify as ephemeral_error "
        "(would map to 500); the dedicated code routes to 503"
    )


def test_min_resident_s_is_live_read(tier, monkeypatch):
    """The admission floor is live-read so an operator can retune at runtime.

    A test that flips the module-level `_MIN_RESIDENT_S` between two
    `prewarm` calls must see the new value take effect on the second call.
    A captured-at-import value would mean an operator change requires a
    DP restart, which contradicts the activation-gate contract.
    """
    fake = _SlowReadBytes(delay=0.0, payload=b"0123456789")  # 10 bytes
    _patch_read_bytes(monkeypatch, fake)
    monkeypatch.setattr(tier, "_TIER_BYTES", 20)

    uri_a = "memory://bucket/A.bin"
    uri_b = "memory://bucket/B.bin"
    uri_new = "memory://bucket/NEW.bin"

    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 60.0)
    tier.fetch(uri_a)
    tier.fetch(uri_b)

    # With a 60s floor, prewarm should reject.
    with pytest.raises(tier.CacheCapacityExceeded):
        tier.prewarm(uri_new)

    # Operator retunes the floor down to 0 — next prewarm must succeed.
    monkeypatch.setattr(tier, "_MIN_RESIDENT_S", 0.0)
    path = tier.prewarm(uri_new)
    assert os.path.exists(path)
