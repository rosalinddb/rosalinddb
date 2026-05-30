"""Single-flight coalescing for `_ensure_cached` in `services/query_api/v1_query.py`.

Pins the contract for per-URI single-flight download coalescing. The bug it
addresses: a concurrent burst of queries targeting an uncached shard each issue
their own object-store GET. With a 6 GB shard and 10 concurrent virtual users,
the object store sees 10 parallel GETs, throttles (`TooManyRequests`), and the
DP returns `storage_unavailable`. The cache never warms; the system thrashes.

The fix is the textbook "single-flight" pattern: when N threads simultaneously
request the same uncached URI, only ONE actually downloads; the other N-1
block on a per-URI `threading.Event` and return the same local path once the
download completes. The lock is per-URI, not global, so independent URIs do
not serialise. Failures clear the in-flight entry so a transient error does
not poison the URI forever, and waiters are bounded by `RB_DOWNLOAD_COALESCE_WAIT_S`
so a hung initiator cannot wedge every waiter indefinitely.

The tests pin the contract by counting `read_bytes` invocations and by timing
the parallel-vs-serial gap. The implementation lives in `v1_query._ensure_cached`
plus `_INFLIGHT_DOWNLOADS` state.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest


pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


@pytest.fixture
def v1q(monkeypatch, tmp_path):
    """Import `v1_query` with `CACHE_DIR` pointed at a clean tmp dir.

    The module reads `CACHE_DIR` at import time (module-level constant), so
    we set the env var and reload to pick up the fresh value. Also wipes any
    stray in-flight state from prior tests in the same process — the dict is
    module-global and persists across tests otherwise.
    """
    import importlib

    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    import services.query_api.v1_query as v1_query

    importlib.reload(v1_query)
    # Belt-and-braces: clear in-flight state. Reload should already do this
    # (the dict is rebuilt on import), but a test that fails mid-run could
    # otherwise leak an entry into the next test.
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()
    yield v1_query
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()


class _SlowReadBytes:
    """A stand-in for `adapters.storage.storage.read_bytes`.

    Counts invocations and sleeps `delay` seconds per call so concurrent
    callers visibly stack up. Records the per-call start/end timestamps so a
    test can assert that two calls overlapped in time (i.e. ran in parallel).
    The internal lock guards `count` / `intervals` so concurrent threads
    cannot race the bookkeeping itself.
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
    """Patch the cache-fill download path at the storage module.

    `_ensure_cached` uses `download_to(uri, local_path)` to stream bytes to
    disk. To keep the existing `_SlowReadBytes`-style fakes (which take just a
    URI and return bytes) usable, we patch BOTH symbols:

      - `read_bytes` is kept patched so a direct caller (sidecar load,
        legacy code path) sees the same fake.
      - `download_to` is replaced with a thin wrapper that calls the
        fake to get bytes and writes them to the local path. The wrapper
        preserves the "1 call per URI" counting + "calls overlap in
        time" property the fake exposes via `_SlowReadBytes`.

    The kludge is intentional: keeping the tests' fake-shape stable
    avoids rewriting six sequenced threading tests for a refactor that
    doesn't change the test's actual contract. Tests that pin the
    streaming behaviour live in `tests/unit/test_storage_download_to.py`.
    """
    import adapters.storage.storage as storage_mod

    monkeypatch.setattr(storage_mod, "read_bytes", fake)

    def _download_to(uri: str, local_path: str) -> None:
        # The fake `_SlowReadBytes` does its bookkeeping when called as
        # `fake(uri)` and returns the payload bytes — we then commit them
        # to disk so the post-condition (a file at `local_path`) holds.
        payload = fake(uri)
        with open(local_path, "wb") as f:
            f.write(payload)

    monkeypatch.setattr(storage_mod, "download_to", _download_to)


# --- tests ----------------------------------------------------------------


def test_concurrent_callers_trigger_one_download(v1q, monkeypatch):
    """N=10 threads on the same uncached URI ⇒ exactly 1 `read_bytes` call.

    Without coalescing each thread races to download independently and the
    object store sees 10 parallel GETs — the symptom that took down MinIO
    in the mmap bench. With coalescing only the initiator downloads; the
    other 9 block on the per-URI event and reuse the same local file.
    """
    fake = _SlowReadBytes(delay=0.2)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-coalesce-1.faiss"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(v1q._ensure_cached, uri) for _ in range(10)]
        results = [f.result(timeout=10) for f in as_completed(futures)]

    assert fake.count == 1, (
        f"expected exactly 1 download for the coalesced URI; got {fake.count}"
    )
    assert len(set(results)) == 1, (
        f"expected all callers to return the same local path; got {set(results)}"
    )


def test_concurrent_callers_get_same_local_path(v1q, monkeypatch):
    """The shared local path actually exists on disk after the burst.

    A stricter form of the previous test: not just "same string", but the
    file the path names is materialised and non-empty. This rules out the
    failure mode where the initiator's atomic rename never ran (e.g. the
    waiter returned the *intended* path before the file existed).
    """
    fake = _SlowReadBytes(delay=0.15, payload=b"some-bytes-here")
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-coalesce-2.faiss"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(v1q._ensure_cached, uri) for _ in range(10)]
        results = [f.result(timeout=10) for f in as_completed(futures)]

    path = results[0]
    assert all(r == path for r in results)
    assert os.path.exists(path), f"local cache file missing at {path}"
    with open(path, "rb") as f:
        assert f.read() == b"some-bytes-here"


def test_sequential_callers_after_completion_dont_re_download(v1q, monkeypatch):
    """A caller arriving AFTER the download completes hits the local cache.

    This is the boring single-caller path, but pinned here so a regression
    that, say, never deletes the in-flight entry on success (and so makes
    every subsequent caller a "waiter" with nothing to wait for) is caught.
    """
    fake = _SlowReadBytes(delay=0.05)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-coalesce-3.faiss"

    first = v1q._ensure_cached(uri)
    second = v1q._ensure_cached(uri)

    assert first == second
    assert fake.count == 1, (
        f"expected only the first call to download; got count={fake.count}"
    )


def test_failed_download_does_not_leave_stale_lock(v1q, monkeypatch):
    """A failed initiator MUST clear the in-flight entry so retries are possible.

    Without the cleanup, the entry persists forever: every subsequent caller
    becomes a waiter on a fired-but-unrecoverable event and either returns a
    bogus path or blocks. With the fix, the cleanup runs in `finally`, the
    entry is removed, and the next caller can attempt the download afresh
    (a fresh initiator on a fresh event).
    """
    call_count = {"n": 0}

    def flaky_read(uri: str) -> bytes:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient storage failure")
        return b"recovered-payload"

    _patch_read_bytes(monkeypatch, flaky_read)

    uri = "memory://bucket/shard-coalesce-4.faiss"

    with pytest.raises(RuntimeError):
        v1q._ensure_cached(uri)

    # The in-flight entry must have been cleared on the failure path. If it
    # weren't, the next caller would either wait forever on a stale event or
    # return a non-existent path.
    with v1q._INFLIGHT_DOWNLOADS_LOCK:
        assert uri not in v1q._INFLIGHT_DOWNLOADS, (
            "stale in-flight entry left after failed download"
        )

    # The retry succeeds — the new caller is a fresh initiator.
    path = v1q._ensure_cached(uri)
    assert os.path.exists(path)
    assert call_count["n"] == 2


def test_different_uris_do_not_serialize(v1q, monkeypatch):
    """The single-flight lock is per-URI, not global.

    10 callers each fetching a DIFFERENT URI must run in parallel — a global
    download mutex would serialise them and take ~10x longer. The threshold
    is set comfortably between the parallel time (~0.2 s) and the serial
    time (~2 s) so the test is robust to CI jitter.
    """
    fake = _SlowReadBytes(delay=0.2)
    _patch_read_bytes(monkeypatch, fake)

    uris = [f"memory://bucket/shard-parallel-{i}.faiss" for i in range(10)]

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(v1q._ensure_cached, uri) for uri in uris]
        [f.result(timeout=10) for f in as_completed(futures)]
    elapsed = time.monotonic() - start

    assert fake.count == 10, (
        f"expected one download per distinct URI; got {fake.count}"
    )
    # 10 * 0.2 s parallel ≈ 0.2 s; serial would be ≈ 2 s. Allow generous
    # headroom for thread-pool scheduling on a busy CI runner — anything
    # under 1 s proves the calls overlapped rather than serialised.
    assert elapsed < 1.0, (
        f"different-URI downloads serialised (elapsed={elapsed:.2f}s); "
        "the in-flight lock must be per-URI"
    )


def test_wait_is_bounded(v1q, monkeypatch):
    """A hung initiator must not block waiters indefinitely.

    Override `_DOWNLOAD_COALESCE_WAIT_S` to a small value for the test so it
    runs in well under a second. The contract: a waiter that exceeds the
    deadline raises `DownloadCoalescingTimeout` (or a subclass) with a clear
    message — it does NOT silently return a bogus path.
    """
    hang_started = threading.Event()
    release = threading.Event()

    def hanging_read(uri: str) -> bytes:
        hang_started.set()
        # Block until the test releases us, well after the waiter's deadline.
        release.wait(timeout=5)
        return b"eventually"

    _patch_read_bytes(monkeypatch, hanging_read)
    monkeypatch.setattr(v1q, "_DOWNLOAD_COALESCE_WAIT_S", 0.2)

    uri = "memory://bucket/shard-hang.faiss"

    # Initiator thread takes the in-flight slot and hangs.
    initiator_done = threading.Event()

    def initiator():
        try:
            v1q._ensure_cached(uri)
        except Exception:
            pass
        finally:
            initiator_done.set()

    t = threading.Thread(target=initiator, daemon=True)
    t.start()

    # Wait until the initiator has actually started its (hanging) download
    # so the next caller is guaranteed to land on the waiter path.
    assert hang_started.wait(timeout=2), "initiator never started"

    waiter_start = time.monotonic()
    with pytest.raises(v1q.DownloadCoalescingTimeout):
        v1q._ensure_cached(uri)
    waiter_elapsed = time.monotonic() - waiter_start

    # The waiter must give up around the deadline (0.2 s), not after the
    # initiator's 5 s hang. Allow some slack for scheduling.
    assert waiter_elapsed < 2.0, (
        f"waiter blocked far past its deadline ({waiter_elapsed:.2f}s)"
    )

    # Release the initiator so the test process does not leave a hung thread.
    release.set()
    initiator_done.wait(timeout=5)


@pytest.fixture
def v1q_mmap(monkeypatch, tmp_path):
    """`v1q` fixture variant with `RB_FAISS_MMAP=true` at reload time.

    Pins the cross-feature seam between the mmap branch (captured at module
    import) and the per-URI single-flight coalescing in `_ensure_cached`.
    The waiter path would call `faiss.read_index(IO_FLAG_MMAP|RO)` on the
    local file the initiator just wrote — if `event.set()` ever ran BEFORE
    `os.replace(tmp, path)` completed, a waiter could mmap-open a file the
    kernel had not finished publishing. The contract this fixture exists to
    pin is that ordering invariant under the mmap flag.
    """
    import importlib

    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_FAISS_MMAP", "true")
    import services.query_api.v1_query as v1_query

    importlib.reload(v1_query)
    assert v1_query._MMAP_ENABLED, "fixture failed to enable RB_FAISS_MMAP"
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()
    yield v1_query
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()


def test_mmap_on_concurrent_callers_get_complete_file(v1q_mmap, monkeypatch):
    """Cross-loop seam: under mmap, every waiter sees a fully-published file.

    With `RB_FAISS_MMAP=true` and per-URI single-flight coalescing both active,
    the cold-load path is touched by both features. The race they would jointly
    expose: an initiator's `event.set()` runs before `os.replace(tmp, path)`
    becomes visible, so a waiter wakes, finds the path "present", opens it
    via `mmap(2)`, and reads partial bytes.

    The current implementation (`_ensure_cached` in v1_query) does
    `os.replace` THEN `pop` THEN `event.set()` so the file is durable before
    any waiter can proceed. This test pins that invariant against a 10-way
    burst: every caller's returned path is openable and contains the full
    expected payload byte-for-byte.
    """
    payload = b"M" * (4 * 1024 * 1024)  # 4 MiB — large enough that a partial
    # write would be obvious if the ordering invariant were broken.
    fake = _SlowReadBytes(delay=0.1, payload=payload)
    _patch_read_bytes(monkeypatch, fake)

    uri = "memory://bucket/shard-mmap-coalesce.faiss"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(v1q_mmap._ensure_cached, uri) for _ in range(10)]
        results = [f.result(timeout=10) for f in as_completed(futures)]

    assert fake.count == 1, (
        f"expected one download under coalescing+mmap; got {fake.count}"
    )
    path = results[0]
    assert all(r == path for r in results)
    assert os.path.exists(path)
    with open(path, "rb") as f:
        observed = f.read()
    assert observed == payload, (
        f"waiter saw a file with {len(observed)} bytes; expected the full "
        f"{len(payload)} — the rename-before-set ordering may be broken"
    )
