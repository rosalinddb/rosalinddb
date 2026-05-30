"""SSD tier wiring into the hot path and the ephemeral path.

Pins the activation contract for `services/query_api/v1_query.py:_ensure_cached`
and `services/ephemeral_runner/run.py:_ensure_cached`:

  - `RB_SHARD_TIER_BYTES` unset  -> legacy single-flight path runs (unchanged).
  - `RB_SHARD_TIER_BYTES` set    -> delegates to `adapters.storage.shard_tier.fetch(shard_uri)`.

The cache key in either path is the shard URI itself; the URI is already
content-addressed so it is the canonical identifier for the bytes the caller
wants. No separate `shard_id` parameter is threaded through the tier.

Also pins the classifier extension: `shard_tier.ShardTierTimeout` -> the same
wire code as `DownloadCoalescingTimeout` (`storage_unavailable` / 503).
The two timeouts describe the same operational condition from the customer's
perspective ("another caller is mid-download; please retry") and must collapse
to the same retry hint.

The tests are tier-agnostic about the *implementation* — they patch
`shard_tier.fetch` and `adapters.storage.storage.read_bytes` and assert which
one ran. A regression that wires the tier without honouring the env gate
would silently break the rollback contract and surface here.
"""
from __future__ import annotations

import importlib
import os
import threading
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def v1q_tier_off(monkeypatch, tmp_path):
    """`v1_query` reloaded with the SSD tier OFF (env var unset).

    The activation gate reads `RB_SHARD_TIER_BYTES` at call time (not import
    time) so the env-var dance is the contract. We still reload the module
    so any test-local state from previous tests is wiped — the in-flight
    dict is module-global.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.delenv("RB_SHARD_TIER_BYTES", raising=False)
    import services.query_api.v1_query as v1_query

    importlib.reload(v1_query)
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()
    yield v1_query
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()


@pytest.fixture
def v1q_tier_on(monkeypatch, tmp_path):
    """`v1_query` reloaded with the SSD tier ON.

    Sets `RB_SHARD_TIER_BYTES` to a small but non-zero value (the activation
    gate is "is the env var present", not its numeric value). Also sets
    `RB_SHARD_TIER_DIR` so the tier's own files do not collide with the
    legacy cache.
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", "65536")
    monkeypatch.setenv(
        "RB_SHARD_TIER_DIR", str(tmp_path / "shards" / "tier-managed"),
    )
    import services.query_api.v1_query as v1_query

    importlib.reload(v1_query)
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()
    # Reset the tier's module-global state too so each test sees a fresh
    # residency table.
    from adapters.storage import shard_tier

    importlib.reload(shard_tier)
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()
    yield v1_query
    with v1_query._INFLIGHT_DOWNLOADS_LOCK:
        v1_query._INFLIGHT_DOWNLOADS.clear()
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()


@pytest.fixture
def eph_tier_off(monkeypatch, tmp_path):
    """`ephemeral_runner.run` reloaded with the SSD tier OFF."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards-eph"))
    monkeypatch.delenv("RB_SHARD_TIER_BYTES", raising=False)
    import services.ephemeral_runner.run as eph_run

    importlib.reload(eph_run)
    yield eph_run


@pytest.fixture
def eph_tier_on(monkeypatch, tmp_path):
    """`ephemeral_runner.run` reloaded with the SSD tier ON."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "shards-eph"))
    monkeypatch.setenv("RB_SHARD_TIER_BYTES", "65536")
    monkeypatch.setenv(
        "RB_SHARD_TIER_DIR", str(tmp_path / "shards-eph" / "tier-managed"),
    )
    import services.ephemeral_runner.run as eph_run

    importlib.reload(eph_run)
    from adapters.storage import shard_tier

    importlib.reload(shard_tier)
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()
    yield eph_run
    with shard_tier._RESIDENCY_LOCK:
        shard_tier._RESIDENCY.clear()
        shard_tier._BYTES_USED = 0
    with shard_tier._INFLIGHT_LOCK:
        shard_tier._INFLIGHT.clear()


class _CountingReadBytes:
    """Stand-in for `adapters.storage.storage.read_bytes` that records calls.

    Tests use call count + URI list to assert which path executed (legacy
    `_ensure_cached` invokes `read_bytes`; tier `fetch` also does, but the
    tests patch `shard_tier.fetch` itself so the underlying `read_bytes`
    never runs on the tier-on path).
    """

    def __init__(self, payload: bytes = b"shard-bytes"):
        self.payload = payload
        self.count = 0
        self.uris: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, uri: str) -> bytes:
        with self._lock:
            self.count += 1
            self.uris.append(uri)
        return self.payload


def _patch_read_bytes(monkeypatch, fake):
    """Patch the cache-fill download at the storage module.

    `_ensure_cached` uses `download_to(uri, local_path)` to stream bytes to
    disk. We patch both symbols: `read_bytes` so a direct caller sees the
    same fake, and `download_to` with a thin wrapper that calls the fake and
    writes the returned bytes to disk. The wiring tests assert the
    legacy-path branch ran by counting `read_bytes`-shaped calls, so the
    wrapper preserves that signal.
    """
    import adapters.storage.storage as storage_mod

    monkeypatch.setattr(storage_mod, "read_bytes", fake)

    def _download_to(uri: str, local_path: str) -> None:
        payload = fake(uri)
        with open(local_path, "wb") as f:
            f.write(payload)

    monkeypatch.setattr(storage_mod, "download_to", _download_to)


# --- v1_query: activation gate -------------------------------------------


def test_tier_off_uses_legacy_path(v1q_tier_off, monkeypatch):
    """Env var unset -> legacy `read_bytes` runs; `shard_tier.fetch` is NOT called.

    This is the rollback contract. A deployment that has not opted in to the
    SSD tier must see byte-for-byte unchanged behaviour from `_ensure_cached`.
    """
    fake_read = _CountingReadBytes(payload=b"legacy-bytes")
    _patch_read_bytes(monkeypatch, fake_read)

    # Patch `shard_tier.fetch` so a regression that ignores the env gate
    # surfaces as an unexpected call rather than running the real downloader.
    from adapters.storage import shard_tier

    fetch_spy = MagicMock(side_effect=AssertionError(
        "shard_tier.fetch must not be called when RB_SHARD_TIER_BYTES is unset"
    ))
    monkeypatch.setattr(shard_tier, "fetch", fetch_spy)

    uri = "memory://bucket/shard-tier-off.faiss"
    path = v1q_tier_off._ensure_cached(uri)

    assert fake_read.count == 1, (
        f"legacy path must call read_bytes exactly once; got {fake_read.count}"
    )
    assert fetch_spy.call_count == 0
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"legacy-bytes"


def test_tier_on_delegates_to_shard_tier(v1q_tier_on, monkeypatch):
    """Env set -> `shard_tier.fetch(uri)` runs.

    The legacy `read_bytes` MUST NOT be invoked by `_ensure_cached` on this
    path (the tier may call it internally, but we patch `fetch` itself, so
    the underlying downloader is also unreachable in the test).
    """
    from adapters.storage import shard_tier

    expected_path = "/tmp/fake-tier-path/shard-tier-on.faiss"
    fetch_spy = MagicMock(return_value=expected_path)
    monkeypatch.setattr(shard_tier, "fetch", fetch_spy)

    fake_read = _CountingReadBytes()
    _patch_read_bytes(monkeypatch, fake_read)

    uri = "memory://bucket/shard-tier-on.faiss"
    returned = v1q_tier_on._ensure_cached(uri)

    assert returned == expected_path
    fetch_spy.assert_called_once_with(uri)
    assert fake_read.count == 0, (
        "legacy read_bytes must not run when delegation succeeds"
    )


def test_tier_on_propagates_filenotfound(v1q_tier_on, monkeypatch):
    """`FileNotFoundError` from `shard_tier.fetch` propagates untouched.

    The classifier maps `FileNotFoundError` to `storage_unavailable` -> 503
    via the long-standing branch (`_classify_hot_path_error`). Pinning that
    `_ensure_cached` does not swallow the exception or re-wrap it as a
    different type — the contract that "tier raises, caller classifies" is
    load-bearing for the wire code.
    """
    from adapters.storage import shard_tier

    fetch_spy = MagicMock(side_effect=FileNotFoundError("memory:// key not found"))
    monkeypatch.setattr(shard_tier, "fetch", fetch_spy)

    with pytest.raises(FileNotFoundError):
        v1q_tier_on._ensure_cached("memory://bucket/missing.faiss")


# --- v1_query: classifier extension --------------------------------------


def test_classifier_maps_shard_tier_timeout_to_503(v1q_tier_on):
    """`ShardTierTimeout` -> `("storage_unavailable", <safe msg>)`.

    The classifier returns `(code, safe_message)`; the wire-status mapping
    in `run_query` then turns `storage_unavailable` into HTTP 503. The
    contract under test here is just the classifier output — the 503 mapping
    is exercised by the run_query path tests elsewhere.
    """
    from adapters.storage import shard_tier

    exc = shard_tier.ShardTierTimeout("timed out after 300s waiting for ...")
    code, _msg = v1q_tier_on._classify_hot_path_error(exc)
    assert code == "storage_unavailable", (
        f"expected storage_unavailable for ShardTierTimeout; got {code}"
    )


def test_classifier_shard_tier_timeout_matches_download_coalescing(v1q_tier_on):
    """`ShardTierTimeout` and `DownloadCoalescingTimeout` map identically.

    The two exceptions describe the same operational condition from the
    customer's perspective ("a coalesced waiter exceeded its bounded wait
    on someone else's in-flight download"). They MUST surface as the same
    error code so a customer-side retry policy does not need to distinguish
    "which layer timed out" — they get the same 503 either way.
    """
    from adapters.storage import shard_tier

    tier_code, _ = v1q_tier_on._classify_hot_path_error(
        shard_tier.ShardTierTimeout("x")
    )
    legacy_code, _ = v1q_tier_on._classify_hot_path_error(
        v1q_tier_on.DownloadCoalescingTimeout("y")
    )
    assert tier_code == legacy_code == "storage_unavailable"


def test_classifier_maps_filenotfound_to_storage_unavailable(v1q_tier_on):
    """`FileNotFoundError` -> `storage_unavailable` (pre-existing branch).

    Pinning this so a future refactor of the classifier cannot silently
    drop the long-standing mapping that the tier's missing-URI path relies
    on. The wiring contract "tier raises, caller classifies" is only
    meaningful if both ShardTierTimeout AND FileNotFoundError surface as
    503; the tier exposes both as raise paths in its public docstring.
    """
    code, _msg = v1q_tier_on._classify_hot_path_error(
        FileNotFoundError("missing")
    )
    assert code == "storage_unavailable"


# --- ephemeral_runner mirror tests ---------------------------------------


def test_eph_tier_off_uses_legacy_path(eph_tier_off, monkeypatch):
    """Ephemeral path mirror of `test_tier_off_uses_legacy_path`."""
    fake_read = _CountingReadBytes(payload=b"eph-legacy")
    _patch_read_bytes(monkeypatch, fake_read)

    from adapters.storage import shard_tier

    fetch_spy = MagicMock(side_effect=AssertionError(
        "shard_tier.fetch must not be called when RB_SHARD_TIER_BYTES is unset"
    ))
    monkeypatch.setattr(shard_tier, "fetch", fetch_spy)

    uri = "memory://bucket/eph-tier-off.faiss"
    path = eph_tier_off._ensure_cached(uri)

    assert fake_read.count == 1
    assert fetch_spy.call_count == 0
    assert os.path.exists(path)


def test_eph_tier_on_delegates_to_shard_tier(eph_tier_on, monkeypatch):
    """Ephemeral path mirror of `test_tier_on_delegates_to_shard_tier`."""
    from adapters.storage import shard_tier

    expected = "/tmp/fake-eph-tier-path/eph.faiss"
    fetch_spy = MagicMock(return_value=expected)
    monkeypatch.setattr(shard_tier, "fetch", fetch_spy)

    fake_read = _CountingReadBytes()
    _patch_read_bytes(monkeypatch, fake_read)

    uri = "memory://bucket/eph-tier-on.faiss"
    returned = eph_tier_on._ensure_cached(uri)

    assert returned == expected
    fetch_spy.assert_called_once_with(uri)
    assert fake_read.count == 0


def test_eph_classifier_maps_shard_tier_timeout_to_503(eph_tier_on):
    """Ephemeral runner's `_classify_error` mirrors the hot-path mapping.

    Same operational condition, same wire code. The duplication exists
    because the ephemeral runner deliberately avoids importing from the
    query_api package (circular import) — keep the two classifiers in sync.
    """
    from adapters.storage import shard_tier

    exc = shard_tier.ShardTierTimeout("timed out after 300s")
    code, _msg = eph_tier_on._classify_error(exc)
    assert code == "storage_unavailable"
