"""Unit tests for the cached boto3 S3 client singleton.

A boto3 client is expensive to build (it loads service-model JSON, resolves
the endpoint, wires up credential providers, and opens a fresh HTTP connection
pool) and is documented to be thread-safe and reusable. The storage adapter
therefore builds it once per process and reuses it on every call. These tests
pin that contract:

  - repeated ``_s3_client()`` calls return the *same* object (identity)
  - the client is NOT constructed at import time (modules import the storage
    adapter without S3 configured)
  - the lazy first construction is guarded so concurrent threads on a cold
    process still see exactly one client
"""
from __future__ import annotations

import importlib
import threading

import pytest

from adapters.storage import storage


@pytest.fixture(autouse=True)
def _drop_cached_client_after_test():
    """Reset the module-global client after every test.

    These tests build a real client into ``storage._S3_CLIENT``; without this
    teardown that dummy-credentialed singleton would survive session-wide and
    any later test calling ``_s3_client()`` would get it instead of its own.
    """
    yield
    storage._reset_s3_client()


def _reset(monkeypatch):
    """Point S3 env vars at a dummy backend and drop any cached client."""
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "test-access")
    monkeypatch.setenv("S3_SECRET_KEY", "test-secret")
    monkeypatch.setenv("S3_REGION", "us-east-1")
    storage._reset_s3_client()


# --- identity / reuse -----------------------------------------------------


def test_repeated_calls_return_same_client(monkeypatch):
    _reset(monkeypatch)
    first = storage._s3_client()
    second = storage._s3_client()
    assert first is second


# --- not built at import time --------------------------------------------


def test_client_not_constructed_at_import_time():
    # Re-importing the module must not build a client: tests and some
    # environments import the storage adapter with no S3 configured. The
    # singleton must stay None until the first _s3_client() call.
    storage._reset_s3_client()
    reloaded = importlib.reload(storage)
    assert reloaded._S3_CLIENT is None


# --- thread-safe lazy init -----------------------------------------------


def test_concurrent_lazy_init_yields_one_client(monkeypatch):
    _reset(monkeypatch)

    results: list = []
    barrier = threading.Barrier(8)

    def grab():
        # The barrier maximizes the odds every thread races the cold-start
        # construction at once, so an unguarded init would build several.
        barrier.wait()
        results.append(storage._s3_client())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(c is results[0] for c in results)
