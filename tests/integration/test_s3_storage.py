"""Integration tests for the ``s3://`` storage adapter against real MinIO.

The unit suite exercises the storage contract against the ``memory://``
adapter; this module runs the *same* contract against a real S3-compatible
backend (MinIO via ``testcontainers``, wired by ``conftest.py``).

The critical case here is missing-key behaviour: a caller that branches on the
exception type must see the *same* exception on both tiers. The ``s3://``
adapter normalizes botocore's ``NoSuchKey``/404 ``ClientError`` into
``FileNotFoundError``, exactly as the ``memory://`` adapter raises it — so
``test_read_missing_key_raises`` here mirrors the unit-tier assertion.
"""
from __future__ import annotations

import os
import uuid

import pytest

from adapters.storage import storage

pytestmark = pytest.mark.integration


def _unique_uri() -> str:
    """A fresh ``s3://`` URI under the per-test landing prefix."""
    prefix = os.environ["RB_TEST_LANDING_PREFIX"]
    return f"{prefix}/s3-storage-{uuid.uuid4().hex[:12]}.bin"


# --- round-trip -----------------------------------------------------------


def test_write_then_read_round_trip():
    uri = _unique_uri()
    storage.write_bytes(uri, b"hello-s3-bytes")
    assert storage.read_bytes(uri) == b"hello-s3-bytes"


# --- missing key ----------------------------------------------------------


def test_read_missing_key_raises():
    # FileNotFoundError *exactly* — the s3:// adapter normalizes botocore's
    # NoSuchKey/404 ClientError to the same type the memory:// adapter raises.
    # This is the integration-tier twin of the unit assertion in
    # tests/unit/test_memory_storage.py; both must agree.
    with pytest.raises(FileNotFoundError):
        storage.read_bytes(_unique_uri())


def test_open_reader_missing_key_raises():
    # open_reader's s3:// path fetches independently of read_bytes; it must
    # normalize a missing key to FileNotFoundError too.
    with pytest.raises(FileNotFoundError):
        list(storage.open_reader(_unique_uri()))
