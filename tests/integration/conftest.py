"""Integration-suite fixtures: a real MinIO backend via ``testcontainers``.

RosalindDB is object-storage-first, so the integration suite must exercise the
storage adapter against a real S3-compatible service rather than a stand-in.
``testcontainers`` spins up an ephemeral MinIO container once per test session
— self-contained, works in CI (GitHub runners ship Docker), and needs no
manual ``docker compose up``.

The session-scoped ``minio_container`` fixture starts the container and creates
the shared bucket. The function-scoped ``minio_env`` fixture (autouse) points
the storage adapter's S3 env vars at it and hands every test a *unique* landing
/ index prefix so uploads never collide across tests.

A Docker dependency for this suite is intended. There is deliberately no
no-Docker fallback: if Docker is unavailable the integration suite fails fast.
"""
from __future__ import annotations

import os
import uuid

import pytest

try:  # testcontainers is an integration-only dependency.
    from testcontainers.minio import MinioContainer
except ImportError as exc:  # pragma: no cover - surfaced as a clear skip
    MinioContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_BUCKET = "rosalinddb-test"


@pytest.fixture(scope="session")
def minio_container():
    """Start one MinIO container for the whole integration session.

    Yields the container's ``get_config()`` dict (endpoint/access/secret). The
    shared test bucket is created once up front; per-test isolation is achieved
    by unique key prefixes rather than per-test buckets.
    """
    if MinioContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the integration suite; "
            f"install it into the venv (`pip install testcontainers[minio]`). "
            f"Original import error: {_IMPORT_ERROR}"
        )
    with MinioContainer() as minio:
        config = minio.get_config()
        client = minio.get_client()
        if not client.bucket_exists(_BUCKET):
            client.make_bucket(_BUCKET)
        yield config


@pytest.fixture(autouse=True)
def minio_env(minio_container, monkeypatch):
    """Point the storage adapter at the session MinIO and isolate each test.

    Sets the ``S3_*`` env vars the storage adapter reads, and exports
    ``RB_TEST_LANDING_PREFIX`` / ``RB_TEST_INDEXES_PREFIX`` — unique per test —
    so test fixtures can build ``s3://`` prefixes that never collide. Each test
    still sets ``LANDING_PREFIX`` / ``INDEXES_PREFIX`` itself (then reloads the
    pipeline modules), exactly as before; only the scheme changes from
    ``file://`` to ``s3://``.
    """
    monkeypatch.setenv("S3_ENDPOINT_URL", f"http://{minio_container['endpoint']}")
    monkeypatch.setenv("S3_ACCESS_KEY", minio_container["access_key"])
    monkeypatch.setenv("S3_SECRET_KEY", minio_container["secret_key"])
    monkeypatch.setenv("S3_REGION", "us-east-1")

    run_id = uuid.uuid4().hex[:12]
    monkeypatch.setenv("RB_TEST_LANDING_PREFIX", f"s3://{_BUCKET}/{run_id}/landing")
    monkeypatch.setenv("RB_TEST_INDEXES_PREFIX", f"s3://{_BUCKET}/{run_id}/indexes")
    yield


@pytest.fixture
def s3_landing_prefix() -> str:
    """Unique ``s3://`` landing prefix for the current test."""
    return os.environ["RB_TEST_LANDING_PREFIX"]


@pytest.fixture
def s3_indexes_prefix() -> str:
    """Unique ``s3://`` indexes prefix for the current test."""
    return os.environ["RB_TEST_INDEXES_PREFIX"]
