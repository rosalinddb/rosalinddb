"""Shared pytest configuration for the RosalindDB backend test suite.

The suite is split into two tiers, marked automatically by directory:

  - ``tests/unit/*``         -> ``@pytest.mark.unit``
        Fast, hermetic. Zero filesystem and zero network I/O. Storage, when
        touched at all, goes through the dict-backed ``memory://`` adapter.
        Run with ``pytest -m unit`` (no Docker required).

  - ``tests/integration/*``  -> ``@pytest.mark.integration``
        Exercise FastAPI + state + storage + landing + FAISS together against
        a *real* MinIO instance, spun up per session via ``testcontainers``
        (see ``tests/integration/conftest.py``). Requires Docker.
        Run with ``pytest -m integration``.

The marking is done by directory in ``pytest_collection_modifyitems`` so no
individual test needs a hand-written marker.
"""
import os
import sys
from pathlib import Path


# Ensure repo root is on sys.path so 'adapters', 'services' are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default to in-memory state for tests (the integration suite still uses the
# memory:// state adapter — real Postgres for integration is a documented
# follow-up; storage is the firm object-storage mandate).
os.environ.setdefault("DATABASE_URL", "memory://local")
os.environ.setdefault("DIMENSION", "4")

# Observability: tests run without an OTLP collector. Disable the OpenTelemetry
# SDK so `init_observability` is a no-op — instrumentation stays additive and
# the test run is fast and quiet (no exporter retry noise).
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# OSS opt-in: `RB_REQUIRE_AUTH` defaults OFF in production (self-host friendly)
# but the existing test suite was written before the gate existed and assumes
# every request must authenticate. Default it ON here so the existing tests
# keep their signup/login/JWT assertions; `tests/integration/test_auth_disabled.py`
# explicitly clears this in its own per-test fixture before building the
# TestClient. See services/auth/jwt_utils.py:auth_required.
os.environ.setdefault("RB_REQUIRE_AUTH", "true")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``unit`` / ``integration`` marker based on test path.

    Anything under ``tests/unit/`` is a unit test; anything under
    ``tests/integration/`` is an integration test. This keeps ``pytest -m unit``
    / ``pytest -m integration`` working without decorating every function.
    """
    import pytest

    for item in items:
        path = str(item.fspath)
        if f"{os.sep}integration{os.sep}" in path:
            item.add_marker(pytest.mark.integration)
        elif f"{os.sep}unit{os.sep}" in path:
            item.add_marker(pytest.mark.unit)
