"""`services/_common/dp_id.py` — DP identity resolution.

The fallback chain is:

  1. `RB_DP_ID` env var (explicit override, wins unconditionally).
  2. `HOSTNAME` env var (the Docker / Kubernetes default DP identifier —
     deployment-layer concerns set this; the application reads it).
  3. A UUID4 generated on first call and persisted to `${CACHE_DIR}/.dp_id`
     so a single DP keeps the same identity across process restarts (and
     so a re-import in the same process reads back the persisted value).

The persistence file path is `${CACHE_DIR}/.dp_id`; the first DP to write
wins (an open-or-create atomic write). Subsequent calls in the same process
or in any subsequent process must read back the same value.

Deployment-layer identifiers (whatever the host platform exposes) must not
leak into application code — those are deployment-layer concerns and would
couple the application code to a single platform's environment contract.
The fallback chain above is the only one the application knows.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


@pytest.fixture
def dp_id_mod(monkeypatch, tmp_path):
    """Reload `services._common.dp_id` with a clean per-test environment.

    The module caches its resolved value (a `dp_id()` call must be cheap
    after the first) so the fixture reload is what resets state between
    tests. `CACHE_DIR` is pointed at a per-test tmp_path so the on-disk
    persistence file (`${CACHE_DIR}/.dp_id`) is isolated.
    """
    monkeypatch.delenv("RB_DP_ID", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    import services._common.dp_id as mod

    importlib.reload(mod)
    return mod


def test_explicit_env_var_wins(monkeypatch, dp_id_mod):
    """`RB_DP_ID` is the unconditional winner — HOSTNAME and UUID lose."""
    monkeypatch.setenv("HOSTNAME", "docker-host-123")  # would have won otherwise
    monkeypatch.setenv("RB_DP_ID", "explicit-dp-7")
    # Reload so the module re-resolves under the new env.
    import services._common.dp_id as mod
    importlib.reload(mod)

    assert mod.dp_id() == "explicit-dp-7"


def test_hostname_wins_when_explicit_unset(monkeypatch, dp_id_mod):
    """With `RB_DP_ID` unset, `HOSTNAME` is the identifier.

    Mirrors the Docker / Kubernetes convention: the orchestrator already
    sets `HOSTNAME` to the container / pod identifier, and we adopt it as
    the DP identity without forcing the operator to set a second env var.
    """
    monkeypatch.setenv("HOSTNAME", "k8s-pod-dp-7-abc12")
    import services._common.dp_id as mod
    importlib.reload(mod)

    assert mod.dp_id() == "k8s-pod-dp-7-abc12"


def test_uuid_generated_and_persisted_on_first_call(dp_id_mod, tmp_path):
    """No env hints -> generate a UUID and persist it to `${CACHE_DIR}/.dp_id`.

    The UUID must be a syntactically valid v4 string so an operator
    reading the file (or the table) can recognise the format.
    """
    value = dp_id_mod.dp_id()
    assert _UUID_RE.match(value), f"expected v4 UUID, got {value!r}"
    persisted = (tmp_path / ".dp_id").read_text(encoding="utf-8").strip()
    assert persisted == value


def test_uuid_read_back_on_subsequent_calls(dp_id_mod):
    """The persisted UUID is reused across a module reload in the same process.

    This is the load-bearing property — a DP restart (or a fresh import in
    the same process) MUST observe the same identity. Otherwise the
    residency registry rows would re-key on every restart and rendezvous
    routing would never see warm hits.
    """
    first = dp_id_mod.dp_id()
    import services._common.dp_id as mod
    importlib.reload(mod)
    second = mod.dp_id()
    assert first == second


def test_uuid_read_back_from_disk_for_a_fresh_module(monkeypatch, tmp_path):
    """A separate fresh import (simulating a process restart) reads from disk.

    `${CACHE_DIR}/.dp_id` survives DP restart so the identity is stable
    across the deployment lifetime.
    """
    monkeypatch.delenv("RB_DP_ID", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    # Seed the file by hand to simulate "a previous process wrote this".
    (tmp_path / ".dp_id").write_text("11111111-2222-4333-8444-555555555555\n")
    import services._common.dp_id as mod
    importlib.reload(mod)

    assert mod.dp_id() == "11111111-2222-4333-8444-555555555555"


def test_dp_id_is_cached_within_a_process(dp_id_mod):
    """Repeated calls in one process return the same value without re-reading.

    `dp_id()` is on the hot startup path (residency writer + routing
    lookups), so the resolved value is memoised after the first
    call. The test exercises this by deleting the env vars and the disk
    file between two calls and asserting the value is unchanged.
    """
    first = dp_id_mod.dp_id()
    # Burn down every source of truth; the cached value must still come back.
    persistence = Path(os.environ["CACHE_DIR"]) / ".dp_id"
    if persistence.exists():
        persistence.unlink()
    os.environ.pop("HOSTNAME", None)
    os.environ.pop("RB_DP_ID", None)
    second = dp_id_mod.dp_id()
    assert first == second


def test_persistence_file_is_created_in_cache_dir(monkeypatch, tmp_path):
    """The file is written under `CACHE_DIR`, not the current working dir.

    Self-host friendliness: a deployment that mounts a persistent volume
    at `CACHE_DIR` (the Docker pattern) keeps its DP identity across
    container restarts even when the container root filesystem is
    ephemeral.
    """
    nested = tmp_path / "var" / "cache" / "shards"
    monkeypatch.delenv("RB_DP_ID", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setenv("CACHE_DIR", str(nested))
    import services._common.dp_id as mod
    importlib.reload(mod)
    _ = mod.dp_id()

    assert (nested / ".dp_id").is_file()


def test_no_platform_specific_identifiers_consulted(monkeypatch, tmp_path):
    """Deployment-layer identifiers (whatever the host platform exposes) must not leak into application code.

    Setting vendor-style env vars such as `SOME_PLATFORM_MACHINE_ID` /
    `SOME_PLATFORM_APP_NAME` must NOT influence dp_id — the application
    only reads `RB_DP_ID`, `HOSTNAME`, and the on-disk UUID file. An
    operator can wire `RB_DP_ID=$SOME_PLATFORM_MACHINE_ID` from the
    deployment config; the module itself must not bake in that knowledge.
    """
    monkeypatch.delenv("RB_DP_ID", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setenv("SOME_PLATFORM_MACHINE_ID", "vendor-machine-zzz")
    monkeypatch.setenv("SOME_PLATFORM_APP_NAME", "rb-app-yyy")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    import services._common.dp_id as mod
    importlib.reload(mod)

    value = mod.dp_id()
    assert "vendor-machine-zzz" not in value
    assert "rb-app-yyy" not in value
    # And the value should match the UUID-on-disk path because no other
    # signal was available.
    assert _UUID_RE.match(value)
