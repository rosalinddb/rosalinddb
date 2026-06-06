from __future__ import annotations

"""DP identity resolution for the residency registry.

A DP needs a stable string identifier so the registry rows it writes via
`state.register_dp_shard_warm(dp_id, shard_uri, ...)` survive its own
restart (the same DP after a process restart must keep the same identity,
or every restart re-keys the residency table and the routing read sees
zero warm hits).

Fallback chain (first match wins):

  1. `RB_DP_ID` env var — explicit operator override. Useful when a
     deployment platform's container identifier is not what you want as
     the DP identity, or when an operator wants a human-readable id.
  2. `HOSTNAME` env var — the Docker / Kubernetes convention. The
     orchestrator already sets this to the container / pod name, so
     adopting it as the DP identity costs nothing.
  3. UUID4 generated and persisted to `${CACHE_DIR}/.dp_id`. The first
     write wins (open-or-create atomic); subsequent calls in the same
     process (or any future process on the same `CACHE_DIR` volume) read
     back the persisted value.

Why no platform-specific identifiers here: deployment-layer identifiers
(a cloud instance id, the orchestrator-provided pod name, etc.) are
concerns of the deployment platform, not the application. An operator who
wants to use such an identifier wires it via `RB_DP_ID=<value>` from
outside; the module itself must not bake in platform-specific env var
names — that would couple the codebase to one provider and break the
Docker/Kubernetes-first contract.

The resolved value is memoised at module scope: `dp_id()` is on the hot
startup path and a single call per process is enough.
"""

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from adapters import config


_LOG = logging.getLogger(__name__)

# Process-wide cache for the resolved identifier. `None` means "not yet
# resolved"; any string is the answer. Guarded by `_RESOLVE_LOCK` so two
# concurrent first calls do not race to generate two UUIDs.
_RESOLVED: Optional[str] = None
_RESOLVE_LOCK = threading.Lock()

# Filename used inside `CACHE_DIR` for the persistence record. Hidden
# (leading dot) so a casual `ls` of the cache directory does not show it;
# the file is operator metadata, not a user-facing artifact.
_PERSISTENCE_FILENAME = ".dp_id"


def _cache_dir() -> Path:
    """Return the `CACHE_DIR` path. Read at call time so a fresh import
    after the env var is set picks up the new location (test fixtures
    rely on this — production sets the env once at startup).
    """
    return Path(config.cache_dir())


def _persistence_path() -> Path:
    return _cache_dir() / _PERSISTENCE_FILENAME


def _read_persisted() -> Optional[str]:
    """Read the persisted identifier from disk, or return None if absent
    or unreadable. Best-effort: a permissions error logs and falls
    through to generate-and-persist, which will hit the same error on
    write and propagate it — caller will see a real failure, not a
    silent identity churn.
    """
    path = _persistence_path()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:  # permissions, broken FS, etc.
        _LOG.warning("dp_id: could not read %s: %s", path, exc)
        return None
    return value or None


def _write_persisted(value: str) -> None:
    """Persist `value` to `${CACHE_DIR}/.dp_id` using an open-or-create
    write so two DPs starting concurrently on the same `CACHE_DIR`
    converge on the first writer's id.

    `O_CREAT | O_EXCL` is the atomic "create if absent, fail if present"
    primitive — POSIX guarantees one writer wins. If we lose the race
    (the file appeared between our `_read_persisted` and this write) we
    re-read and adopt the winner's value rather than overwriting it.
    The caller (`dp_id()`) handles the re-read on `FileExistsError`.
    """
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _persistence_path()
    # `os.O_EXCL` makes the open fail if another process has already
    # created the file, which is exactly the "first writer wins"
    # semantics we want. Without `O_EXCL`, two concurrent boots would
    # both overwrite and end up with mismatched identities.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, (value + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _resolve() -> str:
    """Walk the fallback chain. Caller holds `_RESOLVE_LOCK`.

    Order matters — see module docstring. The function is allowed to
    write to disk (generate-and-persist path) but never reads from
    deployment-platform-specific env vars beyond `HOSTNAME`.
    """
    explicit = config.dp_id()
    if explicit:
        return explicit
    hostname = config.hostname()
    if hostname:
        return hostname
    persisted = _read_persisted()
    if persisted is not None:
        return persisted
    # Generate-and-persist. The `O_EXCL` write below loses gracefully to
    # a racing DP that just persisted its own id; we re-read and adopt.
    generated = str(uuid.uuid4())
    try:
        _write_persisted(generated)
        return generated
    except FileExistsError:
        # A concurrent process wrote first; adopt their id rather than
        # leaving two DPs with different identities for the same volume.
        winner = _read_persisted()
        if winner is not None:
            return winner
        # Should never happen — the file existed but we cannot read it.
        # Fall through to our generated id rather than raise, so a
        # broken volume does not crash the DP.
        _LOG.warning(
            "dp_id: lost race for %s but could not re-read winner",
            _persistence_path(),
        )
        return generated


def dp_id() -> str:
    """Return the stable DP identity string. Memoised per process.

    Cheap after the first call — subsequent calls return the cached
    value without touching env or disk. Thread-safe: a concurrent first
    call is serialised by `_RESOLVE_LOCK` so we never generate two
    UUIDs and only one of them wins.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    with _RESOLVE_LOCK:
        if _RESOLVED is not None:
            return _RESOLVED
        _RESOLVED = _resolve()
        return _RESOLVED
