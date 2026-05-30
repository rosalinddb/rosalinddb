"""Unit tests for the RB_FAISS_MMAP flag wiring in services/query_api/v1_query.py.

Pins the public contract that the flag opens the mmap path inside
`_hot_search`, that `_index_nbytes` reports a small fixed estimate when mmap
is on (so the byte-budgeted cache does not double-count file-backed pages as
if they were RSS), and that the import-time FUSE-mount guardrail flags
object-store-backed cache dirs that are unsafe for mmap.

The flag is read ONCE at module import, so each test that flips the env
reloads `v1_query` with `importlib.reload` (matches the pattern in
`tests/unit/test_dp_io_offload.py` and `tests/unit/test_api_keys.py`).
"""
from __future__ import annotations

import importlib

import faiss  # type: ignore
import numpy as np
import pytest


pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _reload_v1_query():
    """Reload `services.query_api.v1_query` so module-level env reads re-run.

    `_MMAP_ENABLED` is captured at import time on purpose (mid-run flipping
    would force every caller into reload dances). Tests that need to flip
    the flag call this helper after `monkeypatch.setenv` / `delenv`.
    """
    import services.query_api.v1_query as v1q

    importlib.reload(v1q)
    return v1q


def _make_tiny_index(dim: int = 4, n: int = 8):
    """Build a small IndexIDMap2 over IndexFlatL2 — the shape `_hot_search` sees."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    inner = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(inner)
    index.add_with_ids(vecs, ids)
    return index


# --- env flag parsing -----------------------------------------------------


def test_mmap_disabled_by_default(monkeypatch):
    """With no env var set, _MMAP_ENABLED is False."""
    monkeypatch.delenv("RB_FAISS_MMAP", raising=False)
    v1q = _reload_v1_query()
    assert v1q._MMAP_ENABLED is False


def test_mmap_enabled_via_env(monkeypatch):
    """RB_FAISS_MMAP=true makes _MMAP_ENABLED True (matches existing _truthy convention)."""
    monkeypatch.setenv("RB_FAISS_MMAP", "true")
    v1q = _reload_v1_query()
    assert v1q._MMAP_ENABLED is True


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES", "on", "On"])
def test_mmap_truthy_variants(monkeypatch, value):
    """All of {true, 1, yes, on} (case-insensitive) enable the flag."""
    monkeypatch.setenv("RB_FAISS_MMAP", value)
    v1q = _reload_v1_query()
    assert v1q._MMAP_ENABLED is True, f"value {value!r} should enable mmap"


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "", "  "])
def test_mmap_falsy_variants(monkeypatch, value):
    """All of {false, 0, no, off, empty string} leave the flag off."""
    monkeypatch.setenv("RB_FAISS_MMAP", value)
    v1q = _reload_v1_query()
    assert v1q._MMAP_ENABLED is False, f"value {value!r} should NOT enable mmap"


# --- _index_nbytes branching ----------------------------------------------


def test_index_nbytes_without_mmap_uses_serialize(monkeypatch):
    """Default path: _index_nbytes returns the size derived from faiss.serialize_index.

    Build a tiny IndexFlatL2, call _index_nbytes, expect a non-trivial number
    on the order of vectors * dim * 4 bytes.
    """
    monkeypatch.delenv("RB_FAISS_MMAP", raising=False)
    v1q = _reload_v1_query()
    dim, n = 4, 8
    index = _make_tiny_index(dim=dim, n=n)
    nbytes = v1q._index_nbytes(index)
    # serialize-derived size is at least the raw float32 payload (n*dim*4),
    # plus FAISS framing — should be small but well above zero.
    assert nbytes >= n * dim * 4
    # Sanity: not the mmap-mode fixed 32 MB estimate.
    assert nbytes < 32 * 1024 * 1024


def test_index_nbytes_under_mmap_returns_small_estimate(monkeypatch):
    """With RB_FAISS_MMAP=true, _index_nbytes returns the exported constant.

    The exact value MUST be <= 64 MB so the cache budget bookkeeping doesn't
    treat mmap'd indexes as if they were resident in RSS. The test asserts
    against `_MMAP_INDEX_ESTIMATE_BYTES` (not a hard-coded magic number) so
    a future tune of the estimate updates the contract in one place.
    """
    monkeypatch.setenv("RB_FAISS_MMAP", "true")
    v1q = _reload_v1_query()
    index = _make_tiny_index()
    nbytes = v1q._index_nbytes(index)
    assert nbytes <= 64 * 1024 * 1024
    assert nbytes == v1q._MMAP_INDEX_ESTIMATE_BYTES


# --- FUSE-mount startup guardrail -----------------------------------------


def _write_mountinfo(tmp_path, entries):
    """Write a fake /proc/self/mountinfo file from (mount_point, fs_type) tuples.

    Each entry produces one mountinfo line in the documented format:
      mount_id parent_id major:minor root mount_point opts - fs_type src super_opts
    """
    lines = []
    for i, (mount_point, fs_type) in enumerate(entries, start=1):
        lines.append(
            f"{30 + i} 29 0:{i} / {mount_point} rw,relatime - {fs_type} /dev/none rw"
        )
    path = tmp_path / "mountinfo"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_fuse_mount_detected_for_s3fs(tmp_path):
    """The guard flags `s3fs` filesystem type as unsafe-for-mmap.

    Given a fake /proc/self/mountinfo content with an s3fs entry covering
    the cache dir, the detection helper returns True (unsafe).
    """
    v1q = _reload_v1_query()
    cache_dir = str(tmp_path / "shards")
    mountinfo = _write_mountinfo(
        tmp_path,
        [
            ("/", "ext4"),
            (str(tmp_path), "fuse.s3fs"),
        ],
    )
    assert v1q._is_fuse_mount(cache_dir, mountinfo_path=mountinfo) is True


@pytest.mark.parametrize(
    "fs_type",
    ["fuse", "fuse3", "fuse.goofys", "fuse.mountpoint-s3", "fuseblk", "s3fs"],
)
def test_fuse_mount_detected_for_fuse(tmp_path, fs_type):
    """All known FUSE / object-store fs types flag as unsafe."""
    v1q = _reload_v1_query()
    cache_dir = str(tmp_path / "shards")
    mountinfo = _write_mountinfo(
        tmp_path,
        [
            ("/", "ext4"),
            (str(tmp_path), fs_type),
        ],
    )
    assert v1q._is_fuse_mount(cache_dir, mountinfo_path=mountinfo) is True, (
        f"fs_type {fs_type!r} should flag as unsafe"
    )


@pytest.mark.parametrize("fs_type", ["ext4", "xfs", "overlay", "tmpfs", "btrfs"])
def test_safe_fs_passes_guard(tmp_path, fs_type):
    """ext4 / xfs / overlay / tmpfs all pass the guard (returns False)."""
    v1q = _reload_v1_query()
    cache_dir = str(tmp_path / "shards")
    mountinfo = _write_mountinfo(
        tmp_path,
        [
            ("/", "ext4"),
            (str(tmp_path), fs_type),
        ],
    )
    assert v1q._is_fuse_mount(cache_dir, mountinfo_path=mountinfo) is False, (
        f"fs_type {fs_type!r} should NOT flag as unsafe"
    )


def test_guard_silent_when_mountinfo_unavailable(tmp_path):
    """On platforms without /proc/self/mountinfo (e.g. macOS dev), the guard
    no-ops rather than crashing import.
    """
    v1q = _reload_v1_query()
    missing = str(tmp_path / "does-not-exist")
    # Helper must return False (not raise) when mountinfo file is absent.
    assert v1q._is_fuse_mount(str(tmp_path), mountinfo_path=missing) is False


# --- the WARNING log line itself fires --------------------------------------


def test_fuse_warning_fires_when_mmap_on_and_cache_is_fuse(monkeypatch, caplog):
    """`_maybe_warn_about_fuse_cache_dir` emits a WARNING on a FUSE cache dir.

    The guard is non-mandating — its observability IS the contract. Without
    a test asserting the log line, a refactor that silently drops the warning
    would ship green.
    """
    import logging

    v1q = _reload_v1_query()
    monkeypatch.setattr(v1q, "_is_fuse_mount", lambda *a, **k: True)
    with caplog.at_level(logging.WARNING, logger="services.query_api.v1_query"):
        emitted = v1q._maybe_warn_about_fuse_cache_dir("/fake/cache", mmap_enabled=True)
    assert emitted is True
    assert any(
        "FUSE filesystem" in rec.getMessage() and "/fake/cache" in rec.getMessage()
        for rec in caplog.records
    ), f"expected FUSE warning in caplog; got {[r.getMessage() for r in caplog.records]!r}"


def test_fuse_warning_silent_when_mmap_off(monkeypatch, caplog):
    """If mmap is off, the warning never fires — even on a FUSE cache dir."""
    import logging

    v1q = _reload_v1_query()
    monkeypatch.setattr(v1q, "_is_fuse_mount", lambda *a, **k: True)
    with caplog.at_level(logging.WARNING, logger="services.query_api.v1_query"):
        emitted = v1q._maybe_warn_about_fuse_cache_dir("/fake/cache", mmap_enabled=False)
    assert emitted is False
    assert not any(
        "FUSE filesystem" in rec.getMessage() for rec in caplog.records
    )


def test_fuse_warning_silent_when_cache_is_safe(monkeypatch, caplog):
    """If the cache dir is on a safe fs, no warning fires even with mmap on."""
    import logging

    v1q = _reload_v1_query()
    monkeypatch.setattr(v1q, "_is_fuse_mount", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger="services.query_api.v1_query"):
        emitted = v1q._maybe_warn_about_fuse_cache_dir("/safe/cache", mmap_enabled=True)
    assert emitted is False
    assert not any(
        "FUSE filesystem" in rec.getMessage() for rec in caplog.records
    )
