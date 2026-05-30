"""Unit tests for the bench seed_corpus single-tenant + chunked-ingest path.

Pins three contracts on `bench/seed_corpus.py`:

  1. The script accepts a `--single-tenant` flag that forces tenant count to 1
     (the mmap-comparison bench seeds 1 tenant x 1M vectors; the existing
     multi-tenant matrix bench keeps its own semantics untouched).
  2. With `--single-tenant`, the resulting cache file has exactly one entry
     regardless of what was passed for `--tenants`.
  3. The new chunked-ingest helper splits an NDJSON record list at the byte
     budget without splitting individual records, preserves order, and has a
     well-defined behavior for an oversized single record.

`bench/` is not a Python package, so we load `seed_corpus.py` via importlib
spec rather than `import bench.seed_corpus`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


# --- module loader --------------------------------------------------------


def _load_seed_corpus():
    """Load `bench/seed_corpus.py` as a module by file path.

    `bench/` lacks an `__init__.py` (it ships scripts, not a package), so a
    plain `import` won't find it. The module is cached under
    `bench_seed_corpus` so repeated test calls reuse the same object.
    """
    if "bench_seed_corpus" in sys.modules:
        return sys.modules["bench_seed_corpus"]
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "bench" / "seed_corpus.py"
    spec = importlib.util.spec_from_file_location("bench_seed_corpus", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_seed_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- argparse contract ----------------------------------------------------


def test_argparse_accepts_single_tenant_flag():
    """`--single-tenant` parses without raising SystemExit."""
    mod = _load_seed_corpus()
    parser = mod._build_argparser()
    args = parser.parse_args(
        [
            "--dim", "4",
            "--vectors-per", "10",
            "--out", "/tmp/never-written.json",
            "--single-tenant",
        ]
    )
    assert args.single_tenant is True


# --- single-tenant cache shape -------------------------------------------


def test_single_tenant_writes_one_entry_to_cache(tmp_path, monkeypatch):
    """`--single-tenant` produces exactly one cache entry, even if --tenants is high.

    The seeder is monkeypatched to skip all network calls: signup returns a
    fixed fake key; create_dataset, ingest, and wait_indexed all return True.
    The contract under test is the cache-shape behavior of the orchestration
    loop, not any HTTP interaction.
    """
    mod = _load_seed_corpus()

    monkeypatch.setattr(mod, "signup", lambda *a, **kw: "fake-api-key")
    monkeypatch.setattr(mod, "create_dataset", lambda *a, **kw: True)
    monkeypatch.setattr(mod, "ingest", lambda *a, **kw: True)
    monkeypatch.setattr(mod, "wait_indexed", lambda *a, **kw: True)

    out_path = tmp_path / "cache.json"
    rc = mod.main(
        [
            "--dim", "4",
            "--tenants", "50",       # deliberately high; --single-tenant should win
            "--vectors-per", "1",
            "--out", str(out_path),
            "--single-tenant",
        ]
    )
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    entry = data[0]
    assert set(entry.keys()) == {"api_key", "dataset"}
    assert entry["api_key"] == "fake-api-key"


# --- chunked-ingest helper -----------------------------------------------


def _approx_record(i: int, dim: int = 16) -> dict:
    return {"id": f"v{i}", "values": [0.123456 for _ in range(dim)], "metadata": {"category": "books"}}


def test_chunked_ingest_splits_at_size_limit():
    """Each chunk is <= max_bytes; record-boundary is never split."""
    mod = _load_seed_corpus()
    records = [_approx_record(i) for i in range(200)]
    # Serialised record size is fairly uniform; pick a budget that forces
    # several chunks.
    per_line = len(json.dumps(records[0]).encode("utf-8")) + 1  # +1 for newline
    max_bytes = per_line * 7  # ~7 records per chunk

    chunks = list(mod._chunk_ndjson(records, max_bytes))
    assert len(chunks) > 1

    for chunk in chunks:
        body = "\n".join(json.dumps(r) for r in chunk).encode("utf-8")
        assert len(body) <= max_bytes, (
            f"chunk size {len(body)} exceeds max_bytes {max_bytes}"
        )
        assert len(chunk) >= 1


def test_chunked_ingest_preserves_order_and_count():
    """Concatenating chunks back yields the original record list, in order."""
    mod = _load_seed_corpus()
    records = [_approx_record(i) for i in range(123)]
    per_line = len(json.dumps(records[0]).encode("utf-8")) + 1
    max_bytes = per_line * 5

    flat: list[dict] = []
    for chunk in mod._chunk_ndjson(records, max_bytes):
        flat.extend(chunk)

    assert flat == records
    assert [r["id"] for r in flat] == [f"v{i}" for i in range(123)]


def test_chunked_ingest_handles_oversized_single_record():
    """A single record larger than max_bytes is yielded on its own.

    Contract choice: the helper does NOT raise — it yields the oversized
    record as a one-element chunk. The seeder relies on the server to reject
    it with a 413; the helper's job is to avoid silently dropping data.
    """
    mod = _load_seed_corpus()
    big = {"id": "huge", "values": [0.5] * 100_000, "metadata": {}}
    small = _approx_record(0)
    records = [small, big, small]

    chunks = list(mod._chunk_ndjson(records, max_bytes=1024))

    # Recover every record, in order.
    flat: list[dict] = []
    for chunk in chunks:
        flat.extend(chunk)
    assert flat == records

    # The oversized record sits alone in its own chunk.
    big_chunks = [c for c in chunks if big in c]
    assert len(big_chunks) == 1
    assert big_chunks[0] == [big]


def test_chunked_ingest_empty_input():
    """No records in -> no chunks out. Edge case that prevents an empty POST."""
    mod = _load_seed_corpus()
    assert list(mod._chunk_ndjson([])) == []
