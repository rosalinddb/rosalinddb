"""Unit tests for the Parquet landing writer/reader round-trip.

Loop "rough-edges" — item 3. The writer previously inferred an Arrow struct
from each record's `metadata` dict; a batch where every record had
`metadata: {}` produced an empty `struct<>` that Parquet cannot write, and a
sentinel `__rb_empty__` field was injected to mask it. The writer now stores
`metadata` as a JSON-encoded string column, so an all-`{}` batch (and arbitrary
nested metadata) round-trips with a fixed schema and no sentinel.

Each `write_parquet` call also now produces a uniquely-named part so two writes
under the same prefix accumulate rather than overwrite.
"""
from __future__ import annotations

import pytest

from adapters.storage import storage
from adapters.landing.parquet_writer import write_parquet
from adapters.landing.parquet_reader import read_landing_vectors


@pytest.fixture(autouse=True)
def _clean_memory_store():
    storage.memory_reset()
    yield
    storage.memory_reset()


def test_all_empty_metadata_records_round_trip():
    """A batch where every record has `metadata: {}` writes and reads back."""
    prefix = "memory://rosalinddb/landing/empty-meta"
    records = [
        {"id": "a", "values": [1.0, 2.0, 3.0, 4.0], "metadata": {}},
        {"id": "b", "values": [5.0, 6.0, 7.0, 8.0], "metadata": {}},
    ]
    write_parquet(prefix, records)
    ids, vectors, metas = read_landing_vectors(prefix)

    assert ids == ["a", "b"]
    assert vectors.shape == (2, 4)
    # No sentinel field leaks through — customers see the original `{}` shape.
    assert metas == [{}, {}]


def test_nested_metadata_round_trips():
    """Arbitrary nested/mixed-typed metadata survives the JSON-string column."""
    prefix = "memory://rosalinddb/landing/nested-meta"
    records = [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0],
         "metadata": {"tags": ["a", "b"], "n": 5, "ok": True}},
    ]
    write_parquet(prefix, records)
    _, _, metas = read_landing_vectors(prefix)

    assert metas == [{"tags": ["a", "b"], "n": 5, "ok": True}]


def test_successive_writes_accumulate_under_same_prefix():
    """Two writes to one prefix produce two distinct parts (no overwrite)."""
    prefix = "memory://rosalinddb/landing/accum"
    uri1 = write_parquet(prefix, [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ])
    uri2 = write_parquet(prefix, [
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])

    assert uri1 != uri2
    ids, vectors, _ = read_landing_vectors(prefix)
    assert sorted(ids) == ["a", "b"]
    assert vectors.shape == (2, 4)
