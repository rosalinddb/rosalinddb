"""Unit tests for the ``memory://`` storage adapter.

The ``memory://`` adapter is a dict-backed, process-local implementation of the
full storage contract (`adapters/storage/storage.py`). It is the test double
for every unit test: a real adapter that implements the actual interface and
keeps bytes in memory, doing zero filesystem and zero network I/O.

These tests are written TDD-first against the contract:
  - round-trip ``write_bytes`` / ``read_bytes``
  - prefix listing (`list`) and parquet-part discovery
  - overwrite semantics (a second write replaces the first)
  - missing-key behaviour (read/list of an absent key/prefix)
  - ``open_reader`` JSONL/JSON/parquet decode
  - ``presign_get`` passthrough
"""
from __future__ import annotations

import pytest

from adapters.storage import storage


@pytest.fixture(autouse=True)
def _clean_memory_store():
    """Each test starts with an empty in-memory store."""
    storage.memory_reset()
    yield
    storage.memory_reset()


# --- round-trip -----------------------------------------------------------


def test_write_then_read_round_trip():
    storage.write_bytes("memory://bucket/a/b.bin", b"hello-bytes")
    assert storage.read_bytes("memory://bucket/a/b.bin") == b"hello-bytes"


def test_write_empty_payload_round_trips():
    storage.write_bytes("memory://bucket/empty.bin", b"")
    assert storage.read_bytes("memory://bucket/empty.bin") == b""


# --- overwrite ------------------------------------------------------------


def test_second_write_overwrites_first():
    storage.write_bytes("memory://bucket/k", b"first")
    storage.write_bytes("memory://bucket/k", b"second")
    assert storage.read_bytes("memory://bucket/k") == b"second"


# --- missing key ----------------------------------------------------------


def test_read_missing_key_raises():
    # FileNotFoundError *exactly* — the s3:// adapter normalizes botocore's
    # NoSuchKey/404 ClientError to the same type, so a caller branching on the
    # exception type behaves identically on the unit (memory://) and
    # integration (MinIO) tiers. See tests/integration/test_s3_storage.py.
    with pytest.raises(FileNotFoundError):
        storage.read_bytes("memory://bucket/does-not-exist")


def test_list_missing_prefix_returns_empty():
    assert storage.list("memory://bucket/no-such-prefix/") == []


# --- prefix listing -------------------------------------------------------


def test_list_returns_uris_under_prefix():
    storage.write_bytes("memory://bucket/p/one.txt", b"1")
    storage.write_bytes("memory://bucket/p/two.txt", b"2")
    storage.write_bytes("memory://bucket/other/three.txt", b"3")
    found = sorted(storage.list("memory://bucket/p/"))
    assert found == [
        "memory://bucket/p/one.txt",
        "memory://bucket/p/two.txt",
    ]


def test_list_is_recursive():
    storage.write_bytes("memory://bucket/p/sub/deep.txt", b"d")
    storage.write_bytes("memory://bucket/p/top.txt", b"t")
    found = sorted(storage.list("memory://bucket/p"))
    assert found == [
        "memory://bucket/p/sub/deep.txt",
        "memory://bucket/p/top.txt",
    ]


# --- delete ---------------------------------------------------------------


def test_delete_removes_key():
    storage.write_bytes("memory://bucket/k", b"x")
    storage.delete("memory://bucket/k")
    assert storage.list("memory://bucket/") == []
    with pytest.raises(FileNotFoundError):
        storage.read_bytes("memory://bucket/k")


def test_delete_missing_key_is_noop():
    # Deleting an absent key must not raise — mirrors S3 idempotent delete.
    storage.delete("memory://bucket/never-existed")


# --- open_reader ----------------------------------------------------------


def test_open_reader_jsonl_yields_lines():
    storage.write_bytes(
        "memory://bucket/data.jsonl",
        b'{"id":"a"}\n{"id":"b"}\n',
    )
    lines = list(storage.open_reader("memory://bucket/data.jsonl"))
    assert lines == ['{"id":"a"}', '{"id":"b"}']


def test_open_reader_json_yields_single_blob():
    storage.write_bytes("memory://bucket/data.json", b'{"id":"a"}')
    chunks = list(storage.open_reader("memory://bucket/data.json"))
    assert chunks == ['{"id":"a"}']


def test_open_reader_parquet_yields_bytes():
    storage.write_bytes("memory://bucket/data.parquet", b"\x00\x01PAR1")
    chunks = list(storage.open_reader("memory://bucket/data.parquet"))
    assert chunks == [b"\x00\x01PAR1"]


# --- presign --------------------------------------------------------------


def test_presign_get_passthrough():
    storage.write_bytes("memory://bucket/k", b"x")
    assert storage.presign_get("memory://bucket/k", ttl_s=60) == "memory://bucket/k"


# --- presign_put ----------------------------------------------------------


def test_presign_put_returns_url_method_and_content_type():
    target = storage.presign_put("memory://bucket/up/upload.ndjson", expires=60)
    # The fake mirrors a real presigned-PUT target: a single `url`, the `PUT`
    # method, and the exact `content_type` the URL is signed for — no
    # multipart `fields`, since a PUT has no upload policy.
    assert target["url"] == "memory://bucket/up/upload.ndjson"
    assert target["method"] == "PUT"
    assert target["content_type"] == storage.IMPORT_UPLOAD_CONTENT_TYPE
    assert "fields" not in target


def test_presign_put_target_is_writable():
    target = storage.presign_put("memory://bucket/up/upload.ndjson", expires=60)
    storage.write_bytes(target["url"], b"payload")
    assert storage.read_bytes("memory://bucket/up/upload.ndjson") == b"payload"


# --- object_size ----------------------------------------------------------


def test_object_size_returns_byte_count():
    storage.write_bytes("memory://bucket/blob", b"hello world")
    assert storage.object_size("memory://bucket/blob") == 11


def test_object_size_none_when_absent():
    assert storage.object_size("memory://bucket/missing") is None


# --- isolation ------------------------------------------------------------


def test_memory_reset_clears_store():
    storage.write_bytes("memory://bucket/k", b"x")
    storage.memory_reset()
    assert storage.list("memory://bucket/") == []
