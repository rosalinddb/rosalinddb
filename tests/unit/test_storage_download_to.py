"""Unit tests for `adapters.storage.storage.download_to`.

`download_to` streams the GET response to disk: it replaces the legacy
`f.write(read_bytes(uri))` pattern for the cache-fill path. ``read_bytes``
materialises the entire object in process memory before returning; on a
multi-GB shard inside a typical DP cgroup limit (2-4 GB), the Python
``bytes`` allocation OOMs the container before any bytes hit disk.
``download_to`` streams the response to disk a chunk at a time (boto3
TransferManager) with bounded RAM regardless of object size.

The tests pin:

  - **Functional parity** with ``read_bytes`` on the ``memory://`` backend
    (the file ends up at the path with the right bytes).
  - **The missing-key contract**: ``FileNotFoundError`` on a URI that does
    not exist, matching what callers (``_classify_hot_path_error``) map to
    HTTP 503.
  - **Streaming behaviour** at the file-handle level for the ``s3://``
    backend: ``download_to`` MUST call boto3's
    ``s3.download_file(Bucket, Key, local_path)`` and MUST NOT call
    ``s3.get_object(...)`` (the latter forces the response body into one
    in-memory blob, which is the bug).
  - **Caller-side atomicity**: the function writes directly to the path
    handed in, not to a temp file. Callers (``_ensure_cached`` and
    ``shard_tier.fetch``) are responsible for the temp-then-rename publish
    step; ``download_to`` only owns the write.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit


# --- memory:// backend ---------------------------------------------------


def test_download_to_writes_memory_payload_to_path(tmp_path, monkeypatch):
    """`download_to` writes the in-memory bytes object to the local path."""
    import adapters.storage.storage as storage_mod

    uri = "memory://bucket/streaming-1.bin"
    payload = b"hello-streamed-bytes"
    storage_mod._MEM_OBJECTS[uri] = payload

    out = str(tmp_path / "out.bin")
    storage_mod.download_to(uri, out)

    assert os.path.exists(out), "download_to must create the local file"
    with open(out, "rb") as f:
        assert f.read() == payload


def test_download_to_overwrites_existing_local_file(tmp_path):
    """A second `download_to` to the same path overwrites cleanly.

    The legacy `_ensure_cached` writes to a unique `*.tmp` path then
    `os.replace`s — so collisions on the FINAL path are atomic at the
    rename layer. But the temp path itself could collide on a retry,
    and `download_to` must overwrite rather than refuse / leak.
    """
    import adapters.storage.storage as storage_mod

    uri = "memory://bucket/overwrite.bin"
    storage_mod._MEM_OBJECTS[uri] = b"first-payload"
    out = str(tmp_path / "out.bin")
    storage_mod.download_to(uri, out)
    with open(out, "rb") as f:
        assert f.read() == b"first-payload"

    # New payload, same path. Should overwrite.
    storage_mod._MEM_OBJECTS[uri] = b"second-payload-NEW"
    storage_mod.download_to(uri, out)
    with open(out, "rb") as f:
        assert f.read() == b"second-payload-NEW"


def test_download_to_raises_file_not_found_on_missing_memory_key(tmp_path):
    """A missing memory:// key raises `FileNotFoundError`.

    Matches `read_bytes`'s shape so the hot-path classifier
    (`_classify_hot_path_error` -> `storage_unavailable` -> 503) routes
    the same way regardless of which adapter the caller used.
    """
    import adapters.storage.storage as storage_mod

    uri = "memory://bucket/no-such-key.bin"
    out = str(tmp_path / "out.bin")
    with pytest.raises(FileNotFoundError) as exc_info:
        storage_mod.download_to(uri, out)
    assert "memory://" in str(exc_info.value)
    # No partial file should have been written on the error path.
    assert not os.path.exists(out)


def test_download_to_rejects_unsupported_scheme(tmp_path):
    """A `file://` or HTTP URI is rejected like `read_bytes` does."""
    import adapters.storage.storage as storage_mod

    with pytest.raises(ValueError):
        storage_mod.download_to(
            "http://example.com/shard.bin", str(tmp_path / "out.bin")
        )
    with pytest.raises(ValueError):
        storage_mod.download_to(
            "file:///tmp/shard.bin", str(tmp_path / "out.bin")
        )


# --- s3:// backend (the load-bearing one) --------------------------------


def test_download_to_calls_boto_download_file_not_get_object(
    tmp_path, monkeypatch,
):
    """`download_to` MUST use boto3's streaming `download_file`.

    The bug this PR fixes: `read_bytes` calls `get_object(...).read()`
    which materialises the full object in RAM. The fix: route through
    `s3.download_file(Bucket, Key, Filename)` which streams via boto3's
    TransferManager. This test pins the contract at the API-call level so
    a future "let's use get_object then write" refactor regression-tests
    immediately.

    Asserts:
      - `s3.download_file` IS called, with the right (Bucket, Key, path).
      - `s3.get_object` is NOT called by `download_to` itself.
    """
    import adapters.storage.storage as storage_mod

    fake_s3 = MagicMock()
    monkeypatch.setattr(storage_mod, "_s3_client", lambda: fake_s3)

    out = str(tmp_path / "shard.bin")
    storage_mod.download_to("s3://my-bucket/path/to/shard.bin", out)

    fake_s3.download_file.assert_called_once_with(
        "my-bucket", "path/to/shard.bin", out,
    )
    fake_s3.get_object.assert_not_called()


def test_download_to_maps_s3_nosuchkey_to_filenotfound(tmp_path, monkeypatch):
    """An S3 `NoSuchKey` (or 404) becomes `FileNotFoundError`.

    Mirrors `_s3_get_object`'s mapping so callers can branch on
    `FileNotFoundError` regardless of which storage method they invoked.
    """
    import adapters.storage.storage as storage_mod
    from botocore.exceptions import ClientError

    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
    )
    monkeypatch.setattr(storage_mod, "_s3_client", lambda: fake_s3)

    out = str(tmp_path / "missing.bin")
    with pytest.raises(FileNotFoundError) as exc_info:
        storage_mod.download_to("s3://b/missing.bin", out)
    assert "s3://" in str(exc_info.value)


def test_download_to_propagates_non_404_client_errors(tmp_path, monkeypatch):
    """A throttling / connection ClientError propagates untouched.

    The classifier upstream knows how to map these to `storage_unavailable`
    via the existing botocore branch; `download_to` does not catch them
    or convert them.
    """
    import adapters.storage.storage as storage_mod
    from botocore.exceptions import ClientError

    fake_s3 = MagicMock()
    throttle_err = ClientError(
        {
            "Error": {
                "Code": "TooManyRequests",
                "Message": "Please reduce your request rate",
            }
        },
        "GetObject",
    )
    fake_s3.download_file.side_effect = throttle_err
    monkeypatch.setattr(storage_mod, "_s3_client", lambda: fake_s3)

    out = str(tmp_path / "throttled.bin")
    with pytest.raises(ClientError) as exc_info:
        storage_mod.download_to("s3://b/throttled.bin", out)
    assert exc_info.value.response["Error"]["Code"] == "TooManyRequests"
