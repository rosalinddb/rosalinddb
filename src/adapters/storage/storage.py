from __future__ import annotations

"""Storage adapter abstraction for object storage.

RosalindDB is object-storage-first: every deployment runs on object storage
(MinIO for local/CI; any S3-compatible store otherwise). This module provides
a minimal, portable interface over those backends using URI schemes:

- ``s3://bucket/key``          — S3 / MinIO (any S3-compatible store)
- ``memory://bucket/key``      — dict-backed, process-local (unit-test double)
- ``http(s)://host/path``      — read-only passthrough for external sources

There is deliberately **no** ``file://`` backend. Nobody deploys RosalindDB on a
local filesystem, and a ``file://`` test double let OS-specific filesystem
quirks (path separators, permission models, a hardcoded ``/var/landing`` that is
not writable on macOS) leak into the suite. Unit tests use ``memory://`` — a
real adapter that implements this exact contract but keeps bytes in a dict;
integration tests use a real MinIO container via ``testcontainers``.

Functions include streaming reads, byte writes, prefix listing, delete,
existence checks, object size (`head`), and presigned GET / presigned-PUT
upload generation. These utilities are used across services for landing
writes, index shard IO, cache hydration, and the async bulk-import staging
flow.
"""

import threading
from typing import Dict, Iterator, List, Optional, Tuple

import boto3
import requests
from botocore.exceptions import ClientError

from adapters import config


# --- memory:// store ------------------------------------------------------
#
# The ``memory://`` adapter is a real implementation of this module's contract,
# not a ``mock.patch``. It keeps every written object as raw bytes in a single
# process-wide dict keyed by the full ``memory://...`` URI. A lock guards the
# dict so concurrent writers (the dev-harness daemon threads) stay consistent.
# It mirrors the ``memory://`` *state* adapter pattern: process-local, default
# for tests, reset between tests via ``memory_reset()``.
_MEM_OBJECTS: Dict[str, bytes] = {}
_MEM_LOCK = threading.Lock()


def memory_reset() -> None:
    """Drop every object from the in-memory store.

    Test hook: unit tests call this between cases so the dict-backed adapter
    starts empty, exactly as the ``memory://`` state adapter is reset by
    clearing its module-level dicts.
    """
    with _MEM_LOCK:
        _MEM_OBJECTS.clear()


def _detect_type_from_uri(uri: str) -> str:
    """Infer a coarse file type from URI suffix.

    Returns one of: "jsonl", "json", "parquet", or "unknown".
    """
    lower = uri.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".parquet"):
        return "parquet"
    return "unknown"


def _s3_missing_key(exc: ClientError) -> bool:
    """Return True if a botocore ``ClientError`` means "object not found".

    S3 surfaces a missing key as ``NoSuchKey``; MinIO and presigned/HEAD-style
    requests can surface it as a plain ``404``. Both must normalize to
    ``FileNotFoundError`` so the ``s3://`` and ``memory://`` adapters raise the
    *same* exception type for an absent key.
    """
    err = exc.response.get("Error", {})
    code = err.get("Code")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("NoSuchKey", "404", "NoSuchObject") or status == 404


def _s3_get_object(bucket: str, key: str):
    """Fetch an S3 object, re-raising a missing key as ``FileNotFoundError``.

    Centralizes the botocore ``ClientError`` -> ``FileNotFoundError`` mapping so
    every S3 read path (``read_bytes``, ``open_reader``) reports a missing key
    identically to the ``memory://`` adapter.
    """
    try:
        return _s3_client().get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _s3_missing_key(exc):
            raise FileNotFoundError(f"s3:// key not found: s3://{bucket}/{key}") from exc
        raise


def open_reader(uri: str, file_type: Optional[str] = None) -> Iterator[bytes | str]:
    """Yield contents of a resource identified by a URI.

    - For JSONL: yields one line (str) per record
    - For JSON: yields a single JSON string
    - For Parquet: yields a single bytes blob

    Args:
        uri: Resource URI. Supports s3://, memory://, http(s)://
        file_type: Optional override. If omitted, inferred from URI.

    Yields:
        str or bytes chunks as described above depending on file type.
    """
    ftype = file_type or _detect_type_from_uri(uri)
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        obj = _s3_get_object(bucket, key)
        body = obj["Body"].read()
        yield from _decode_body(body, ftype)
    elif uri.startswith("memory://"):
        body = read_bytes(uri)
        yield from _decode_body(body, ftype)
    elif uri.startswith("http://") or uri.startswith("https://"):
        r = requests.get(uri, stream=True, timeout=30)
        r.raise_for_status()
        if ftype in ("json", "jsonl"):
            for line in r.iter_lines():
                if line:
                    yield line.decode("utf-8")
        else:
            yield r.content
    else:
        raise ValueError(f"Unsupported URI: {uri}")


def _decode_body(body: bytes, ftype: str) -> Iterator[bytes | str]:
    """Decode a fully-read object body according to ``ftype``.

    Shared by the s3:// and memory:// read paths so both schemes apply
    identical JSONL/JSON/parquet framing.
    """
    if ftype in ("json", "jsonl"):
        text = body.decode("utf-8")
        if ftype == "jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield line
        else:
            yield text
    else:
        yield body


def write_bytes(uri: str, data: bytes) -> None:
    """Write raw bytes to a URI destination (s3:// or memory://)."""
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    elif uri.startswith("memory://"):
        with _MEM_LOCK:
            _MEM_OBJECTS[uri] = bytes(data)
    else:
        raise ValueError(f"Unsupported write URI: {uri}")


def read_bytes(uri: str) -> bytes:
    """Read raw bytes from a URI source (s3:// or memory://).

    Counterpart of `write_bytes`. Used by the index builder's incremental
    path to load a previously written FAISS shard back into memory so new
    vectors can be `index.add()`-ed onto it without a full rebuild.

    Raises ``FileNotFoundError`` if the key is absent. Both backends report a
    missing key identically: the ``memory://`` adapter raises it directly, and
    the ``s3://`` path normalizes botocore's ``NoSuchKey``/404 ``ClientError``
    into ``FileNotFoundError`` (see ``_s3_get_object``).

    Caller-contract note: this function **materialises the entire object in
    process memory** before returning. Safe for small objects (sidecars,
    metadata JSON, individual FAISS shards on the index-builder's incremental
    rebuild path where the bytes are needed in-memory anyway). For
    streaming an object into a local file — especially a multi-GB shard
    on the query DP's cache-fill path — use ``download_to`` instead. The
    DP container's memory cap is typically a few GB; ``read_bytes`` on a
    6 GB shard OOMs the container before any of those bytes hit disk.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        obj = _s3_get_object(bucket, key)
        return obj["Body"].read()
    elif uri.startswith("memory://"):
        with _MEM_LOCK:
            if uri not in _MEM_OBJECTS:
                raise FileNotFoundError(f"memory:// key not found: {uri}")
            return _MEM_OBJECTS[uri]
    else:
        raise ValueError(f"Unsupported read URI: {uri}")


def download_to(uri: str, local_path: str) -> None:
    """Stream the object at ``uri`` into ``local_path`` without buffering in RAM.

    Solves the "buffer-the-world download" footgun the 6 GB shard bench
    exposed: ``read_bytes`` + ``f.write`` for a multi-GB object forces the
    Python process to hold the entire response body in memory before the
    first byte hits disk, and a typical DP container cap (2-4 GB) OOMs.
    ``download_to`` uses boto3's TransferManager (under the hood of
    ``s3.download_file``), which streams the GET response to disk a chunk
    at a time with bounded RAM (default 8 MB per chunk, 10 concurrent
    chunks => ~80 MB peak).

    Atomicity is the **caller's** responsibility — this function writes
    directly to ``local_path``. The caller should write to a unique
    ``.tmp`` path and ``os.replace`` to publish atomically, so a partial
    file is never visible to a concurrent reader.

    Errors:
    - ``FileNotFoundError`` if the object does not exist (same shape as
      ``read_bytes`` to keep the classifier branches stable).
    - ``OSError`` for disk-side failures (full disk, permission).
    - Any other ``ClientError`` propagates (transient throttling, network).

    ``memory://`` parity: writes the in-process bytes object to disk so
    the unit suite can exercise the cache-fill path without a real S3.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        try:
            _s3_client().download_file(bucket, key, local_path)
        except ClientError as exc:
            if _s3_missing_key(exc):
                raise FileNotFoundError(
                    f"s3:// key not found: s3://{bucket}/{key}"
                ) from exc
            raise
    elif uri.startswith("memory://"):
        # The memory backend already holds the bytes in RAM — there's no
        # streaming win to be had — but writing the file matches the
        # contract callers depend on: "after this returns, the bytes are
        # at local_path".
        with _MEM_LOCK:
            if uri not in _MEM_OBJECTS:
                raise FileNotFoundError(f"memory:// key not found: {uri}")
            payload = _MEM_OBJECTS[uri]
        with open(local_path, "wb") as f:
            f.write(payload)
    else:
        raise ValueError(f"Unsupported read URI: {uri}")


def delete(uri: str) -> None:
    """Delete an object at a URI (s3:// or memory://).

    Idempotent: deleting an absent key is a no-op, matching S3's
    ``DeleteObject`` semantics.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        _s3_client().delete_object(Bucket=bucket, Key=key)
    elif uri.startswith("memory://"):
        with _MEM_LOCK:
            _MEM_OBJECTS.pop(uri, None)
    else:
        raise ValueError(f"Unsupported delete URI: {uri}")


def list(prefix: str) -> List[str]:
    """List objects under a prefix.

    Args:
        prefix: A prefix URI (s3://bucket/prefix or memory://bucket/prefix)

    Returns:
        A list of fully qualified URIs for matching objects.
    """
    if prefix.startswith("s3://"):
        bucket, key_prefix = _split_s3(prefix)
        s3 = _s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                results.append(f"s3://{bucket}/" + obj["Key"])
        return results
    elif prefix.startswith("memory://"):
        with _MEM_LOCK:
            # NOTE: this does a raw ``startswith`` on the full ``memory://...``
            # URI, whereas real S3 matches ``Prefix`` against the *key* only.
            # The two coincide only when ``prefix`` is a fully-qualified
            # ``scheme://bucket/path`` string — which every call site passes.
            # The fake is therefore faithful for fully-qualified prefixes; do
            # not rely on it for bare keys or bucket-relative prefixes.
            return [uri for uri in _MEM_OBJECTS if uri.startswith(prefix)]
    else:
        raise ValueError(f"Unsupported list prefix: {prefix}")


def presign_get(uri: str, ttl_s: int) -> str:
    """Return a presigned GET URL or passthrough link for a resource.

    For s3:// URIs, generates a time-limited presigned URL. For memory:// and
    http(s):// URIs, returns the input unchanged.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        return _s3_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl_s,
        )
    elif uri.startswith("http://") or uri.startswith("https://"):
        return uri
    elif uri.startswith("memory://"):
        return uri
    else:
        raise ValueError(f"Unsupported presign URI: {uri}")


# The Content-Type a bulk-import upload is presigned for. A presigned PUT URL
# is a SigV4 signature over a fixed set of parameters; if the client sends a
# `Content-Type` header that was NOT part of the signed params, S3/MinIO/R2
# reject the PUT with `403 SignatureDoesNotMatch`. So the URL is signed *with*
# this Content-Type and the client MUST send exactly this header. The import
# is a binary blob (NDJSON or Parquet); `application/octet-stream` is correct
# and uniform for both.
IMPORT_UPLOAD_CONTENT_TYPE = "application/octet-stream"


def presign_put(uri: str, expires: int) -> dict:
    """Return a presigned-PUT upload target for `uri`.

    The result is `{"url": str, "method": "PUT", "content_type": str}` — a
    client (or browser) does a single `PUT url` with the file as the *raw
    request body* (no multipart form, no fields) and a `Content-Type` header
    set to `content_type`. The object lands at `uri` directly in object
    storage, with no bytes flowing through the application. This is the bulk
    import path — the small `POST .../vectors` endpoint keeps its in-app
    byte cap.

    Why PUT and not POST: presigned PUT works against AWS S3, R2, MinIO, and
    other S3-compatible stores; presigned POST is not universal — some
    backends do not implement it (returning ``501 NotImplemented``). The
    trade-off is that a presigned PUT URL carries no upload *policy*, so it
    cannot enforce a `content-length-range` size cap server-side the way a
    POST policy did. That cap is now re-homed: the import worker
    (`process_import`) `head`s the staged object and fails the job if it
    exceeds `max_bytes`. See `services/validator_worker/run.py`.

    The URL is signed *with* a fixed `Content-Type` (`content_type` in the
    result) — the client must send exactly that header, because a presigned
    PUT signature covers the headers it was minted for and an unsigned/extra
    `Content-Type` would be rejected `403 SignatureDoesNotMatch`.

    - ``s3://`` — a real `generate_presigned_url("put_object", ...)` from boto3.
    - ``memory://`` — a faithful fake: there is no HTTP server, so `url` is the
      `memory://...` object key itself. The unit-test client "uploads" by
      calling `write_bytes(url, data)`, exactly mirroring what a browser PUT
      to MinIO/R2 would do. This keeps the import flow exercisable with zero
      network I/O.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        url = _s3_client().generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": IMPORT_UPLOAD_CONTENT_TYPE,
            },
            ExpiresIn=int(expires),
        )
        return {
            "url": url,
            "method": "PUT",
            "content_type": IMPORT_UPLOAD_CONTENT_TYPE,
        }
    elif uri.startswith("memory://"):
        return {
            "url": uri,
            "method": "PUT",
            "content_type": IMPORT_UPLOAD_CONTENT_TYPE,
        }
    else:
        raise ValueError(f"Unsupported presign_put URI: {uri}")


def object_size(uri: str) -> Optional[int]:
    """Return the byte size of the object at `uri`, or None if it is absent.

    Used by the import worker to enforce the bulk-upload size cap that a
    presigned-PUT URL — unlike the old presigned-POST policy — cannot enforce
    server-side. ``head_object`` / a dict lookup is O(1); no bytes are read.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        try:
            head = _s3_client().head_object(Bucket=bucket, Key=key)
            return int(head["ContentLength"])
        except ClientError as exc:
            if _s3_missing_key(exc):
                return None
            raise
    elif uri.startswith("memory://"):
        with _MEM_LOCK:
            data = _MEM_OBJECTS.get(uri)
            return None if data is None else len(data)
    else:
        raise ValueError(f"Unsupported object_size URI: {uri}")


def exists(uri: str) -> bool:
    """Return True if an object exists at `uri` (s3:// or memory://).

    Used by the import `complete` step to verify the client actually uploaded
    the staged object before transitioning the job to `validating`.
    """
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if _s3_missing_key(exc):
                return False
            raise
    elif uri.startswith("memory://"):
        with _MEM_LOCK:
            return uri in _MEM_OBJECTS
    else:
        raise ValueError(f"Unsupported exists URI: {uri}")


def _split_s3(uri: str) -> Tuple[str, str]:
    """Split an s3://bucket/key URI into (bucket, key)."""
    no_scheme = uri[len("s3://") :]
    bucket, key = no_scheme.split("/", 1)
    return bucket, key


# --- cached boto3 S3 client ----------------------------------------------
#
# A boto3 client is expensive to build (it loads service-model JSON, resolves
# the endpoint, wires up credential providers, and creates a fresh HTTP
# connection pool) and boto3 clients are documented to be thread-safe and
# meant to be created once and reused. Every object-storage operation here
# went through ``_s3_client()``, so rebuilding it per call also paid a fresh
# TLS handshake to the configured S3-compatible backend each time.
#
# The client's config comes from ``S3_*`` env vars that are fixed for the life
# of the process, so a lazily-initialised process-wide singleton is correct.
# It is built on first use (NOT at import time — tests and some environments
# import this module with no S3 configured) and the first construction is
# guarded by a lock so concurrent threads on a cold process build exactly one.
_S3_CLIENT = None
_S3_CLIENT_LOCK = threading.Lock()


def _reset_s3_client() -> None:
    """Drop the cached S3 client so the next call rebuilds it.

    Test hook only: lets the suite force a fresh client after changing the
    ``S3_*`` env vars. Not used by production code, where the config is fixed
    for the life of the process.
    """
    global _S3_CLIENT
    with _S3_CLIENT_LOCK:
        _S3_CLIENT = None


def _s3_client():
    """Return the process-wide boto3 S3 client, building it once on first use.

    Honors the ``S3_*`` env-var overrides used to point at MinIO in dev/CI.
    """
    global _S3_CLIENT
    # Fast path: an already-built client is read without taking the lock.
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    with _S3_CLIENT_LOCK:
        # Re-check inside the lock: a thread that lost the race must reuse the
        # client the winner built rather than construct a second one.
        if _S3_CLIENT is None:
            endpoint_url = config.s3_endpoint_url()
            access_key = config.s3_access_key()
            secret_key = config.s3_secret_key()
            region = config.s3_region()
            _S3_CLIENT = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
        return _S3_CLIENT
