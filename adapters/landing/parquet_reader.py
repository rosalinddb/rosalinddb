from __future__ import annotations

"""Parquet landing reader.

Sibling of `parquet_writer.py`. Reads every `.parquet` file under a landing
prefix and returns the concatenated ids/vectors/metadatas. Used by the index
builder to materialise training data for FAISS.

Empty directories yield empty arrays so the builder can short-circuit without
crashing.
"""

import io
import json
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

from adapters.observability.tracing import span as _span
from adapters.storage.storage import _s3_client, _split_s3, read_bytes


def read_shard_sidecar(shard_uri: str) -> Dict[str, dict]:
    """Read the `{shard_uri}.meta.json` sidecar written by the index builder.

    The sidecar maps each FAISS int64 id (as a string key) to the customer's
    original record: `{"<int64>": {"id": "<original>", "metadata": {…}}}`.
    Both the hot query path and the ephemeral runner use this to invert
    FAISS's SHA1-derived hashes back to the uploaded string ids/metadata.

    Returns an empty dict if the sidecar is missing or unreadable so callers
    can degrade gracefully (a hit with no sidecar entry falls back to the
    stringified hash with empty metadata).

    Emits a `shard.load_sidecar` OTel span so the previously-invisible S3 GET
    (7-29 ms on cold queries) is attributable in traces. Previously this was a
    silent gap between the `shard.download` and `faiss.search` spans in Tempo.

    Follow-up (noted, not implemented here): download the sidecar concurrently
    with the main `.bin` shard — both URIs are known at the same time so they
    could be fetched in parallel with asyncio.gather or concurrent.futures.
    Deferred because the reader is currently synchronous and the refactor
    carries structural risk; the span alone closes the observability gap.
    """
    meta_uri = f"{shard_uri}.meta.json"
    with _span("shard.load_sidecar", {"rosalinddb.uri": meta_uri}):
        try:
            if meta_uri.startswith("s3://") or meta_uri.startswith("memory://"):
                raw = read_bytes(meta_uri)
            else:
                return {}
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}


def list_landing_parts(landing_prefix: str) -> List[str]:
    """Return the sorted URIs of every `.parquet` part under `landing_prefix`.

    Exposed for the incremental index builder: the builder compares this list
    against the set of parts already folded into the current shard to decide
    which parts are *new* and need `index.add()`. An empty/missing prefix
    yields `[]`.
    """
    return _list_parquet_parts(landing_prefix)


def read_landing_vectors(
    landing_prefix: str,
) -> Tuple[List[str], np.ndarray, List[dict]]:
    """Read all parquet parts under `landing_prefix`.

    Returns `(ids, vectors, metadatas)` where:
      - ids:        list[str] length N
      - vectors:    np.ndarray, shape (N, D), dtype float32
      - metadatas:  list[dict] length N

    For an empty/missing prefix returns ([], np.empty((0, 0), float32), []).
    Supports `s3://` and `memory://` prefixes via the storage adapter helpers.
    Non-`.parquet` files are silently skipped so co-located JSONL uploads
    (the MVP default) do not break the builder.
    """
    parts = _list_parquet_parts(landing_prefix)
    return read_landing_parts(parts)


def read_landing_parts(
    parts: List[str],
) -> Tuple[List[str], np.ndarray, List[dict]]:
    """Read an explicit list of parquet part URIs into `(ids, vectors, metas)`.

    Sibling of `read_landing_vectors` but takes the part URIs directly rather
    than discovering them under a prefix. The incremental index builder uses
    this to read *only* the not-yet-indexed parts, avoiding the
    rebuild-amplification cost of re-reading every previously indexed upload.

    For an empty list returns `([], np.empty((0, 0), float32), [])`.
    """
    if not parts:
        return [], np.empty((0, 0), dtype=np.float32), []

    all_ids: List[str] = []
    all_vectors: List[np.ndarray] = []
    all_metas: List[dict] = []
    for uri in parts:
        ids, vecs, metas = _read_one(uri)
        all_ids.extend(ids)
        all_vectors.append(vecs)
        all_metas.extend(metas)

    if not all_vectors:
        return [], np.empty((0, 0), dtype=np.float32), []

    stacked = np.concatenate(all_vectors, axis=0) if len(all_vectors) > 1 else all_vectors[0]
    return all_ids, stacked.astype(np.float32, copy=False), all_metas


def _list_parquet_parts(prefix: str) -> List[str]:
    """Return URIs of every `.parquet` object under `prefix` (s3:// or memory://)."""
    if prefix.startswith("s3://"):
        bucket, key_prefix = _split_s3(prefix.rstrip("/"))
        s3 = _s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    out.append(f"s3://{bucket}/{key}")
        return sorted(out)
    if prefix.startswith("memory://"):
        from adapters.storage.storage import list as _storage_list

        return sorted(
            uri for uri in _storage_list(prefix.rstrip("/"))
            if uri.endswith(".parquet")
        )
    raise ValueError(f"Unsupported landing prefix: {prefix}")


def _read_one(uri: str) -> Tuple[List[str], np.ndarray, List[dict]]:
    """Read a single parquet object (s3:// or memory://) into vectors/metadata."""
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf)
    elif uri.startswith("memory://"):
        buf = io.BytesIO(read_bytes(uri))
        table = pq.read_table(buf)
    else:
        raise ValueError(f"Unsupported parquet uri: {uri}")

    ids = [str(x) for x in table.column("id").to_pylist()]
    # `values` is a FixedSizeListArray<float32>; `.to_pylist()` yields list[list[float]].
    values_lists = table.column("values").to_pylist()
    if not values_lists:
        return [], np.empty((0, 0), dtype=np.float32), []
    vectors = np.asarray(values_lists, dtype=np.float32)
    if "metadata" in table.column_names:
        metas = [_decode_metadata(m) for m in table.column("metadata").to_pylist()]
    else:
        metas = [{} for _ in ids]
    return ids, vectors, metas


def _decode_metadata(value) -> dict:
    """Normalise one Parquet `metadata` cell back to a plain dict.

    The landing schema `parquet_writer` emits stores `metadata` as a
    JSON-encoded **string** (a stable schema that round-trips `{}` and arbitrary
    nested JSON). A bulk-import Parquet upload may instead carry `metadata` as
    a native Arrow **struct** (→ a dict from `to_pylist()`) or **map** (→ a
    list of key/value tuples). All three are accepted here so either landing
    source reads back uniformly.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        # pyarrow map column → list[(key, value), ...]
        try:
            return {k: v for k, v in value}
        except (ValueError, TypeError):
            return {}
    return {}
