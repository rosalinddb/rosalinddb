from __future__ import annotations

"""Parquet landing writer.

Transforms validated records into a compact Parquet part and writes it to a
dataset landing prefix via the storage adapter.

Schema
------
A landing part has three columns:
  - ``id``       : string
  - ``values``   : fixed-size list<float32>
  - ``metadata`` : string — the per-record metadata object, JSON-encoded.

``metadata`` is a JSON **string** column rather than an inferred Arrow struct.
Inferring a struct from per-record dicts is fragile: a batch in which every
record has ``metadata: {}`` infers an empty ``struct<>`` which Parquet cannot
write at all, and arbitrary nested/mixed-typed customer metadata produces an
unstable schema that varies batch to batch. Encoding the object as a JSON
string makes the column schema fixed and able to round-trip *any* JSON value —
including ``{}`` — losslessly. ``parquet_reader`` decodes it back to a dict.

Part naming
-----------
Each call writes a uniquely named part (``part-<uuid>.parquet``) so successive
writes under the same prefix accumulate rather than overwrite each other. The
index builder discovers parts by listing the prefix, so a unique name per part
is required for it to see every upload.
"""

import io
import json
import uuid
from typing import Iterable, List, Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adapters.storage.storage import write_bytes


def write_parquet(dataset_uri_prefix: str, records: Iterable[Dict[str, Any]]) -> str:
    """Write records to a uniquely-named Parquet part under the landing prefix.

    Args:
        dataset_uri_prefix: Landing prefix (e.g., s3://..., memory://...)
        records: Iterable of validated records with id, values, metadata.

    Returns:
        The URI that was written to.
    """
    table = _records_to_table(list(records))
    buf = io.BytesIO()
    pq.write_table(table, buf)

    uri = _next_part_uri(dataset_uri_prefix)
    write_bytes(uri, buf.getvalue())
    return uri


def _records_to_table(records: List[Dict[str, Any]]) -> pa.Table:
    """Convert a list of dict records into a PyArrow table.

    `metadata` is JSON-encoded into a string column so an all-`{}` batch (and
    arbitrary nested customer metadata) writes to Parquet with a stable schema.
    """
    ids = [r["id"] for r in records]
    vectors = [np.array(r["values"], dtype=np.float32) for r in records]
    metas = [json.dumps(r.get("metadata", {}) or {}) for r in records]
    vec_fixed = pa.FixedSizeListArray.from_arrays(
        pa.array(np.concatenate(vectors)) if vectors else pa.array([], pa.float32()),
        len(vectors[0]) if vectors else 0,
    )
    return pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "values": vec_fixed,
            "metadata": pa.array(metas, type=pa.string()),
        }
    )


def _next_part_uri(prefix: str) -> str:
    """Return a unique part URI under a prefix.

    Each part gets a `uuid`-suffixed name so two writes under the same prefix
    never collide — the index builder lists the prefix to discover parts, so a
    fixed filename would silently drop every upload but the last.
    """
    part = f"part-{uuid.uuid4().hex}.parquet"
    if prefix.endswith("/"):
        return prefix + part
    return prefix + "/" + part
