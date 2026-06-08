from __future__ import annotations

"""Embedded in-process numpy recall backend (the memtable).

The single-process / no-docker recall tier. When `RB_RECALL` is on but no
`RB_RECALL_DSN` is configured (the all-in-one eval default), the recall_*
interface in `adapters.recall.__init__` dispatches HERE instead of to pgvector.
This module reimplements the FULL recall surface over a process-local dict +
numpy, with semantics BYTE-IDENTICAL to the pgvector path so the query union
(`recall_search`) and the consolidation fold (`recall_snapshot_for_consolidation`
+ `recall_trim`) behave the same regardless of backend.

Storage model — a `_MemRecall` singleton holding:

  * `_lock`  — a single `RLock` guarding ALL mutation + the snapshot COPY of a
    partition. The lock is held ONLY long enough to mutate or to copy a
    partition's rows; the numpy L2 math / suppress-match split runs OUTSIDE the
    lock so a query never serialises behind another query's distance compute.
  * `_parts` — `{(tenant, dataset): {id: _Row}}`. `_Row` is a plain dict
    `{values, metadata, lsn, deleted, created_at}`.
  * `_lsn`   — `{(tenant, dataset): int}`, the monotonic per-partition LSN
    counter (mirrors `recall_lsn_seq`).

PARITY CONTRACT with the pgvector path (`adapters.recall.__init__`):

  * LSN allocation is a contiguous block reserved ATOMICALLY under the lock, in
    input order (mirrors the single `last_lsn = last_lsn + N RETURNING` upsert).
  * upsert is last-write-wins per (tenant, dataset, id) with intra-batch dedup
    (last occurrence wins), and clears any prior tombstone (deleted -> False).
  * search is an exact L2-SQUARED brute force over rows with `lsn > watermark`,
    deriving `suppress_ids` (EVERY row above the watermark) and `matches` (live +
    filter-pass + top_k) from ONE snapshot of the partition copied under the lock
    — the b1 single-MVCC-snapshot anti-over-suppression property.
  * delete writes a tombstone with a FRESH lsn strictly above the partition max
    (read-your-deletes), dim-matched zero-vector placeholder.
  * snapshot_for_consolidation copies the WHOLE partition atomically under the
    lock and returns `(max_lsn, rows asc by lsn)`; a write that lands after the
    copy has a higher lsn and is excluded (single-snapshot).
  * the metadata filter reuses `_metadata_matches_filter` from the package so
    equality (type+value, null never matches) is byte-identical.

Scoring is computed in FLOAT32 (`float(np.sum((q - v) ** 2))`) to match the FAISS
L2-squared distances the cold path produces, so the union ranks recall vs cold
consistently.
"""

import datetime as _dt
import threading
from typing import List, Optional, Tuple

import numpy as np

# Reuse the package's filter predicate VERBATIM so AND-of-equals (type+value,
# null never matches, empty filter matches all) is byte-identical to the cold +
# pgvector recall paths. Imported lazily inside the call to avoid any import-
# order surprise (`memtable` is imported BY the package's dispatch).


def _filter_matches(metadata: dict, flt: dict) -> bool:
    from adapters.recall import _metadata_matches_filter

    return _metadata_matches_filter(metadata, flt)


class _MemRecall:
    """Process-wide singleton store for the embedded recall memtable.

    All state is class-level so the dispatch in `adapters.recall` reaches one
    shared store regardless of how the module is referenced. `_reset()` clears it
    (a test hook mirroring the memory:// state/storage reset idiom).
    """

    _lock = threading.RLock()
    # {(tenant, dataset): {id: {"values", "metadata", "lsn", "deleted", "created_at"}}}
    _parts: dict = {}
    # {(tenant, dataset): int} monotonic per-partition LSN counter
    _lsn: dict = {}


def _reset() -> None:
    """Clear the whole memtable (test hook). Mirrors memory:// store reset."""
    with _MemRecall._lock:
        _MemRecall._parts = {}
        _MemRecall._lsn = {}


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _alloc_block(key, n: int) -> int:
    """Reserve a contiguous block of `n` LSNs under the lock; return the LAST lsn.

    Mirrors the pgvector `last_lsn = last_lsn + N RETURNING last_lsn` allocation:
    the block is `last_lsn - n + 1 .. last_lsn`, strictly monotonic. MUST be
    called with `_MemRecall._lock` already held.
    """
    cur = _MemRecall._lsn.get(key, 0)
    cur += n
    _MemRecall._lsn[key] = cur
    return cur


def recall_upsert_vectors(tenant_id: str, dataset: str, records: List[dict]) -> int:
    """Embedded mirror of pgvector `recall_upsert_vectors`. Returns the count."""
    if not records:
        return 0
    key = (tenant_id, dataset)
    # Intra-batch dedup: last occurrence wins, surviving id keeps its LAST
    # position (so the LSN block is assigned in winning-record order). del-then-
    # set moves a repeated id to the end (mirrors the pgvector dedup exactly).
    deduped: dict = {}
    for rec in records:
        rid = rec["id"]
        if rid in deduped:
            del deduped[rid]
        deduped[rid] = rec
    winners = list(deduped.values())
    n = len(winners)
    now = _now()
    with _MemRecall._lock:
        last_lsn = _alloc_block(key, n)
        first_lsn = last_lsn - n + 1
        part = _MemRecall._parts.setdefault(key, {})
        for offset, rec in enumerate(winners):
            part[rec["id"]] = {
                # Store as float32 list so scoring matches FAISS L2-squared.
                "values": [float(v) for v in rec["values"]],
                "metadata": dict(rec.get("metadata") or {}),
                "lsn": first_lsn + offset,
                "deleted": False,
                "created_at": now,
            }
    return n


def _snapshot_above(key, watermark: int) -> List[dict]:
    """Copy (under the lock) every row in a partition with `lsn > watermark`.

    Returns a fresh list of row COPIES with their id, so the caller can score /
    split OUTSIDE the lock without a torn read. This is the single-snapshot
    primitive that preserves the b1 suppress-superset-of-matches property.
    """
    with _MemRecall._lock:
        part = _MemRecall._parts.get(key, {})
        return [
            {
                "id": rid,
                "values": row["values"],
                "metadata": row["metadata"],
                "lsn": row["lsn"],
                "deleted": row["deleted"],
            }
            for rid, row in part.items()
            if row["lsn"] > watermark
        ]


def recall_search(
    tenant_id: str,
    dataset: str,
    vector: List[float],
    top_k: int,
    watermark: int,
    flt: Optional[dict] = None,
) -> Tuple[set, List[dict]]:
    """Embedded mirror of pgvector `recall_search` (exact L2-squared brute force).

    Returns `(suppress_ids, matches)` from ONE partition snapshot taken under the
    lock: `suppress_ids` is EVERY id above the watermark (live AND tombstoned);
    `matches` is up to `top_k` filter-passing LIVE rows ascending by L2-squared
    score. Tombstones never match; a live row failing the filter or ranked past
    `top_k` still suppresses its cold twin but is not a match.
    """
    flt = flt or {}
    # SINGLE snapshot taken under the lock; everything below derives from it.
    snapshot = _snapshot_above((tenant_id, dataset), watermark)

    q = np.asarray(vector, dtype=np.float32)
    suppress_ids: set = set()
    scored: List[dict] = []  # live, filter-passing candidates with score
    for row in snapshot:
        rid = row["id"]
        # Every returned id suppresses its cold twin, ALWAYS — before any
        # match/filter/top_k decision (tombstone, filter-fail, past-top_k all
        # still drop their stale cold copy).
        suppress_ids.add(rid)
        # Tombstone: NEVER a match (split on the explicit flag, not by distance).
        if row["deleted"]:
            continue
        meta = row["metadata"] or {}
        if flt and not _filter_matches(meta, flt):
            continue
        v = np.asarray(row["values"], dtype=np.float32)
        score = float(np.sum((q - v) ** 2))
        scored.append({"id": rid, "score": score, "metadata": meta, "deleted": False})

    # Rank the live, filter-passing candidates by ascending score and truncate to
    # top_k (the recall set is small by construction). Stable by id for ties so
    # the result is deterministic.
    scored.sort(key=lambda m: (m["score"], m["id"]))
    matches = scored[: max(0, top_k)]
    return suppress_ids, matches


def recall_get_vector(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict]]:
    """Embedded mirror of pgvector `recall_get_vector` (tri-state point lookup)."""
    with _MemRecall._lock:
        row = _MemRecall._parts.get((tenant_id, dataset), {}).get(vector_id)
        if row is None or row["lsn"] <= watermark:
            return None, None
        deleted = row["deleted"]
        meta = None if deleted else dict(row["metadata"] or {})
    if deleted:
        return "tombstone", None
    return "live", meta


def recall_get_vector_with_embedding(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict], Optional[List[float]]]:
    """Embedded mirror of `recall_get_vector_with_embedding` (adds the embedding)."""
    with _MemRecall._lock:
        row = _MemRecall._parts.get((tenant_id, dataset), {}).get(vector_id)
        if row is None or row["lsn"] <= watermark:
            return None, None, None
        deleted = row["deleted"]
        if deleted:
            return "tombstone", None, None
        meta = dict(row["metadata"] or {})
        values = [float(v) for v in row["values"]]
    return "live", meta, values


def recall_list_rows(
    tenant_id: str, dataset: str, watermark: int
) -> Tuple[List[dict], set]:
    """Embedded mirror of pgvector `recall_list_rows` (single-snapshot split)."""
    snapshot = _snapshot_above((tenant_id, dataset), watermark)
    suppress_ids: set = set()
    live_rows: List[dict] = []
    for row in snapshot:
        suppress_ids.add(row["id"])
        if row["deleted"]:
            continue
        live_rows.append({"id": row["id"], "metadata": row["metadata"] or {}})
    return live_rows, suppress_ids


def recall_delete_vector(
    tenant_id: str, dataset: str, vector_id: str, dimension: int
) -> int:
    """Embedded mirror of pgvector `recall_delete_vector`.

    Upserts a tombstone (`deleted=True`) with a FRESH lsn strictly above the
    partition max (read-your-deletes), dim-matched zero-vector placeholder.
    Returns the allocated lsn.
    """
    key = (tenant_id, dataset)
    placeholder = [0.0] * max(1, int(dimension))
    now = _now()
    with _MemRecall._lock:
        lsn = _alloc_block(key, 1)  # one fresh lsn strictly above the current max
        part = _MemRecall._parts.setdefault(key, {})
        existing = part.get(vector_id)
        # On conflict, leave the existing embedding untouched (mirror pgvector's
        # `DO UPDATE SET lsn=..., deleted=TRUE` which does not change embedding).
        values = existing["values"] if existing is not None else placeholder
        part[vector_id] = {
            "values": values,
            # pgvector tombstone upsert does not overwrite metadata on conflict;
            # a brand-new tombstone writes '{}'. Existing metadata is preserved.
            "metadata": existing["metadata"] if existing is not None else {},
            "lsn": lsn,
            "deleted": True,
            "created_at": now,
        }
    return int(lsn)


def recall_snapshot_for_consolidation(
    tenant_id: str, dataset: str
) -> Tuple[int, List[dict]]:
    """Embedded mirror of pgvector `recall_snapshot_for_consolidation`.

    Copies the WHOLE partition atomically under the lock and returns
    `(max_lsn, rows)` where `rows` are ascending by lsn, one per id (live +
    tombstones), `values` as `list[float]`. A write that lands after this copy has
    a higher lsn and is simply absent from `rows` (single-snapshot property).
    `max_lsn` is the highest lsn among the returned rows (0 when empty).
    """
    with _MemRecall._lock:
        part = _MemRecall._parts.get((tenant_id, dataset), {})
        rows = [
            {
                "id": rid,
                "values": [float(v) for v in row["values"]],
                "metadata": dict(row["metadata"] or {}),
                "lsn": int(row["lsn"]),
                "deleted": bool(row["deleted"]),
            }
            for rid, row in part.items()
        ]
    rows.sort(key=lambda r: r["lsn"])
    max_lsn = rows[-1]["lsn"] if rows else 0
    return max_lsn, rows


def recall_partition_count(tenant_id: str, dataset: str) -> int:
    """Embedded mirror of pgvector `recall_partition_count` (live + tombstones)."""
    with _MemRecall._lock:
        return len(_MemRecall._parts.get((tenant_id, dataset), {}))


def recall_trim(tenant_id: str, dataset: str, grace_watermark: int) -> int:
    """Embedded mirror of pgvector `recall_trim` (hard-delete lsn<=grace).

    No-op returning 0 when `grace_watermark <= 0`. Returns the number of rows
    drained. Note: the LSN counter is NOT reset on trim (mirrors the seq row),
    so subsequent allocations stay strictly above any trimmed lsn.
    """
    if grace_watermark <= 0:
        return 0
    key = (tenant_id, dataset)
    with _MemRecall._lock:
        part = _MemRecall._parts.get(key)
        if not part:
            return 0
        victims = [rid for rid, row in part.items() if row["lsn"] <= grace_watermark]
        for rid in victims:
            del part[rid]
        return len(victims)


def recall_idle_partitions(idle_seconds: float) -> List[Tuple[str, str]]:
    """Embedded mirror of pgvector `recall_idle_partitions`.

    Returns `(tenant, dataset)` partitions whose newest write (`max(created_at)`)
    is older than `idle_seconds` ago AND that still have rows. Empty / freshly-
    written partitions are excluded.
    """
    cutoff = _now() - _dt.timedelta(seconds=idle_seconds)
    out: List[Tuple[str, str]] = []
    with _MemRecall._lock:
        for key, part in _MemRecall._parts.items():
            if not part:
                continue
            newest = max(row["created_at"] for row in part.values())
            if newest <= cutoff:
                out.append(key)
    return out
