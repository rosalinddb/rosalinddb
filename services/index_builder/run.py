from __future__ import annotations

"""Index Builder service.

Consumes `DATASET_READY` messages and builds FAISS shards from the validated
records previously written to the landing area by the validator worker. The
serialised shard is written to object storage (or local path) and cataloged
in `shard_catalog`.

Incremental indexing:
  - The validator writes each upload into its own `upload-<id>/` sub-prefix.
    The *first* ingest for a dataset trains + builds a shard from all landing
    data. A *subsequent* ingest loads the current shard's FAISS index, reads
    ONLY the landing parts not already in that shard's `indexed_landing_uris`
    manifest, `index.add()`s those new vectors onto the trained index, and
    writes an updated shard. Previously-indexed uploads are never re-read.
  - A FAISS IVF index, once trained, supports `add()` without retraining; we
    rely on that. The manifest on the newest shard is the authoritative
    "what has been indexed" record, which also makes a duplicate
    `DATASET_READY` for an already-indexed batch a clean no-op.
  - Retraining policy: we ALWAYS incremental-`add()` once a shard exists.
    We do NOT auto-retrain on vector-distribution drift. If a dataset's
    distribution shifts far from the original training sample, IVF recall can
    degrade; the remedy is to delete + re-create the dataset. A future
    improvement can add a drift threshold / periodic full-rebuild. Documented
    in `docs/indexing.md`.
"""

import hashlib
import json
import os
import time
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Optional

import faiss  # type: ignore
import numpy as np

from adapters.observability import init_observability
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import build_index_span, landing_read_span, landing_write_span
from adapters.queue.queue import consume, ack, nack, publish
from adapters.queue.reaper import start_reaper_thread
from adapters.queue.shutdown import install_signal_handlers, should_stop, stop_event
from adapters.storage import shard_uri as _shard_uri
from adapters.storage.storage import write_bytes, read_bytes
from adapters.state.state import (
    migrate,
    add_shard,
    dataset_build_lock,
    get_latest_shard,
    list_shards,
    set_row_count,
    update_dataset_status,
    get_dataset,
    superseded_shards,
    delete_shards,
    list_import_jobs,
    recall_enabled,
    recall_snapshot_for_consolidation,
    recall_trim,
    recall_idle_partitions,
)
from adapters.storage.storage import delete as storage_delete, exists as storage_exists
from adapters.landing.parquet_reader import (
    id_to_int64 as _id_to_int64,
    list_landing_parts,
    read_landing_parts,
    read_shard_sidecar,
)
from adapters.metrics.metrics import counter, snapshot

# Observability bootstrap at import — see validator_worker/run.py for the
# single-process vs separate-process rationale. Idempotent.
init_observability("rosalinddb-index-builder")

DIMENSION = int(os.getenv("VECTOR_DIM", os.getenv("DIMENSION", "1536")))
# Recall touch-up: for non-tiny datasets the builder builds a FAISS **IVFFlat**
# index — an IVF coarse quantizer over **raw, uncompressed float32 vectors** —
# instead of the previous IVF+PQ. IVF+PQ's ~8x lossy Product-Quantization
# ceilinged recall@10 at ~0.65 on the SIFT benchmark even after the `nprobe`
# fix; IVFFlat ranks on *exact* L2 distances and reaches ~0.95 on the same
# data. The tradeoff is shard size — raw vectors cost ~8x more bytes than PQ
# codes — which is acceptable for an object-storage-first product: object
# storage is cheap, and RosalindDB's cost pitch is about not paying for idle
# compute, not about squeezing index bytes. See `docs/indexing.md`.
INDEX_TYPE = os.getenv("INDEX_TYPE", "ivfflat")
INDEXES_PREFIX = os.getenv("INDEXES_PREFIX", "s3://rosalinddb/indexes")
LANDING_PREFIX = os.getenv("LANDING_PREFIX", "s3://rosalinddb/landing")
TENANT_PREFIX = os.getenv("TENANT_PREFIX", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9101"))

# --- Recall→Consolidated consolidation knobs (RB_RECALL, default OFF) ------
#
# Both default to safe/current behaviour and are read live (per call) so a test
# can retune them without a module reload. They ONLY ever take effect under
# `recall_enabled()`; with the flag off the consolidate consumer never runs, the
# idle sweep is skipped, and nothing here is reached. See
# docs/architecture/recall-consolidate.md, "Scale-to-zero preservation".


def _recall_idle_seconds() -> float:
    """Idle window (s) after which a recall partition is consolidated to zero.

    `RB_RECALL_IDLE_S` (default 60): a (tenant, dataset) whose newest recall
    write is older than this is swept to a `CONSOLIDATE` by the builder's idle
    tick → drains to 0 recall rows → idle queries skip pgvector entirely
    (scale-to-zero). A missing/malformed value falls back to the default.
    """
    raw = os.getenv("RB_RECALL_IDLE_S")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return 60.0


def _truthy(value: Optional[str]) -> bool:
    """Env-flag parser. Mirror of `services.query_api.v1_query._truthy` —
    duplicated to keep this module's import graph slim (one line of parsing
    is not worth a cross-package import)."""
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _compute_shard_uri(tenant: str, dataset: str, shard_name: str, blob: bytes) -> str:
    """Pick the URI shape for a freshly-built shard.

    `RB_SHARD_VERSIONED_URIS=false` (the default) preserves the legacy
    `{INDEXES_PREFIX}/{tenant}/{dataset}/indexes/{YYYY-MM-DD}/{shard_name}`
    shape bit-identically — flipping the flag off is the rollback contract.

    `RB_SHARD_VERSIONED_URIS=true` switches to the content-addressed
    `s3://{bucket}/{tenant}/{dataset}/{shard_id}-{content_hash}.bin` shape
    defined in `docs/architecture/ssd-cache.md`. The bucket is
    derived from `INDEXES_PREFIX`'s first path segment; the resulting layout
    is intentionally flatter than the legacy one (no `/indexes/{date}/`) so
    that two builds of the same bytes converge on the same S3 key (cheap
    dedup) and two builds of different bytes can never collide.
    """
    if _truthy(os.getenv("RB_SHARD_VERSIONED_URIS")):
        # `INDEXES_PREFIX` is `s3://{bucket}[/{prefix...}]`. The new shape
        # ignores any path prefix and writes flat under the bucket — see the
        # versioned-URI rationale in `docs/architecture/ssd-cache.md`.
        bucket = INDEXES_PREFIX[len("s3://") :].split("/", 1)[0]
        shard_id = shard_name[:-len(".bin")] if shard_name.endswith(".bin") else shard_name
        return _shard_uri.build(bucket, tenant, dataset, shard_id, blob)
    return (
        f"{INDEXES_PREFIX}/{tenant}/{dataset}/indexes/"
        f"{time.strftime('%Y-%m-%d')}/{shard_name}"
    )

# Observability/test hook: the most recent `run_once` records what it did
# here so tests can assert the incremental path was taken without scraping
# OTel exporters. `build_type` is one of `full`, `incremental`, `noop` (no new
# landing parts) or `error`. NOT a metric — metrics go through `obs_metrics`.
_LAST_BUILD: dict = {
    "build_type": None,
    "vectors_added": 0,
    "parts_read": 0,
    "parts_read_uris": [],
}


# Distinct "skipped — not done" sentinel returned by `run_once` when the
# per-dataset advisory lock could not be acquired (another builder replica is
# already building this dataset). It MUST be distinguishable from a genuine
# empty no-op: an empty no-op returns 0 and the message is safely `ack`-ed (the
# work is done); a SKIP means the build did NOT run, so the consume loop must
# `nack(msg, requeue=True)` to redeliver it — the skipped message may carry a
# newer upload than the in-progress build. A negative value is used so the
# common "N vectors added" / "0 = no-op" integer contract is untouched for
# every existing caller that only cares about the count.
BUILD_SKIPPED = -1


# IVF training floor. An IVFFlat index has a SINGLE training step — k-means
# clustering of the `nlist` coarse-quantizer centroids. FAISS k-means needs at
# least as many training points as centroids; we additionally require a sane
# minimum batch (`>= IVF_TRAINING_FLOOR` rows and `nlist >= 4`) below which IVF
# cell partitioning buys nothing and the build falls back to an exact flat
# index. `IVF_TRAINING_FLOOR` is small (64) because IVFFlat trains no PQ
# codebook.
#
# Unlike the previous IVF+PQ builder there is NO second, larger PQ-codebook
# training floor (`2^PQ_NBITS`, 256 for 8-bit codes): IVFFlat stores raw
# vectors and trains no codebook, so the index-type gate only needs IVF's
# floor. The old `_pq_training_floor()` / `_choose_pq_m()` helpers and the
# `PQ_NBITS`/`PQ_M` env knobs are therefore gone.
IVF_TRAINING_FLOOR = max(4, int(os.getenv("IVF_TRAINING_FLOOR", "64")))


def _choose_nlist(n_vectors: int) -> int:
    """Choose the IVFFlat coarse-quantizer cell count for `n_vectors`.

    The FAISS rule of thumb for an IVF index is `nlist ≈ sqrt(N)` to
    `4*sqrt(N)` — large enough for cells to be selective, small enough that
    each cell still holds a meaningful posting. OpenData Vector's flat-IVF
    sizing (RFC-0005 / their bench config) targets roughly **~100 vectors per
    cluster** ("it is optimal to maintain one centroid for ~100 vectors"); for
    SIFT-scale data `4*sqrt(N)` lands close to that (100k vectors → ~1264 cells
    → ~80 per cell). We borrow that target here.

    The result is clamped so k-means can always train (`nlist <= N`, and at
    least `N//8` of headroom is left so every cell is non-degenerate) and
    capped by the optional `IVF_NLIST` env override (a hard ceiling, kept for
    backwards compatibility). Returns `>= 1`.
    """
    import math

    rule_of_thumb = int(4 * math.sqrt(max(1, n_vectors)))
    ceiling = int(os.getenv("IVF_NLIST", "4096"))
    # never more cells than 1/8 of the points — keeps every posting non-empty
    # and leaves k-means enough training points per centroid.
    nlist = min(rule_of_thumb, ceiling, max(1, n_vectors // 8))
    return max(1, nlist)


def _landing_prefix(dataset: str, tenant: str) -> str:
    """Compute landing prefix respecting tenancy setting.

    Mirrors `validator_worker._landing_prefix`. Kept duplicated rather than
    imported to avoid coupling the two services through a private helper.
    """
    base = LANDING_PREFIX
    if not base.endswith("/"):
        base += "/"
    if TENANT_PREFIX:
        return f"{base}{tenant}/{dataset}"
    return f"{base}{dataset}"


# `_id_to_int64` is the shared SHA1->int64 hash, now sourced from
# `adapters.landing.parquet_reader` (imported above) so the consolidated-tier CRUD
# surface in `source_registry` hashes ids to the EXACT same int64 the builder
# stamps onto FAISS vectors. Re-exported under the private name so the build
# path's many call sites stay unchanged.


def build_ivfflat(vectors: np.ndarray, ids: np.ndarray | None = None) -> bytes:
    """Build and serialize a FAISS IVFFlat index for the given vectors.

    IVFFlat is an IVF index with an `IndexFlatL2` coarse quantizer and **raw,
    uncompressed float32 vectors** stored in each cell — no Product
    Quantization. It still partitions the space into `nlist` cells, so the
    query-time `nprobe` knob works exactly as it did for IVF+PQ; but because it
    ranks candidates on *exact* L2 distances rather than PQ-approximate ones it
    reaches far higher recall (~0.95 on SIFT vs ~0.65 for IVF+PQ). The tradeoff
    is shard size: raw vectors cost ~8x more bytes than 8-bit PQ codes —
    acceptable for an object-storage-first product. See `docs/indexing.md`.

    `nlist` is sized by `_choose_nlist` (FAISS `4*sqrt(N)` rule of thumb,
    informed by OpenData Vector's ~100-vectors-per-cluster target).

    If `ids` is provided it must be int64 and the same length as `vectors`;
    the result is wrapped in an `IndexIDMap2` so search yields the original
    ids rather than the internal contiguous offsets.
    """
    dim = int(vectors.shape[1])
    nlist = _choose_nlist(int(vectors.shape[0]))
    quantizer = faiss.IndexFlatL2(dim)
    inner = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    inner.train(vectors)

    if ids is not None:
        index = faiss.IndexIDMap2(inner)
        index.add_with_ids(vectors, ids)
    else:
        inner.add(vectors)
        index = inner

    blob = faiss.serialize_index(index)
    # faiss may return a numpy array of uint8; normalize to bytes
    if isinstance(blob, np.ndarray):
        return blob.tobytes()
    return blob


def build_flat(vectors: np.ndarray, ids: np.ndarray | None = None) -> bytes:
    """Build a simple `IndexFlatL2` for tiny datasets where IVF training fails.

    Used as a fallback path when there are too few rows to train IVFFlat
    sensibly. The shape of the API is identical to `build_ivfflat` so the
    caller can swap them based on row count.
    """
    dim = int(vectors.shape[1])
    inner = faiss.IndexFlatL2(dim)
    if ids is not None:
        index = faiss.IndexIDMap2(inner)
        index.add_with_ids(vectors, ids)
    else:
        inner.add(vectors)
        index = inner
    blob = faiss.serialize_index(index)
    if isinstance(blob, np.ndarray):
        return blob.tobytes()
    return blob


def _build_sidecar(ids: list[str], metas: list[dict]) -> bytes:
    """Build the shard sidecar: maps each int64 FAISS id back to its origin.

    FAISS `IndexIDMap2` only stores the SHA1-derived int64 hash of each
    string id (see `_id_to_int64`). A search therefore returns int64s that
    cannot be inverted back to the customer's original string id. To bridge
    that gap we persist a JSON sidecar next to every shard:

        { "<int64_hash_as_string>": {"id": "<original id>", "metadata": {…}} }

    The query path / ephemeral runner load this file and map each hit back
    to `{id, score, metadata}`. Keyed by the *string* form of the int64 so
    the JSON object keys are valid and stable. If two ids collide on the
    same hash (vanishingly unlikely at MVP scale) the last writer wins —
    consistent with FAISS's own IDMap behaviour.
    """
    return json.dumps(_sidecar_dict(ids, metas)).encode("utf-8")


def _sidecar_dict(ids: list[str], metas: list[dict]) -> dict[str, dict]:
    """Build the in-memory sidecar mapping (see `_build_sidecar`).

    Returned as a plain dict so the incremental path can merge a new batch's
    entries into the existing shard's sidecar before re-serialising.
    """
    mapping: dict[str, dict] = {}
    for raw_id, meta in zip(ids, metas):
        key = str(_id_to_int64(raw_id))
        mapping[key] = {"id": raw_id, "metadata": meta if isinstance(meta, dict) else {}}
    return mapping


def _serialize_index(index) -> bytes:
    """Serialize a FAISS index to bytes (normalising the numpy-array case)."""
    blob = faiss.serialize_index(index)
    if isinstance(blob, np.ndarray):
        return blob.tobytes()
    return blob


def _dedup_batch_last_wins(
    ids: list[str], vectors: np.ndarray, metas: list[dict]
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Dedup one ingest batch by id, keeping the LAST occurrence of each id.

    The vectors endpoint is an upsert: a batch that re-sends an id (whether
    twice within itself or across batches) must end as a single row with the
    last value. This collapses within-batch duplicates so a single ingest of
    the same id twice yields exactly one row carrying its final value/metadata.

    Dedup keys on the `_id_to_int64` SHA1 hash — the same key FAISS works in —
    so a hash collision between two distinct string ids would let one evict the
    other. That is a known, accepted limitation at MVP scale and out of scope.
    """
    seen: dict[int, int] = {}
    for pos, raw_id in enumerate(ids):
        seen[_id_to_int64(raw_id)] = pos  # later positions overwrite earlier
    if len(seen) == len(ids):
        return ids, vectors, metas  # no duplicates — common case, no copy
    keep = sorted(seen.values())
    deduped_ids = [ids[i] for i in keep]
    deduped_metas = [metas[i] for i in keep]
    deduped_vectors = vectors[keep]
    return deduped_ids, deduped_vectors, deduped_metas


def _add_to_index(index, vectors: np.ndarray, int_ids: np.ndarray) -> None:
    """Append `vectors`/`int_ids` to an already-built (loaded) FAISS index.

    The index loaded from a prior shard is an `IndexIDMap2` wrapping either a
    trained `IndexIVFFlat` (or, for legacy shards built before the recall
    touch-up, an `IndexIVFPQ`) or a flat index — all support `add_with_ids`
    without retraining (IVF's quantizer/centroids are fixed once trained).
    Raises `ValueError` on a vector-dimension mismatch so the caller can flip
    the dataset to `error` rather than corrupt the shard.
    """
    if int(vectors.shape[1]) != int(index.d):
        raise ValueError(
            f"dimension mismatch: new batch is {vectors.shape[1]}-dim, "
            f"existing index is {index.d}-dim"
        )
    index.add_with_ids(vectors, int_ids)


def _remove_ids(index, int_ids: list[int]) -> None:
    """Remove the given int64 FAISS ids from a loaded shard index (upsert).

    Called on the incremental path ONLY when an incoming batch's ids overlap
    ids already in the shard, so the stale copies are dropped before the new
    ones are `add_with_ids`'d (last-write-wins upsert). The caller gates this
    on a non-empty overlap because `remove_ids` is an O(N) scan of the whole
    index — it must not run in the common append-only case.

    Uses `faiss.IDSelectorBatch` (hashed O(1) membership) rather than
    `IDSelectorArray` (linear scan). `IndexIVF.remove_ids` needs the IVF
    direct map; if it is not initialised FAISS raises, so we initialise it via
    `make_direct_map()` on the inner IVF index before removing.
    """
    selector = faiss.IDSelectorBatch(np.asarray(int_ids, dtype=np.int64))
    try:
        index.remove_ids(selector)
    except RuntimeError as remove_err:
        # IVF shards need an initialised direct map for id-based removal.
        # If `extract_index_ivf` itself fails (e.g. a flat index has no IVF),
        # surface the ORIGINAL `remove_ids` error — not the inner failure —
        # so the caller sees the real reason the removal could not proceed.
        try:
            faiss.extract_index_ivf(index).make_direct_map()
        except Exception as inner_err:  # noqa: BLE001
            raise remove_err from inner_err
        index.remove_ids(selector)


def _reconstruct_surviving(index, survivor_int_ids: list[int]) -> np.ndarray:
    """Reconstruct the raw vectors for `survivor_int_ids` from a loaded shard.

    Used by the incremental consolidation's UNION-REBUILD path (see
    `_build_consolidated_shard`). For an `IndexIDMap2(IndexIVFFlat)` we cannot
    `remove_ids` the stale copies of re-upserted ids — that trips a FAISS 1.8.0
    C++ assertion (`j == index->ntotal`, IndexIDMap.cpp:181) that `abort()`s the
    whole process. Instead we read the *surviving* vectors back out and rebuild
    a fresh shard from scratch (which never removes).

    IVFFlat stores **raw, uncompressed float32** vectors, so reconstruction is
    LOSSLESS — the rebuilt union carries the exact original survivor vectors.
    `make_direct_map()` initialises the per-id offset map an IVF needs for
    `reconstruct`; we reconstruct PER-ID (not `reconstruct_n`) because the id_map
    is sparse/unordered and `reconstruct_n` indexes by contiguous offset.
    Returns an `(len(survivor_int_ids), d)` float32 array (empty when there are
    no survivors).
    """
    if not survivor_int_ids:
        return np.empty((0, int(index.d)), dtype=np.float32)
    # IVF needs a direct map for per-id reconstruct; flat indexes ignore this.
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        ivf.make_direct_map()
    out = np.empty((len(survivor_int_ids), int(index.d)), dtype=np.float32)
    for row, sid in enumerate(survivor_int_ids):
        out[row] = index.reconstruct(int(sid))
    return out


def _is_ivf_index(index) -> bool:
    """True iff `index` wraps an IVF (so `remove_ids` can trip the FAISS abort).

    `remove_ids` is safe on a *flat* `IndexIDMap2(IndexFlatL2)` shard — only the
    IVF removal trips the FAISS 1.8.0 `j == index->ntotal` C++ assertion. We
    detect IVF via `try_extract_index_ivf` (returns `None` for a flat index)
    rather than the catalog's `index_type` string so the decision is grounded in
    the actual loaded index, not a possibly-stale sidecar/catalog label.
    """
    return faiss.try_extract_index_ivf(index) is not None


def _union_rebuild_blob(
    index,
    existing_sidecar: dict,
    drop_int_ids: set[int],
    new_ids: list[str],
    new_vectors: np.ndarray,
    new_metas: list[dict],
) -> tuple[bytes, bytes, str, int]:
    """Rebuild a shard as a UNION instead of `remove_ids`-ing the dropped ids.

    Shared by every path that must drop ids overlapping an existing IVF shard —
    the consolidation fold (#18, `_build_consolidated_shard`), the landing-ingest
    incremental upsert, and the single-id delete (#28). All three would otherwise
    `remove_ids` the stale copies, which on an `IndexIDMap2(IVFFlat)` trips a
    FAISS 1.8.0 C++ assertion (`j == index->ntotal`, IndexIDMap.cpp:181) that
    `abort()`s the whole builder process — NOT a catchable Python error. So we
    REBUILD a fresh shard from scratch (which never removes):

        surviving vectors (existing ids NOT in `drop_int_ids`)
        ∪ the new/updated vectors (`new_ids`/`new_vectors`/`new_metas`)

    IVFFlat survivor reconstruction is LOSSLESS (raw float32), and the rebuild
    re-trains the IVF quantizer over the union, keeping recall sound as the
    dataset evolves. For a delete `new_*` are empty (survivors only); for an
    upsert they carry the incoming batch.

    CRITICAL (review P1, carried from #18): the rebuilt index must carry each
    survivor's ORIGINAL int64 — the exact hash its vector was reconstructed under
    — NOT a re-hash of a synthesised string id. For a survivor present in the
    sidecar `_id_to_int64(entry["id"]) == sid` round-trips, but a survivor MISSING
    from a partial/unreadable sidecar (`read_shard_sidecar` degrades to `{}`) is
    given a synthesised `str(sid)` id; re-hashing that gives
    `_id_to_int64(str(sid)) != sid`, stamping the vector under a WRONG int64 —
    unreachable by its true id and un-removable by a future tombstone. So we
    CONCATENATE the actual survivor int64s with the new ids' int64s rather than
    re-hashing the union string ids, and key the sidecar by those same int64s.

    Returns `(index_blob, sidecar_blob, index_type_str, total_vectors)`.
    """
    existing_int_ids = set(faiss.vector_to_array(index.id_map).tolist())
    survivor_int_ids = [
        int(i) for i in existing_int_ids if int(i) not in drop_int_ids
    ]
    survivor_vectors = _reconstruct_surviving(index, survivor_int_ids)

    # Map each survivor int64 back to its (string id, metadata) via the existing
    # sidecar — for the SIDECAR only. A survivor missing from the sidecar still
    # keeps its vector; we synthesise a stable string id from the int64 so it has
    # a sidecar entry and is never silently dropped.
    survivor_ids: list[str] = []
    survivor_metas: list[dict] = []
    for sid in survivor_int_ids:
        entry = existing_sidecar.get(str(sid))
        if entry is not None:
            survivor_ids.append(entry.get("id", str(sid)))
            survivor_metas.append(entry.get("metadata", {}) or {})
        else:
            survivor_ids.append(str(sid))
            survivor_metas.append({})

    has_new = len(new_ids) > 0 and new_vectors.size > 0
    union_ids = survivor_ids + list(new_ids)
    union_metas = survivor_metas + list(new_metas)
    if survivor_vectors.shape[0] and has_new:
        union_vectors = np.concatenate([survivor_vectors, new_vectors], axis=0)
    elif has_new:
        union_vectors = new_vectors
    else:
        union_vectors = survivor_vectors
    # Carry survivors' ORIGINAL int64s through unchanged (see CRITICAL note
    # above); only the new rows are hashed from their string ids. Order matches
    # `union_vectors`/`union_ids`/`union_metas` (survivors first).
    union_int_ids = np.array(
        survivor_int_ids + [_id_to_int64(i) for i in new_ids],
        dtype=np.int64,
    )

    # Rebuild via the from-scratch builder selection (IVFFlat when the union is
    # large enough to train, else flat) — never `remove_ids`.
    nlist_target = _choose_nlist(int(union_vectors.shape[0]))
    if (
        INDEX_TYPE == "ivfflat"
        and nlist_target >= 4
        and union_vectors.shape[0] >= IVF_TRAINING_FLOOR
    ):
        blob = build_ivfflat(union_vectors, union_int_ids)
        index_type_str = "ivfflat"
    else:
        blob = build_flat(union_vectors, union_int_ids)
        index_type_str = "flat"
    # Keep index↔sidecar consistent: stamp the sidecar under the SAME int64s
    # carried into the index (`union_int_ids`), not a re-hash of `union_ids`,
    # which would re-derive the WRONG key for a missing-from-sidecar survivor.
    sidecar_blob = json.dumps(
        {
            str(int(int64_id)): {
                "id": raw_id,
                "metadata": meta if isinstance(meta, dict) else {},
            }
            for int64_id, raw_id, meta in zip(
                union_int_ids.tolist(), union_ids, union_metas
            )
        }
    ).encode("utf-8")
    return blob, sidecar_blob, index_type_str, int(union_vectors.shape[0])


# Number of newest shards to retain when sweeping superseded ones. The newest
# shard is the one queries load; the one before it is a grace buffer for an
# in-flight query that resolved it as the latest shard moments before this
# build wrote a newer one and is still faulting it into its local cache.
_SHARDS_TO_KEEP = max(2, int(os.getenv("SHARD_KEEP", "2")))


def _sweep_superseded_shards(tenant: str, dataset: str) -> int:
    """Delete shards older than the newest `_SHARDS_TO_KEEP` for the dataset.

    Called after a successful build. For each superseded shard it removes the
    shard `.bin`, its `.meta.json` sidecar, and the `shard_catalog` row. The
    catalog row is deleted *after* the objects so a crash mid-sweep leaves an
    orphan object (harmless, retried next build) rather than a catalog row
    pointing at a missing object.

    Race-safety: queries only ever load `get_latest_shard`. Retaining the
    newest shard plus one previous means a query that resolved the (now
    second-newest) shard just before this build wrote the newest one can still
    read its `.bin`/`.meta.json`; only strictly older shards — which no query
    started after they were superseded could target — are removed.

    Best-effort: a storage error on one object is logged and the sweep moves
    on; the next build retries. Returns the number of catalog rows removed.
    """
    stale = superseded_shards(tenant, dataset, keep=_SHARDS_TO_KEEP)
    if not stale:
        return 0
    swept_objects = 0
    for shard in stale:
        for uri in (shard["shard_uri"], f"{shard['shard_uri']}.meta.json"):
            try:
                storage_delete(uri)
                swept_objects += 1
            except Exception as exc:  # noqa: BLE001
                print(f"builder: sweep could not delete {uri}: {exc}")
            # Drop the swept shard from the query path's in-memory index cache so
        # a stale (now-deleted) FAISS index is never served. This matters when
        # the builder and query API share a process (a single-process dev/test
        # harness); when they are separate processes the query path's own
        # newest-shard-per-query resolution naturally misses a swept shard.
        _evict_query_cache(shard.get("id"))
    removed = delete_shards(tenant, dataset, [s["id"] for s in stale])
    obs_metrics.record_storage_swept(swept_objects, "shard")
    return removed


def _evict_query_cache(shard_id) -> None:
    """Best-effort: drop a swept shard from the query API's in-memory cache.

    Imported lazily so the builder does not hard-depend on the query service,
    and wrapped so an import/eviction failure can never fail a build.
    """
    if shard_id is None:
        return
    try:
        from services.query_api.v1_query import evict_shard

        evict_shard(shard_id)
    except Exception as exc:  # noqa: BLE001
        print(f"builder: could not evict shard {shard_id} from query cache: {exc}")


def _sweep_indexed_landing(tenant: str, dataset: str) -> int:
    """Prune landing/staging objects this dataset has safely captured.

    Run after a successful build. Two classes of object are reclaimed:

      1. **Indexed landing parts.** Every `.parquet` URI in the *newest*
         shard's `indexed_landing_uris` manifest is already folded into the
         queryable shard. Deleting it is safe: the manifest still records the
         URI (so a duplicate `DATASET_READY` is still a no-op — the part just
         no longer appears in the landing listing, which is exactly what
         `run_once` wants), and the builder is single-replica so no concurrent
         build is mid-read of the same part. Queries never read landing.

      2. **Terminal-import staged uploads.** A bulk-import job's *raw* client
         upload sits in the `staging/` root. Once the job is terminal
         (`completed`/`failed`) the raw upload has served its purpose — the
         validated landing part (point 1 above) or a `failed` status is the
         record of record — so the staged `upload.*` object is deleted.

    The `rejected.jsonl` sidecar is deliberately **retained**: customers
    download it via a presigned URL after the job finishes. It is left in place
    for the documented retention window (see `docs/api/imports.md`); a
    time-based prune of old rejected sidecars is a separate scheduled job.

    Best-effort: a storage error on one object is logged and the sweep
    continues. Returns the number of objects deleted.
    """
    swept = 0

    latest = get_latest_shard(tenant, dataset)
    indexed_parts = list((latest or {}).get("indexed_landing_uris", []) or [])
    for uri in indexed_parts:
        try:
            if storage_exists(uri):
                storage_delete(uri)
                swept += 1
        except Exception as exc:  # noqa: BLE001
            print(f"builder: landing sweep could not delete {uri}: {exc}")

    try:
        jobs = list_import_jobs(tenant, dataset)
    except Exception as exc:  # noqa: BLE001
        print(f"builder: landing sweep could not list import jobs: {exc}")
        jobs = []
    for job in jobs:
        if job.get("status") not in ("completed", "failed"):
            continue
        upload_uri = job.get("upload_uri")
        if not upload_uri:
            continue
        try:
            if storage_exists(upload_uri):
                storage_delete(upload_uri)
                swept += 1
        except Exception as exc:  # noqa: BLE001
            print(f"builder: landing sweep could not delete staged "
                  f"upload {upload_uri}: {exc}")

    obs_metrics.record_storage_swept(swept, "landing")
    return swept


def run_once(dataset: str, tenant: str) -> int:
    """Index the dataset's *new* landing parquet, append to the shard, catalog it.

    First ingest (no shard yet): train + build a fresh shard from all landing
    data.

    Subsequent ingest (a shard already exists): load the current shard's FAISS
    index + sidecar, read ONLY the landing parts not in that shard's
    `indexed_landing_uris` manifest, `index.add()` the new vectors, and write
    an updated shard. Previously-indexed uploads are never re-read.

    Returns the number of vectors added by *this* build: `0` if there is
    nothing new (an empty landing area, or a duplicate `DATASET_READY` whose
    batch is already indexed — a genuine no-op, the work is DONE), or the
    `BUILD_SKIPPED` sentinel (a negative value) if the per-dataset lock could
    not be acquired and the build did NOT run. The caller MUST distinguish
    these: a `0` no-op is safe to `ack`; a `BUILD_SKIPPED` must be redelivered.
    Any failure sets `status='error'` with `error_message` and returns 0.

    Multi-worker safety: with `index_builder` replicated, two builder replicas
    can pick up two `DATASET_READY` messages for the SAME dataset at once (or
    a redelivered message races the original) and both fold in the same landing
    parts — double-indexing the vectors. This wraps the build in a per-dataset
    Postgres advisory lock. The lock is NON-blocking (`pg_try_advisory_lock`):
    if another builder already holds it for this dataset we SKIP the build and
    return `BUILD_SKIPPED` immediately. Skipping is only safe if the message is
    RE-DELIVERED — the skipped message may represent a newer upload than the
    in-progress build (the winning builder's landing scan ran before these parts
    landed), so dropping it would lose those vectors. `_handle_dataset_ready` /
    `main_loop` therefore `nack(msg, requeue=True)` on `BUILD_SKIPPED`; the
    retry re-indexes any still-unindexed parts, or is a clean no-op via the
    newest shard's `indexed_landing_uris` manifest if the winning build already
    covered them. A non-blocking try-lock is preferred over a blocking lock so
    a builder thread is never parked waiting on a dataset it should just hand
    back to the queue; builds of *different* datasets get distinct locks and
    still run fully in parallel. In `memory://` / single-process test mode
    there is no concurrency to guard, so the lock is a pure no-op (always
    acquired).
    """
    with dataset_build_lock(tenant, dataset) as acquired:
        if not acquired:
            # Another builder replica is already building this dataset. The
            # build did NOT run — return BUILD_SKIPPED so the consume loop
            # nacks the message for redelivery instead of acking it away. The
            # skipped message may carry a newer upload than the in-progress
            # build, so it must be retried, not discarded.
            print(
                f"builder: dataset {tenant}/{dataset} is already being built "
                f"by another replica — skipping (message will be redelivered)"
            )
            _LAST_BUILD.update(
                build_type="skipped", vectors_added=0, parts_read=0, parts_read_uris=[]
            )
            return BUILD_SKIPPED
        return _run_once_locked(dataset, tenant)


def _write_shard(
    tenant: str,
    dataset: str,
    blob: bytes,
    sidecar_blob: bytes,
    total_vectors: int,
    index_type_str: str,
    *,
    build_type: str,
    indexed_uris: List[str],
    consolidated_lsn: int = 0,
) -> str:
    """Write a freshly-built shard + sidecar, catalog it, reconcile, prewarm.

    The shared tail of every build path: the FAISS `blob` and its JSON
    `sidecar_blob` are written to object storage under a collision-proof URI,
    a `shard_catalog` row is added (which supersedes the previous newest
    shard), `dataset.row_count` is reconciled to `total_vectors`, and an
    opt-in `PREWARM_SHARD` hint is published. Returns the shard URI.

    Extracted from `_run_once_locked` so the `DELETE_VECTORS` path
    (`run_delete_once`) and the `CONSOLIDATE` path (`run_consolidate_once`)
    reuse the EXACT same write/catalog/prewarm behaviour instead of duplicating
    it — a delete or a consolidation is just another way to produce the next
    shard generation. The caller owns the per-dataset advisory lock and the
    status flip + sweep (which differ slightly between build/delete/consolidate).

    `consolidated_lsn` (default 0) is the recall-tier watermark stamped on the
    `shard_catalog` row — the highest recall LSN folded into ANY shard of this
    dataset so far (a per-dataset high-water mark, not a per-build value). The
    consolidation path advances it to the snapshot's `max(lsn)`; every other
    build (ingest/incremental/delete) must CARRY FORWARD the prior newest shard's
    value (`latest_shard.consolidated_lsn`) so the watermark stays monotonic — a
    non-consolidate fold only touches recall-owned rows (`lsn > watermark`), so
    it neither advances nor may regress the watermark. The default 0 applies only
    to a dataset's very first shard (no consolidated predecessor) and, with the
    flag off, to every shard (a flag-off deploy never consolidates).
    """
    checksum = hashlib.sha256(blob).hexdigest()
    # Shard filename must be collision-proof: two builds completing in the
    # same millisecond would otherwise produce the same name and the second
    # `write_bytes` would silently overwrite the first shard's `.bin`. A short
    # uuid suffix makes the name unique. The filename is NOT load-bearing for
    # ordering — `shard_catalog` orders strictly by `created_at` (Postgres) /
    # insertion `id` (memory mode), and the `.meta.json` sidecar name is
    # derived from this (now-unique) shard name, so a random suffix is safe.
    shard_name = f"shard-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.bin"
    # `_compute_shard_uri` picks the legacy vs. versioned shape based on
    # `RB_SHARD_VERSIONED_URIS`. Default (off) is bit-identical to the inline
    # construction this replaces.
    shard_uri = _compute_shard_uri(tenant, dataset, shard_name, blob)
    with landing_write_span(uri=shard_uri):
        write_bytes(shard_uri, blob)
        # Persist the id/metadata sidecar alongside the shard so the query
        # path can invert FAISS's int64 hashes back to the customer's original
        # string ids. For an incremental build this is the merged map (old
        # shard's sidecar + new batch); for a delete it is the old sidecar
        # minus the deleted id.
        write_bytes(f"{shard_uri}.meta.json", sidecar_blob)
    add_shard(
        tenant,
        dataset,
        shard_uri,
        checksum,
        total_vectors,
        index_type_str,
        build_type=build_type,
        indexed_landing_uris=indexed_uris,
        consolidated_lsn=consolidated_lsn,
    )
    # Reconcile dataset.row_count to the just-built shard's true unique-vector
    # count. The validator's `increment_row_count` runs per-batch without
    # knowing which ids already exist, so a batch that upserts existing ids
    # over-counts (re-ingesting `id="x"` would otherwise double `row_count`
    # every retry). `total_vectors == index.ntotal` AFTER the incremental
    # path's `remove_ids` + `add_with_ids` (or a delete's `remove_ids`), which
    # is the authoritative count of unique live ids in the dataset (one shard
    # per dataset is the steady-state invariant — the sweep retains the newest
    # shard plus one grace-buffer second-newest; older shards are purged).
    # `set_row_count` is idempotent so a builder retry that re-commits the same
    # shard leaves `row_count` unchanged, and self-heals any pre-existing drift.
    try:
        set_row_count(tenant, dataset, total_vectors)
    except Exception as exc:  # noqa: BLE001
        # Best-effort: a row_count reconcile failure must not fail the build —
        # the shard is durable and queryable. The next successful build
        # re-runs the reconcile.
        print(f"builder: set_row_count failed for {tenant}/{dataset}: {exc}")
    # Opt-in prewarm hint. The catalog row is now durable; publish a
    # PREWARM_SHARD message so a DP with the consumer enabled
    # (`RB_PREWARM_CONSUMER=true`) can speculatively admit the shard before the
    # first query lands. Gated on `RB_PREWARM_ON_BUILD=true` so the rollback
    # contract holds: an unset env preserves current behaviour. The publish
    # runs AFTER `add_shard` so a queue-side failure cannot leave a PREWARM
    # message pointing at an uncataloged shard, and is wrapped in best-effort
    # error handling so a queue blip cannot fail an otherwise-successful build.
    if os.getenv("RB_PREWARM_ON_BUILD", "false").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        try:
            publish(
                "PREWARM_SHARD",
                {"tenant": tenant, "dataset": dataset, "shard_uri": shard_uri},
            )
        except Exception as exc:  # noqa: BLE001
            # A queue failure on the prewarm hint must NOT fail the build — the
            # shard is on object storage and in the catalog. The first query
            # covers the rendezvous-elected DP via a normal cache miss.
            print(
                "builder: PREWARM_SHARD publish failed for "
                f"{tenant}/{dataset} ({shard_uri}): {exc}"
            )
    return shard_uri


def _run_once_locked(dataset: str, tenant: str) -> int:
    """Run the actual build — caller holds the per-dataset advisory lock.

    See `run_once` for the build semantics; this is the body that runs once
    the per-dataset lock is held (or unconditionally, in `memory://` mode).
    """
    _LAST_BUILD.update(build_type=None, vectors_added=0, parts_read=0, parts_read_uris=[])
    # `build_index` span — child of the upload trace via queue propagation.
    with build_index_span(tenant=tenant, dataset=dataset):
        landing_prefix = _landing_prefix(dataset, tenant)

        # The newest shard is authoritative: its `indexed_landing_uris`
        # manifest records every landing part already folded in. Reading it
        # *before* the landing scan makes a duplicate DATASET_READY a no-op.
        latest_shard = get_latest_shard(tenant, dataset)
        already_indexed = set((latest_shard or {}).get("indexed_landing_uris", []) or [])

        try:
            all_parts = list_landing_parts(landing_prefix)
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(tenant, dataset, "error", error_message=f"landing list: {exc}")
            _LAST_BUILD["build_type"] = "error"
            return 0

        new_parts = [p for p in all_parts if p not in already_indexed]
        if not new_parts:
            # Either landing is genuinely empty (validator fired on an
            # empty/failed upload) or every part is already indexed (a
            # duplicate DATASET_READY). Both are clean no-ops — skipping is
            # correct and the dataset status is left untouched.
            _LAST_BUILD["build_type"] = "noop"
            return 0

        is_incremental = latest_shard is not None

        try:
            with landing_read_span(uri=landing_prefix):
                ids, vectors, metas = read_landing_parts(new_parts)
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(tenant, dataset, "error", error_message=f"landing read: {exc}")
            _LAST_BUILD["build_type"] = "error"
            return 0

        if not ids or vectors.size == 0:
            # New parts existed but held no rows — treat as a clean no-op.
            _LAST_BUILD["build_type"] = "noop"
            return 0

        # Within-batch dedup (upsert, last-write-wins). Applies to BOTH the
        # full first-ingest and the incremental path: a single ingest carrying
        # the same id twice must end as one row holding the last value. Dedup
        # keys on the `_id_to_int64` hash, so a hash collision between two
        # distinct string ids would let one evict the other — known, out of
        # scope at MVP scale.
        ids, vectors, metas = _dedup_batch_last_wins(ids, vectors, metas)

        build_start = time.time()
        try:
            int_ids = np.array([_id_to_int64(i) for i in ids], dtype=np.int64)

            if is_incremental:
                # --- incremental: load the existing shard, add() the new batch
                build_type = "incremental"
                index = faiss.deserialize_index(
                    np.frombuffer(read_bytes(latest_shard["shard_uri"]), dtype=np.uint8)
                )
                # Cross-batch upsert: if any incoming id already exists in the
                # shard, remove the stale copies before add() so a re-sent id
                # is replaced, not duplicated. Overlap-gated: `remove_ids` is an
                # O(N) scan of the whole index, so it is skipped entirely in the
                # common append-only case where no incoming id overlaps.
                #
                # The set of ids already in the shard is derived from the FAISS
                # index itself — `IndexIDMap2.id_map` is the authoritative int64
                # id store — NOT from the metadata sidecar. `read_shard_sidecar`
                # swallows all errors and returns `{}` on any transient read
                # failure; gating on it would then make `overlap` empty and let
                # `_add_to_index` silently APPEND a duplicate. The FAISS index is
                # always present here (we just deserialized it), so it is the
                # correct authority for the upsert gate.
                if not hasattr(index, "id_map"):
                    # Every shard type on the incremental path (IVFFlat, tiny
                    # flat, legacy IVF+PQ) is IDMap2-wrapped per `_add_to_index`.
                    # If one is not, fail loudly rather than degrade to append
                    # and silently reintroduce duplicates.
                    raise RuntimeError(
                        "incremental upsert: loaded shard index has no `id_map` "
                        f"(type {type(index).__name__}); cannot compute the "
                        "overlap gate without an authoritative id store"
                    )
                existing_int_ids = set(
                    faiss.vector_to_array(index.id_map).tolist()
                )
                overlap = [int(i) for i in int_ids if int(i) in existing_int_ids]
                # Still read the sidecar for the metadata merge / union rebuild
                # below — it is NOT used for the overlap gate above.
                existing_sidecar = read_shard_sidecar(latest_shard["shard_uri"])
                if overlap and _is_ivf_index(index):
                    # UNION-REBUILD PATH (#28, same fix as #18's consolidation
                    # fold). The incoming batch re-upserts ids already in an IVF
                    # shard, so we would otherwise `_remove_ids(overlap)` the
                    # stale copies before re-adding. On an `IndexIDMap2(IVFFlat)`
                    # that trips a FAISS 1.8.0 C++ assertion (`j ==
                    # index->ntotal`, IndexIDMap.cpp:181) that `abort()`s the
                    # whole builder process — NOT a catchable Python error. So
                    # instead of remove+add we REBUILD the shard from scratch as
                    # a UNION: surviving cold vectors (existing ids the batch does
                    # NOT re-upsert) ∪ the incoming batch (carrying its NEW
                    # values), preserving every survivor's ORIGINAL int64. The
                    # re-upserted ids appear ONLY in the new set, so they win
                    # last-write and are never duplicated. (See
                    # `_union_rebuild_blob` for the lossless-reconstruct + P1
                    # original-int64 details.)
                    blob, sidecar_blob, index_type_str, total_vectors = (
                        _union_rebuild_blob(
                            index,
                            existing_sidecar,
                            drop_int_ids=set(overlap),
                            new_ids=ids,
                            new_vectors=vectors,
                            new_metas=metas,
                        )
                    )
                else:
                    # Cheap path: no overlap (append-only) OR a flat shard where
                    # `remove_ids` is safe (only the IVF removal trips the
                    # abort). Remove the stale copies — overlap-gated so this
                    # O(N) scan is skipped in the common append-only case — then
                    # `add()` the new batch onto the loaded index, no rebuild.
                    if overlap:
                        _remove_ids(index, overlap)
                    _add_to_index(index, vectors, int_ids)
                    index_type_str = latest_shard.get("index_type", "flat")
                    blob = _serialize_index(index)
                    # Merge the existing sidecar with the new batch's entries so
                    # the shard's id/metadata map covers batch1 + batch2. A dict
                    # update overwrites by key, so an upserted id's metadata
                    # reflects the NEW (within-batch-deduped) value.
                    merged_sidecar = dict(existing_sidecar)
                    merged_sidecar.update(_sidecar_dict(ids, metas))
                    sidecar_blob = json.dumps(merged_sidecar).encode("utf-8")
                    total_vectors = int(getattr(index, "ntotal", 0))
                indexed_uris = sorted(already_indexed | set(new_parts))
            else:
                # --- first ingest: full train + build, exactly as before
                build_type = "full"
                # Index-type gate. IVFFlat has a SINGLE training step —
                # k-means on the `nlist` coarse-quantizer centroids — so
                # the gate only needs IVF's training floor:
                # `>= IVF_TRAINING_FLOOR` (64) rows and `nlist >= 4`. Tiny
                # datasets fall back to an exact flat index; larger ones use
                # IVFFlat. `nlist` is sized inside `build_ivfflat` via
                # `_choose_nlist`.
                nlist_target = _choose_nlist(int(vectors.shape[0]))
                # `nlist_target >= 4` is a defensive guard: any batch that
                # clears `IVF_TRAINING_FLOOR` (>= 64 rows) already yields
                # `nlist >= 8` via `_choose_nlist`, so this never fails in
                # practice — it just makes the IVF-trainability invariant
                # explicit at the gate.
                if (
                    INDEX_TYPE == "ivfflat"
                    and nlist_target >= 4
                    and vectors.shape[0] >= IVF_TRAINING_FLOOR
                ):
                    blob = build_ivfflat(vectors, int_ids)
                    index_type_str = "ivfflat"
                else:
                    blob = build_flat(vectors, int_ids)
                    index_type_str = "flat"
                sidecar_blob = _build_sidecar(ids, metas)
                total_vectors = int(vectors.shape[0])
                indexed_uris = sorted(new_parts)

            _write_shard(
                tenant,
                dataset,
                blob,
                sidecar_blob,
                total_vectors,
                index_type_str,
                build_type=build_type,
                indexed_uris=indexed_uris,
                # Carry the prior newest shard's recall watermark FORWARD. The
                # watermark is a per-dataset high-water mark, not a per-build
                # value: an ingest folds landing parts (recall-tier rows are
                # `lsn > watermark`, never touched), so it neither consolidates
                # nor un-consolidates anything and MUST NOT move the watermark.
                # Defaulting to 0 here would REGRESS it, stalling the grace-trim
                # (the 2nd-newest shard would carry watermark 0 → `recall_trim`
                # deletes nothing → recall never drains) and re-unioning already-
                # consolidated rows on every query. Carrying it forward keeps the
                # partition honest and the watermark monotonic.
                consolidated_lsn=int((latest_shard or {}).get("consolidated_lsn", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(tenant, dataset, "error", error_message=f"index build: {exc}")
            _LAST_BUILD["build_type"] = "error"
            return 0

        added = int(vectors.shape[0])
        # rosalinddb.index_build.duration{index_type} — `index_type`
        # (ivfflat|flat) is the only label; no tenant/dataset on the metric.
        obs_metrics.record_index_build_duration(
            (time.time() - build_start) * 1000.0, index_type_str
        )
        # Make the incremental-vs-full distinction observable.
        # `build_type` is the only label on either instrument — no
        # tenant/dataset (would explode Prometheus series count).
        obs_metrics.record_index_build(build_type)
        obs_metrics.record_vectors_added(added, build_type)

        update_dataset_status(
            tenant,
            dataset,
            "indexed",
            last_indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        counter("index_build_minutes", 1)

        # Reclaim storage: prune shards superseded by this build and the
        # landing objects this dataset has now captured. Best-effort — a
        # failure here must not fail the build, which already succeeded.
        try:
            _sweep_superseded_shards(tenant, dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"builder: shard sweep failed for {tenant}/{dataset}: {exc}")
        try:
            _sweep_indexed_landing(tenant, dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"builder: landing sweep failed for {tenant}/{dataset}: {exc}")

        _LAST_BUILD.update(
            build_type=build_type,
            vectors_added=added,
            parts_read=len(new_parts),
            parts_read_uris=list(new_parts),
        )
        return added


# --- DELETE_VECTORS (consolidated-tier delete-by-id) ----------------------


def run_delete_once(dataset: str, tenant: str, vector_id: str) -> int:
    """Apply a single delete-by-id to a dataset's newest consolidated shard.

    Loads the newest shard's FAISS index + sidecar, removes the hashed
    `vector_id`, drops it from the sidecar, and writes a NEW superseded shard
    via the shared `_write_shard` tail — exactly how an incremental ingest
    produces the next shard generation, only the operation is a removal rather
    than an add. The query cache is evicted for the swept shards so a stale
    index can never serve the deleted id.

    Returns the number of vectors removed by this delete: `1` on a hit, or
    `0` when there is nothing to do — no shard yet, or the id is absent from
    the shard (a genuine no-op; the work is DONE and the message is safe to
    ack). Returns the `BUILD_SKIPPED` sentinel if the per-dataset advisory
    lock is held by another replica (the caller redelivers, as for a build).
    Any failure flips the dataset to `error` and returns 0.

    Tenant-scoped: every state/storage call is keyed by `(tenant, dataset)`,
    so a cross-tenant id can never reach another tenant's shard.
    """
    with dataset_build_lock(tenant, dataset) as acquired:
        if not acquired:
            # Another replica is building/deleting this dataset — skip and let
            # the caller redeliver, identical to the build path's contract.
            print(
                f"builder: dataset {tenant}/{dataset} is already being built "
                f"by another replica — skipping delete (will be redelivered)"
            )
            return BUILD_SKIPPED
        return _run_delete_locked(dataset, tenant, vector_id)


def _run_delete_locked(dataset: str, tenant: str, vector_id: str) -> int:
    """Run the delete-by-id — caller holds the per-dataset advisory lock."""
    with build_index_span(tenant=tenant, dataset=dataset):
        latest_shard = get_latest_shard(tenant, dataset)
        if latest_shard is None:
            # No consolidated shard for this dataset — nothing to delete. Clean no-op:
            # leave the dataset status UNTOUCHED. The CP only flips to
            # `indexing` when a shard exists, so a never-ingested (`empty`) or
            # failed (`error`) dataset is still in its real state here — forcing
            # it to `indexed` (with `row_count=0`) would mask that. Nothing to
            # reindex, so there is no status to settle.
            return 0

        try:
            index = faiss.deserialize_index(
                np.frombuffer(read_bytes(latest_shard["shard_uri"]), dtype=np.uint8)
            )
            if not hasattr(index, "id_map"):
                raise RuntimeError(
                    "delete: loaded shard index has no `id_map` "
                    f"(type {type(index).__name__}); cannot remove by id"
                )
            target = _id_to_int64(vector_id)
            existing_int_ids = set(faiss.vector_to_array(index.id_map).tolist())
            if target not in existing_int_ids:
                # The id is not in the shard (already deleted, or never landed
                # in cold). Clean no-op — no new shard. A shard exists here, so
                # the dataset's true state IS `indexed`; the CP flipped it to
                # `indexing` for the in-flight delete, so settle it back to
                # `indexed`. (Safe: with a shard present the real state can only
                # be `indexed` — an `empty`/`error` dataset has no shard and is
                # handled by the no-shard branch above, which leaves status as is.)
                update_dataset_status(tenant, dataset, "indexed")
                return 0

            if _is_ivf_index(index):
                # UNION-REBUILD PATH (#28, same fix as #18's consolidation fold).
                # Deleting `target` from an IVF shard would `_remove_ids([target])`
                # it, but on an `IndexIDMap2(IVFFlat)` that trips a FAISS 1.8.0 C++
                # assertion (`j == index->ntotal`, IndexIDMap.cpp:181) that
                # `abort()`s the whole builder process — NOT a catchable Python
                # error — and even when it does not abort it leaves the IVF
                # inverted lists with non-sequential ids, so the shard's direct
                # map can no longer be built (a later consolidation fold's
                # `reconstruct` would then fail). So instead of removing we REBUILD
                # the shard from scratch as the UNION of the SURVIVORS (every
                # existing id except `target`) with no new rows, preserving each
                # survivor's ORIGINAL int64. The deleted id is in neither set, so
                # it is simply absent from the rebuilt shard.
                existing_sidecar = read_shard_sidecar(latest_shard["shard_uri"])
                blob, sidecar_blob, index_type_str, total_vectors = (
                    _union_rebuild_blob(
                        index,
                        existing_sidecar,
                        drop_int_ids={target},
                        new_ids=[],
                        new_vectors=np.empty((0, int(index.d)), dtype=np.float32),
                        new_metas=[],
                    )
                )
            else:
                # Flat shard: `remove_ids` is safe (only the IVF removal trips the
                # abort), so drop the id in place and re-serialize — no rebuild.
                _remove_ids(index, [target])
                blob = _serialize_index(index)

                # Drop the deleted id from the sidecar so the id/metadata map
                # matches the index exactly. Keyed by the str(int64) hash.
                sidecar = dict(read_shard_sidecar(latest_shard["shard_uri"]))
                sidecar.pop(str(target), None)
                sidecar_blob = json.dumps(sidecar).encode("utf-8")

                index_type_str = latest_shard.get("index_type", "flat")
                total_vectors = int(getattr(index, "ntotal", 0))
            # Carry the superseded shard's landing manifest forward unchanged —
            # the parts are still folded in (minus one removed vector), so a
            # later ingest still treats them as already-indexed.
            indexed_uris = list(latest_shard.get("indexed_landing_uris", []) or [])

            # Label the shard row + metric as a `delete` rebuild — distinct from
            # an ingest's `incremental` so deletes are not miscounted as ingests
            # in `build_type`-keyed observability (the `index_builds` metric and
            # the `shard_catalog.build_type` column). `build_type` is a
            # free-text column (no CHECK constraint), so a new label needs no
            # migration.
            _write_shard(
                tenant,
                dataset,
                blob,
                sidecar_blob,
                total_vectors,
                index_type_str,
                build_type="delete",
                indexed_uris=indexed_uris,
                # Carry the prior newest shard's recall watermark FORWARD (see
                # the ingest path): a delete-by-id removes one cold vector but
                # touches no recall row, so it must not move — let alone regress
                # to 0 — the per-dataset watermark, or the grace-trim stalls.
                consolidated_lsn=int(
                    (latest_shard or {}).get("consolidated_lsn", 0) or 0
                ),
            )
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(
                tenant, dataset, "error", error_message=f"vector delete: {exc}"
            )
            return 0

        obs_metrics.record_index_build("delete")
        update_dataset_status(
            tenant,
            dataset,
            "indexed",
            last_indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Prune the now-superseded shard(s) and evict them from the query
        # cache. Best-effort — the delete already succeeded and is durable.
        try:
            _sweep_superseded_shards(tenant, dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"builder: shard sweep failed for {tenant}/{dataset}: {exc}")

        return 1


# --- CONSOLIDATE (recall -> consolidated flush) ---------------------------
#
# The recall→consolidated flush: snapshot a (tenant, dataset) recall partition
# up to its current max LSN N, fold the LIVE rows into a new Consolidated shard,
# apply tombstones (deleted ids are removed + never carried forward), COMMIT the
# catalog row with `consolidated_lsn = N` (`build_type='consolidate'`), run the
# supersede sweep + evict superseded shards from the query cache, and THEN trim
# the recall rows — grace-bounded (up to the 2nd-newest shard's watermark) and
# idempotent.
#
# ORDER IS LOAD-BEARING (I2): build → commit catalog → grace-bounded trim. NEVER
# trim before commit. A crash between commit and trim leaves recall rows with
# `lsn <= consolidated_lsn` that the union harmlessly excludes (`lsn >
# consolidated_lsn`) and the next consolidation GCs — safe across the two
# databases WITHOUT a distributed transaction. See
# docs/architecture/recall-consolidate.md, invariants I1/I2/I3/I4.


def run_consolidate_once(dataset: str, tenant: str) -> int:
    """Consolidate a (tenant, dataset) recall partition into a Consolidated shard.

    Gated by the per-dataset advisory lock (single-replica serialization),
    exactly like `run_once` / `run_delete_once`. Returns the number of LIVE
    recall rows folded into the new shard (`0` when the partition is empty — a
    clean no-op, nothing to consolidate, message safe to ack), or the
    `BUILD_SKIPPED` sentinel when another replica holds the lock (the caller
    redelivers). Any failure flips the dataset to `error` and returns 0.

    No-op when the recall tier is OFF (`recall_enabled()` False): a `CONSOLIDATE`
    message can only have been enqueued under the flag, but guard defensively so
    a stray message with the flag off never opens a recall connection.
    """
    if not recall_enabled():
        # Defensive: the flag is off, so there is no recall store to drain.
        # Treat as a clean no-op (ack) — never open a recall connection.
        return 0
    with dataset_build_lock(tenant, dataset) as acquired:
        if not acquired:
            print(
                f"builder: dataset {tenant}/{dataset} is already being built "
                f"by another replica — skipping consolidate (will be redelivered)"
            )
            return BUILD_SKIPPED
        return _run_consolidate_locked(dataset, tenant)


def _grace_watermark(tenant: str, dataset: str) -> int:
    """Return the 2nd-newest shard's `consolidated_lsn` — the grace-bounded trim
    watermark (I4).

    The recall trim deletes rows only up to the watermark of the shard that is
    now ≥ 2 generations old, i.e. the **2nd-newest** shard — symmetric to the
    `SHARD_KEEP=2` sweep that retains the newest shard plus one grace buffer. An
    in-flight query that resolved the (now second-newest) shard moments before
    this consolidation wrote the newest one still filters recall with that older
    shard's smaller watermark, so its rows must remain in recall until that
    shard itself is superseded. Trimming only up to the 2nd-newest watermark
    guarantees that.

    Returns `0` when there is no 2nd-newest shard yet (the dataset has 0 or 1
    shards) — the first consolidation trims nothing, which is correct: its new
    shard is the only one, so an in-flight query can only have resolved it (or
    no shard) and every recall row is still needed.

    Called AFTER the new shard is committed, so `list_shards` already includes
    it: index `[1]` is the shard immediately before the just-written newest.
    """
    shards = list_shards(tenant, dataset)
    if len(shards) < 2:
        return 0
    second_newest = shards[1]
    raw = second_newest.get("consolidated_lsn", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _run_consolidate_locked(dataset: str, tenant: str) -> int:
    """Run the consolidation — caller holds the per-dataset advisory lock.

    build → commit catalog (watermark N) → grace-bounded trim. See
    `run_consolidate_once`.
    """
    with build_index_span(tenant=tenant, dataset=dataset):
        # 1. SNAPSHOT the recall partition up to its current max LSN N. The read
        #    is a SINGLE statement (the bound N is derived in a scalar sub-SELECT
        #    of the same query), so N and the rows it selects come from one MVCC
        #    snapshot and are self-consistent regardless of the recall writer's
        #    internals; a write that lands after this (higher LSN) stays in recall
        #    and the union keeps serving it (read-your-writes through consolidation).
        try:
            max_lsn, recall_rows = recall_snapshot_for_consolidation(tenant, dataset)
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(
                tenant, dataset, "error", error_message=f"consolidate snapshot: {exc}"
            )
            return 0

        if max_lsn == 0 or not recall_rows:
            # Empty partition — nothing to consolidate. Clean no-op: leave the
            # dataset status untouched (a never-written-to recall partition has
            # no in-flight build to settle).
            return 0

        # Partition the snapshot into LIVE upserts (folded into the shard) and
        # tombstones (their ids removed from the shard + never added). A row's
        # `deleted` flag is last-write-wins per id already (recall UPSERT), and
        # the snapshot has at most one row per id (PK is (tenant, dataset, id)),
        # so the two sets are disjoint by id.
        live_rows = [r for r in recall_rows if not r["deleted"]]
        tombstone_ids = [r["id"] for r in recall_rows if r["deleted"]]

        latest_shard = get_latest_shard(tenant, dataset)
        is_incremental = latest_shard is not None

        try:
            shard_uri = _build_consolidated_shard(
                tenant, dataset, latest_shard, live_rows, tombstone_ids, max_lsn
            )
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(
                tenant, dataset, "error", error_message=f"consolidate build: {exc}"
            )
            _LAST_BUILD["build_type"] = "error"
            return 0

        if shard_uri is None:
            # Nothing was actually written (e.g. only tombstones for ids that
            # were never in the cold shard, and no live rows + no prior shard).
            # The catalog is unchanged, so there is no watermark to advance and
            # nothing to trim safely — leave recall as is for the next pass.
            return 0

        # 2. The catalog row is COMMITTED with `consolidated_lsn = N` (inside
        #    `_build_consolidated_shard` -> `_write_shard` -> `add_shard`). Only
        #    NOW (strictly after commit — I2) do we touch recall.
        obs_metrics.record_index_build("consolidate")
        update_dataset_status(
            tenant,
            dataset,
            "indexed",
            last_indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # 3. Sweep superseded shards (keep newest 2) + evict their query-cache
        #    entries — same discipline as a build/delete. Best-effort. Runs
        #    BEFORE the trim does not matter for correctness (the trim's
        #    grace-bound reads `list_shards` itself), but mirrors build/delete.
        try:
            _sweep_superseded_shards(tenant, dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"builder: shard sweep failed for {tenant}/{dataset}: {exc}")

        # 4. GRACE-BOUNDED, IDEMPOTENT TRIM (I4). Delete recall rows only up to
        #    the 2nd-newest shard's watermark, so an in-flight query that
        #    resolved an older shard still finds its recall rows. Best-effort —
        #    a trim failure leaves rows the union harmlessly excludes
        #    (`lsn > consolidated_lsn`) and the next consolidation GCs them
        #    (cross-DB crash safety). The trim must run AFTER the commit above.
        grace = _grace_watermark(tenant, dataset)
        try:
            trimmed = recall_trim(tenant, dataset, grace)
        except Exception as exc:  # noqa: BLE001
            print(
                f"builder: recall trim failed for {tenant}/{dataset} "
                f"(grace_watermark={grace}): {exc}"
            )
            trimmed = 0

        _LAST_BUILD.update(
            build_type="consolidate",
            vectors_added=len(live_rows),
            parts_read=0,
            parts_read_uris=[],
        )
        print(
            f"builder: consolidated {tenant}/{dataset} — folded {len(live_rows)} "
            f"live + {len(tombstone_ids)} tombstones, watermark={max_lsn}, "
            f"incremental={is_incremental}, trimmed={trimmed} (grace={grace})"
        )
        return len(live_rows)


def _build_consolidated_shard(
    tenant: str,
    dataset: str,
    latest_shard: Optional[dict],
    live_rows: List[dict],
    tombstone_ids: List[str],
    consolidated_lsn: int,
) -> Optional[str]:
    """Fold a recall snapshot into a new Consolidated shard; return its URI.

    Reuses the existing build tail: an INCREMENTAL fold loads the current shard's
    FAISS index + sidecar, removes any overlapping/tombstoned ids, `add()`s the
    live rows, merges the sidecar, and writes via `_write_shard`; a FROM-SCRATCH
    fold (no prior shard) trains a fresh index over the live rows. The catalog
    row is committed with `build_type='consolidate'` and `consolidated_lsn` set
    (the watermark, I2). The landing manifest is carried forward unchanged — a
    consolidation does not touch landing parts (recall data never lands).

    Returns the new shard URI, or `None` when there is genuinely nothing to
    write (no prior shard AND no live rows — a tombstone-only first
    consolidation): there is no index to remove from and nothing to add, so the
    catalog is left unchanged and the caller skips the watermark advance + trim.
    """
    live_ids = [r["id"] for r in live_rows]
    live_metas = [r["metadata"] for r in live_rows]
    if live_rows:
        vectors = np.array([r["values"] for r in live_rows], dtype=np.float32)
        live_int_ids = np.array([_id_to_int64(i) for i in live_ids], dtype=np.int64)
    else:
        vectors = np.empty((0, 0), dtype=np.float32)
        live_int_ids = np.array([], dtype=np.int64)

    if latest_shard is not None:
        # --- incremental fold onto the current shard's index ---------------
        index = faiss.deserialize_index(
            np.frombuffer(read_bytes(latest_shard["shard_uri"]), dtype=np.uint8)
        )
        if not hasattr(index, "id_map"):
            raise RuntimeError(
                "consolidate: loaded shard index has no `id_map` "
                f"(type {type(index).__name__}); cannot upsert by id"
            )
        existing_int_ids = set(faiss.vector_to_array(index.id_map).tolist())
        existing_sidecar = read_shard_sidecar(latest_shard["shard_uri"])
        index_type_str = latest_shard.get("index_type", "flat")

        # The set of cold ids being replaced (live re-upserts, last-write-wins)
        # or deleted (tombstones). Both must drop their stale cold copy.
        tombstone_int_ids = [_id_to_int64(i) for i in tombstone_ids]
        replaced_or_removed = {
            int(i)
            for i in (list(live_int_ids) + tombstone_int_ids)
            if int(i) in existing_int_ids
        }
        # Recall data never lands, so the landing manifest is unchanged.
        indexed_uris = list(latest_shard.get("indexed_landing_uris", []) or [])

        if not replaced_or_removed:
            # APPEND-ONLY FAST PATH: no overlap and no tombstone hits an existing
            # id — nothing to remove. Cheap `add_with_ids` onto the loaded index,
            # no rebuild. This is the common case and never trips the FAISS abort.
            if live_rows:
                _add_to_index(index, vectors, live_int_ids)
            blob = _serialize_index(index)
            merged_sidecar = dict(existing_sidecar)
            merged_sidecar.update(_sidecar_dict(live_ids, live_metas))
            sidecar_blob = json.dumps(merged_sidecar).encode("utf-8")
            total_vectors = int(getattr(index, "ntotal", 0))
        else:
            # UNION-REBUILD PATH (#18). The overlap set is non-empty, so we would
            # otherwise `remove_ids` the stale copies. On an `IndexIDMap2(IVFFlat)`
            # that trips a FAISS 1.8.0 C++ assertion (`j == index->ntotal`,
            # IndexIDMap.cpp:181) that `abort()`s the whole builder process — it is
            # NOT a catchable Python error. So instead of remove+add we REBUILD the
            # shard as a correct UNION via the crash-free from-scratch path:
            #   surviving cold vectors (existing ids NOT replaced/tombstoned)
            #   ∪ the new/updated live vectors (carrying their NEW values).
            # Reconstruction of IVFFlat survivors is lossless (raw float32). The
            # rebuild also re-trains the IVF quantizer over the union, which keeps
            # recall sound as the dataset evolves.
            #
            # NOTE (out of scope, follow-ups): FAISS asserts via C++ `abort()`,
            # which Python cannot catch — a defensive subprocess-isolated fold (so
            # any future FAISS abort becomes a handled nack, not a dead consumer)
            # and the consolidate-queue hygiene (reaper for unstamped messages,
            # producers routing through `publish()`) are tracked separately.
            survivor_int_ids = [
                int(i) for i in existing_int_ids if int(i) not in replaced_or_removed
            ]
            survivor_vectors = _reconstruct_surviving(index, survivor_int_ids)
            # Map each survivor int64 back to its original (string id, metadata)
            # via the existing sidecar — for the SIDECAR only. A survivor missing
            # from the sidecar (older shard / partial-write meta — read_shard_sidecar
            # degrades to {} on an unreadable sidecar) still keeps its vector; we
            # synthesise a stable string id from the int64 so it has a sidecar
            # entry and is never silently dropped from the union.
            #
            # CRITICAL (review P1): the rebuilt index must carry each survivor's
            # ORIGINAL int64 (`sid`) — the exact hash its vector was reconstructed
            # under — NOT a re-hash of `survivor_ids`. For a present survivor
            # `_id_to_int64(entry["id"]) == sid` (round-trips), but for a
            # missing-from-sidecar survivor the synthesised `str(sid)` re-hashes to
            # `_id_to_int64(str(sid)) != sid`, stamping the vector under a WRONG
            # int64 — unreachable by its true id (get/delete/upsert all hash the
            # true id back to `sid`) and un-removable by a future tombstone. So we
            # CONCATENATE the actual `survivor_int_ids` with the live int64s rather
            # than re-hashing `union_ids`. (`survivor_int_ids`/`survivor_vectors`/
            # `survivor_ids` are index-aligned, so the sidecar stays consistent: a
            # present survivor's entry is keyed by `str(sid)`; a missing one is
            # keyed by `str(sid)` with fallback metadata.)
            survivor_ids: list[str] = []
            survivor_metas: list[dict] = []
            for sid in survivor_int_ids:
                entry = existing_sidecar.get(str(sid))
                if entry is not None:
                    survivor_ids.append(entry.get("id", str(sid)))
                    survivor_metas.append(entry.get("metadata", {}) or {})
                else:
                    survivor_ids.append(str(sid))
                    survivor_metas.append({})

            # Concatenate survivors with the new/updated live rows. Live ids carry
            # their NEW vectors/metadata; tombstoned ids appear in neither set.
            union_ids = survivor_ids + list(live_ids)
            union_metas = survivor_metas + list(live_metas)
            if survivor_vectors.shape[0] and live_rows:
                union_vectors = np.concatenate([survivor_vectors, vectors], axis=0)
            elif live_rows:
                union_vectors = vectors
            else:
                union_vectors = survivor_vectors
            # Carry survivors' ORIGINAL int64s through unchanged (see CRITICAL note
            # above); only the live rows are hashed from their string ids. Order
            # matches `union_vectors`/`union_ids`/`union_metas` (survivors first).
            union_int_ids = np.array(
                survivor_int_ids + [_id_to_int64(i) for i in live_ids],
                dtype=np.int64,
            )

            # Rebuild via the from-scratch builder selection (IVFFlat when the
            # union is large enough to train, else flat) — never `remove_ids`.
            nlist_target = _choose_nlist(int(union_vectors.shape[0]))
            if (
                INDEX_TYPE == "ivfflat"
                and nlist_target >= 4
                and union_vectors.shape[0] >= IVF_TRAINING_FLOOR
            ):
                blob = build_ivfflat(union_vectors, union_int_ids)
                index_type_str = "ivfflat"
            else:
                blob = build_flat(union_vectors, union_int_ids)
                index_type_str = "flat"
            # Keep index↔sidecar consistent: stamp the sidecar under the SAME
            # int64s carried into the index (`union_int_ids`), not a re-hash of
            # `union_ids`. `_build_sidecar`/`_sidecar_dict` key by
            # `str(_id_to_int64(raw_id))`, which would re-derive the WRONG key for a
            # missing-from-sidecar survivor (the same bug, on the sidecar side), so
            # we build the mapping directly from `union_int_ids`.
            sidecar_blob = json.dumps(
                {
                    str(int(int64_id)): {
                        "id": raw_id,
                        "metadata": meta if isinstance(meta, dict) else {},
                    }
                    for int64_id, raw_id, meta in zip(
                        union_int_ids.tolist(), union_ids, union_metas
                    )
                }
            ).encode("utf-8")
            total_vectors = int(union_vectors.shape[0])
    else:
        # --- from-scratch fold (no prior shard) ----------------------------
        if not live_rows:
            # Tombstone-only first consolidation: no index to write. The trim is
            # still safe to run later (the rows are tombstones for ids that were
            # never consolidated), but there is no shard to advance — return None
            # so the caller leaves the catalog untouched and trims nothing this
            # pass. (The tombstones simply wait for a future consolidation that
            # also carries live rows, or age out via consolidate-on-idle once
            # there is a shard. They never produce a stale read — there is no
            # cold shard for them to leak from.)
            return None
        nlist_target = _choose_nlist(int(vectors.shape[0]))
        if (
            INDEX_TYPE == "ivfflat"
            and nlist_target >= 4
            and vectors.shape[0] >= IVF_TRAINING_FLOOR
        ):
            blob = build_ivfflat(vectors, live_int_ids)
            index_type_str = "ivfflat"
        else:
            blob = build_flat(vectors, live_int_ids)
            index_type_str = "flat"
        sidecar_blob = _build_sidecar(live_ids, live_metas)
        total_vectors = int(vectors.shape[0])
        indexed_uris = []

    # COMMIT the new shard + catalog row with the watermark (I2). `_write_shard`
    # writes the .bin/.meta.json, calls `add_shard(..., consolidated_lsn=N)`,
    # reconciles row_count, and publishes the prewarm hint.
    return _write_shard(
        tenant,
        dataset,
        blob,
        sidecar_blob,
        total_vectors,
        index_type_str,
        build_type="consolidate",
        indexed_uris=indexed_uris,
        consolidated_lsn=consolidated_lsn,
    )


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoints."""

    def do_GET(self):
        """Handle GET requests for metrics."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(snapshot()).encode())
        elif self.path == "/prometheus":
            self._serve_prometheus()
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "service": "index_builder"}')
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_prometheus(self):
        """Serve Prometheus format metrics."""
        try:
            from prometheus_client import CollectorRegistry, generate_latest, Gauge
        except ImportError:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"prometheus-client not installed")
            return

        reg = CollectorRegistry()
        snap = snapshot()
        counters = snap.get("counters", {})
        gauges = snap.get("gauges", {})
        timers = snap.get("timers", {})

        # Export counters as gauges
        for name, value in counters.items():
            g = Gauge(f"builder_{name}", f"builder counter {name}", registry=reg)
            g.set(float(value))

        # Export gauges
        for name, value in gauges.items():
            g = Gauge(f"builder_{name}", f"builder gauge {name}", registry=reg)
            g.set(float(value))

        # Export timer stats
        for name, values in timers.items():
            if values:
                count = len(values)
                avg_ms = (sum(values) / count) * 1000.0
                Gauge(f"builder_{name}_count", f"builder timer {name} count", registry=reg).set(float(count))
                Gauge(f"builder_{name}_avg_ms", f"builder timer {name} avg ms", registry=reg).set(float(avg_ms))

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(generate_latest(reg))

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def start_metrics_server():
    """Start the metrics HTTP server in a background thread."""
    def run_server():
        server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"Metrics server started on port {METRICS_PORT}")


def main_loop():
    """Blocking loop that builds shards, applies deletes, and consolidates.

    Consumes THREE topics: `DATASET_READY` (an ingest needs folding into a
    shard), `DELETE_VECTORS` (a consolidated-tier delete-by-id needs applying to
    the newest shard), and `CONSOLIDATE` (a recall partition needs flushing into
    a Consolidated shard + the watermark advanced — the recall→consolidated
    flush). Each iteration drains one message from each topic so a steady stream
    of one never starves the others; `DATASET_READY` is polled with a short
    block so the loop still parks when all are idle.

    Consolidate-on-idle: each idle tick (when the blocking `DATASET_READY` poll
    times out) the loop runs a lightweight sweep that enqueues `CONSOLIDATE` for
    every recall partition whose newest write is older than `RB_RECALL_IDLE_S`,
    draining idle datasets to zero recall rows (scale-to-zero). The sweep ONLY
    runs under `recall_enabled()` and is rate-limited so it does not hammer the
    recall store; with the flag off it is never reached.

    Reliable-queue contract: a message is `ack`-ed once its handler reaches a
    terminal outcome (success, or a handled failure that flipped the
    dataset/import to `error`/`failed`); an UNHANDLED crash `nack`s it for
    redelivery (then dead-lettering past `QUEUE_MAX_ATTEMPTS`).

    Builder-skip contract: when the per-dataset advisory lock is held by
    another replica the work does NOT run. The message must NOT be `ack`-ed
    — that would discard it, and the skipped message may carry a newer upload
    than the in-progress build. The handler returns `False` in that case and
    the loop `nack`s the message with requeue so it is redelivered and retried
    once the winning build releases the lock.

    The index builder also HOSTS the reconciliation reaper as a background
    thread (`start_reaper_thread`). The reaper is a periodic task rather than a
    new service so the deploy keeps one process group per logical service; the
    builder is single-replica and always running, which makes it the natural
    home. On `SIGTERM` the loop stops pulling new messages and the reaper
    thread is signalled to stop via the shared shutdown event.
    """
    migrate()
    install_signal_handlers()
    start_metrics_server()
    start_reaper_thread(stop_event())
    while not should_stop():
        # `DATASET_READY` is the blocking poll so the loop parks when idle;
        # `DELETE_VECTORS` and `CONSOLIDATE` are then drained non-blocking so
        # neither waits a full `DATASET_READY` timeout behind an empty build
        # queue.
        ready_msg = consume("DATASET_READY", block=True, timeout=1.0)
        if ready_msg:
            _dispatch(ready_msg, _handle_dataset_ready, "build")
        else:
            # Idle tick (the blocking poll timed out): run the lightweight
            # consolidate-on-idle sweep. No-op + opens no recall connection when
            # the flag is off.
            _maybe_sweep_idle_recall()
        delete_msg = consume("DELETE_VECTORS", block=False)
        if delete_msg:
            _dispatch(delete_msg, _handle_delete_vectors, "delete")
        consolidate_msg = consume("CONSOLIDATE", block=False)
        if consolidate_msg:
            _dispatch(consolidate_msg, _handle_consolidate, "consolidate")
    print("builder: shutdown signal received — exiting consume loop")


# Last wall-clock the idle sweep ran. The sweep is rate-limited to once per
# `RB_RECALL_IDLE_S` window so the idle tick (every ~1s when the build queue is
# empty) does not hammer the recall store with a GROUP BY scan every second.
_LAST_IDLE_SWEEP_AT: float = 0.0


def _maybe_sweep_idle_recall() -> None:
    """Consolidate-on-idle: enqueue `CONSOLIDATE` for idle recall partitions.

    Runs on the builder loop's idle tick. No-op (and opens NO recall connection)
    unless `recall_enabled()`. Rate-limited to once per idle window so the
    GROUP BY scan over the recall store runs at most ~once per `RB_RECALL_IDLE_S`,
    not every ~1s idle tick. Each idle `(tenant, dataset)` partition gets one
    `CONSOLIDATE` enqueue, which the consumer drains to ZERO recall rows — after
    which idle queries skip pgvector entirely (scale-to-zero preserved).

    Best-effort: a recall-store error is logged and the next tick retries; it
    must never crash the build loop.
    """
    if not recall_enabled():
        return
    global _LAST_IDLE_SWEEP_AT
    idle_seconds = _recall_idle_seconds()
    now = time.time()
    if now - _LAST_IDLE_SWEEP_AT < idle_seconds:
        return
    _LAST_IDLE_SWEEP_AT = now
    try:
        partitions = recall_idle_partitions(idle_seconds)
    except Exception as exc:  # noqa: BLE001
        print(f"builder: consolidate-on-idle sweep failed: {exc}")
        return
    for tenant, dataset in partitions:
        try:
            publish("CONSOLIDATE", {"tenant": tenant, "dataset": dataset})
        except Exception as exc:  # noqa: BLE001
            print(
                f"builder: consolidate-on-idle enqueue failed for "
                f"{tenant}/{dataset}: {exc}"
            )


def _dispatch(msg, handler, kind: str) -> None:
    """Run one queue `handler` for `msg`, then ack / nack per its contract.

    Shared ack/nack discipline for both the `DATASET_READY` and
    `DELETE_VECTORS` consume paths: the handler returns `True` on a terminal
    outcome (ack) or `False` on a per-dataset-lock skip (nack+requeue); an
    unhandled exception also nacks for redelivery. `kind` is a label for the
    log line only.
    """
    try:
        done = handler(msg)
    except Exception as exc:  # noqa: BLE001
        print(f"builder: unhandled {kind} error, nacking message: {exc}")
        nack(msg, requeue=True)
        return
    if done:
        ack(msg)
    else:
        # SKIPPED (per-dataset lock held by another replica). Redeliver — do
        # NOT ack — so the message is retried after the in-progress build
        # commits. Acking here would lose genuine work.
        print(
            f"builder: {kind} skipped (per-dataset lock held) — "
            "nacking message for redelivery"
        )
        nack(msg, requeue=True)


def _handle_dataset_ready(msg) -> bool:
    """Build a shard for one DATASET_READY message.

    Returns `True` once the build has reached a terminal state (success or a
    handled failure) — the caller then acks. Returns `False` if the build was
    SKIPPED because another replica holds the per-dataset lock — the caller
    then `nack`s the message for redelivery. Raising propagates to the caller's
    nack path so the message is redelivered.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    import_id = msg.get("import_id")
    try:
        result = run_once(dataset, tenant)
    except Exception as exc:  # noqa: BLE001
        # An unhandled crash in `run_once` for an import-driven build must NOT
        # leave the job stuck in `indexing` forever. Flip it to `failed` so
        # the status is terminal, then re-raise so the queue message is
        # redelivered — RosalindDB builds are idempotent (the shard manifest
        # makes a duplicate DATASET_READY a no-op), so a retry is safe.
        print(f"builder: run_once dataset={dataset} crashed: {exc}")
        if import_id:
            _fail_import_job(import_id, f"index build crashed: {exc}")
        raise

    if result == BUILD_SKIPPED:
        # Another replica holds the per-dataset lock — the build did NOT run.
        # Do not finalize the import or ack: signal the caller to nack so the
        # message is redelivered and the build retried. The import job stays
        # in `indexing` (correct — the build is still pending), and the retry
        # finalizes it.
        return False

    # If this build was driven by an async bulk-import job, mark the job
    # terminal now that the build step has finished.
    if import_id:
        try:
            from services.validator_worker.run import (
                fail_import,
                finalize_import,
            )

            if _LAST_BUILD.get("build_type") == "error":
                # `run_once` handled the failure internally (flipped the
                # dataset to `error`) but did not raise. The job must still
                # end `failed`, not be stranded in `indexing`.
                fail_import(import_id, "index build failed")
            else:
                finalize_import(import_id)
        except Exception as exc:  # noqa: BLE001
            # Even the finalize bookkeeping crashing must not strand the
            # job — best-effort flip it to `failed`.
            print(f"builder: finalize import={import_id} failed: {exc}")
            _fail_import_job(import_id, f"import finalize crashed: {exc}")

    # The build reached a terminal outcome — signal the caller to ack.
    return True


def _handle_delete_vectors(msg) -> bool:
    """Apply one DELETE_VECTORS message to the dataset's newest shard.

    Returns `True` once the delete reaches a terminal state — a hit, a clean
    no-op (no shard / id absent), or a handled failure that flipped the
    dataset to `error`; the caller then acks. Returns `False` if the delete
    was SKIPPED because another replica holds the per-dataset lock; the caller
    then `nack`s for redelivery. A missing `id` is treated as a malformed
    message and acked away (no terminal status to set) rather than redelivered
    forever.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    vector_id = msg.get("id")
    if not isinstance(vector_id, str) or not vector_id:
        # Malformed message — nothing to delete. Ack it away rather than
        # redeliver a payload that can never succeed.
        print(f"builder: DELETE_VECTORS for {tenant}/{dataset} missing 'id' — acking")
        return True
    result = run_delete_once(dataset, tenant, vector_id)
    if result == BUILD_SKIPPED:
        # Another replica holds the per-dataset lock — signal a nack/redeliver.
        return False
    # Hit, no-op, or handled error all reached a terminal outcome — ack.
    return True


def _handle_consolidate(msg) -> bool:
    """Consolidate one CONSOLIDATE message — flush a recall partition to a shard.

    Returns `True` once the consolidation reaches a terminal state — folded,
    a clean no-op (empty recall partition / flag off), or a handled failure that
    flipped the dataset to `error`; the caller then acks. Returns `False` if it
    was SKIPPED because another replica holds the per-dataset lock; the caller
    then `nack`s for redelivery. A duplicate `CONSOLIDATE` (cap + idle can both
    enqueue one) is idempotent: the second run snapshots whatever recall rows
    remain and either folds them or is a clean no-op.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    result = run_consolidate_once(dataset, tenant)
    if result == BUILD_SKIPPED:
        # Another replica holds the per-dataset lock — signal a nack/redeliver.
        return False
    # Folded, no-op, or handled error all reached a terminal outcome — ack.
    return True


def _fail_import_job(import_id: str, message: str) -> None:
    """Best-effort flip an import job to `failed` (catch-all for the builder)."""
    try:
        from services.validator_worker.run import fail_import

        fail_import(import_id, message)
    except Exception as exc:  # noqa: BLE001
        print(f"builder: could not fail import={import_id}: {exc}")


if __name__ == "__main__":
    main_loop()
