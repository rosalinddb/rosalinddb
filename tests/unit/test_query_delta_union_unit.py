"""Unit coverage for the delta-tier query read union (PR-C).

Hermetic — `memory://` storage + `memory://` control-plane state, no Docker, no
network. Builds base+delta FAISS shards DIRECTLY with faiss (mirroring
compaction-redesign.md §0: a BARE `IndexIVFFlat` whose coarse quantizer is a
clone of one trained-empty master, native `add_with_ids` ids), writes the `.bin`
+ `.meta.json` sidecar into `memory://` storage, and registers catalog rows via
`state.add_shard(... level=, parent_shard_id=, quantizer_version=,
covered_lsn_lo=, covered_lsn_hi=, tombstone_int_ids=)`. This keeps PR-C
independent of the PR-B index_builder.

Headline properties proven here (spec §4.3, §3.1–3.3, §3.7, §5.1–5.3):

  - **Flag OFF (default)**: `_resolve_shard` is byte-identical to today
    (`resolved["shard"] = shards[0]`, no `"shards"` key) and `_watermark_for_shard`
    is unchanged. The whole existing query suite proves the rest.
  - **Flag ON resolution**: `_resolve_shard` sets `resolved["shards"] = [base] +
    live_deltas` ordered by `covered_lsn_lo` from ONE catalog snapshot, and stashes
    the resolved rows' `tombstone_int_ids`. The no-shard signal (return None) is
    preserved.
  - **Frontier watermark**: a contiguous base+delta cover yields the last
    contiguous `hi`; a GAP clamps to the contiguous max BEFORE the gap (never
    `max(consolidated_lsn)`); the legacy base anchors at `consolidated_lsn` when
    `covered_lsn_hi == 0`.
  - **Loop-search union parity**: the union over base+deltas returns the same
    ids/ranking as an equivalent single (monolithic) index over all the vectors
    (recall parity, shared frozen quantizer — P0-C).
  - **Tombstone suppression**: a cold-vs-cold delete carried as a delta's
    `tombstone_int_ids` (int64) suppresses the matching cold id, mapped back to its
    string id via the per-shard sidecars.
  - **Unreadable frontier shard**: a delta whose `.bin` is missing raises the
    consolidated-error path → `run_query` returns 503 (never silently narrows the
    cold set).
"""
from __future__ import annotations

import json

import faiss  # type: ignore
import numpy as np
import pytest

import adapters.state.state as state_mod
import services.query_api.v1_query as v1q
from adapters.landing.parquet_reader import id_to_int64
from adapters.storage import storage


# --- fixtures: build base+delta shards directly --------------------------------

DIM = 8
NLIST = 4


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """Per-test isolation: clear memory shards + the shard cache, delta flag off."""
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    # Catalog cache off (the SSD-tier gate is unset) so `_cached_list_shards`
    # passes straight through to `list_shards` — every test reads live state.
    monkeypatch.delenv("RB_SHARD_TIER_BYTES", raising=False)
    # `CACHE_DIR` defaults to /var/cache/shards (unwritable in the dev sandbox);
    # point the shard download cache at a writable tmp dir. The constant is
    # captured at import, so patch the module attribute.
    monkeypatch.setattr(v1q, "CACHE_DIR", str(tmp_path / "cache"))
    state_mod._MEM_SHARDS.clear()
    with storage._MEM_LOCK:
        storage._MEM_OBJECTS.clear()
    v1q.cache_clear()
    yield
    state_mod._MEM_SHARDS.clear()
    with storage._MEM_LOCK:
        storage._MEM_OBJECTS.clear()
    v1q.cache_clear()


def _master_quantizer(train_vectors: np.ndarray) -> faiss.IndexIVFFlat:
    """A trained-empty bare `IndexIVFFlat` — the shared frozen quantizer master.

    Mirrors compaction-redesign.md §0/§2.1: train ONCE on a sample, then every
    shard `clone_index`es this (stays `is_trained=True`, no per-add retrain) so the
    base+delta union probes the SAME Voronoi cells (lossless vs a monolith).
    """
    quantizer = faiss.IndexFlatL2(DIM)
    master = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_L2)
    master.train(train_vectors)
    return master


def _build_bare_ivf_shard(master, ids: list[str], vectors: np.ndarray) -> bytes:
    """Build a BARE IndexIVFFlat (clone of master) + native add_with_ids; serialize.

    No IDMap2 wrap — search on a bare IVF still returns the native int64 ids
    (validated FAISS fact). The read path must handle this shape.
    """
    index = faiss.clone_index(master)
    int_ids = np.array([id_to_int64(i) for i in ids], dtype=np.int64)
    index.add_with_ids(vectors, int_ids)
    blob = faiss.serialize_index(index)
    return blob.tobytes() if isinstance(blob, np.ndarray) else blob


def _sidecar_blob(ids: list[str], metas: list[dict]) -> bytes:
    mapping = {
        str(id_to_int64(i)): {"id": i, "metadata": m or {}}
        for i, m in zip(ids, metas)
    }
    return json.dumps(mapping).encode("utf-8")


def _write_shard_objects(uri: str, blob: bytes, ids: list[str], metas: list[dict]):
    storage.write_bytes(uri, blob)
    storage.write_bytes(f"{uri}.meta.json", _sidecar_blob(ids, metas))


def _register(
    tenant, dataset, uri, vectors, *, level, parent_shard_id, quantizer_version,
    covered_lsn_lo, covered_lsn_hi, consolidated_lsn, tombstone_int_ids=None,
):
    return state_mod.add_shard(
        tenant, dataset, uri, checksum="x", vector_count=int(len(vectors)),
        index_type="ivfflat",
        build_type=("consolidate" if level == 0 else "consolidate-delta"),
        consolidated_lsn=consolidated_lsn,
        quantizer_version=quantizer_version,
        parent_shard_id=parent_shard_id,
        level=level,
        covered_lsn_lo=covered_lsn_lo,
        covered_lsn_hi=covered_lsn_hi,
        tombstone_int_ids=tombstone_int_ids,
    )


def _rng_vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


class _Generation:
    """A built base+delta generation: monolithic ground-truth index + catalog rows."""

    def __init__(self, tenant, dataset, master, all_ids, all_vecs, all_metas):
        self.tenant = tenant
        self.dataset = dataset
        self.master = master
        self.all_ids = all_ids
        self.all_vecs = all_vecs
        self.all_metas = all_metas
        # Monolithic single index over ALL vectors (the parity oracle).
        mono = faiss.clone_index(master)
        mono.add_with_ids(
            all_vecs, np.array([id_to_int64(i) for i in all_ids], dtype=np.int64)
        )
        self.mono = mono
        self.mono_sidecar = {
            str(id_to_int64(i)): {"id": i, "metadata": m or {}}
            for i, m in zip(all_ids, all_metas)
        }


def _make_generation(
    tenant="t1", dataset="ds", n_base=40, n_per_delta=(10, 10), seed=7,
    tombstone_ids=None, contiguous=True, base_uri="memory://idx/base.bin",
):
    """Build a base + len(n_per_delta) deltas, all under one shared quantizer.

    Returns `(_Generation, base_id, [delta_ids])`. LSN bands are contiguous from
    the base unless `contiguous=False`, in which case the SECOND delta's lo is
    bumped to leave a gap (a stale-cache missing-delta simulation when only the
    first delta is resolved).
    """
    train = _rng_vectors(max(64, n_base + sum(n_per_delta)), seed=seed + 100)
    master = _master_quantizer(train)

    # Base.
    base_ids = [f"b{i}" for i in range(n_base)]
    base_vecs = _rng_vectors(n_base, seed=seed)
    base_metas = [{"src": "base"} for _ in base_ids]
    base_hi = 100
    base_blob = _build_bare_ivf_shard(master, base_ids, base_vecs)
    _write_shard_objects(base_uri, base_blob, base_ids, base_metas)
    base_id = _register(
        tenant, dataset, base_uri, base_vecs, level=0, parent_shard_id=None,
        quantizer_version=1, covered_lsn_lo=0, covered_lsn_hi=base_hi,
        consolidated_lsn=base_hi,
    )

    all_ids = list(base_ids)
    all_vecs = [base_vecs]
    all_metas = list(base_metas)
    delta_ids_db = []
    prev_hi = base_hi
    next_id = n_base
    for d_idx, n in enumerate(n_per_delta):
        ids = [f"d{d_idx}_{j}" for j in range(n)]
        vecs = _rng_vectors(n, seed=seed + 10 + d_idx)
        metas = [{"src": f"delta{d_idx}"} for _ in ids]
        lo = prev_hi + 1
        if not contiguous and d_idx == 1:
            lo = prev_hi + 50  # gap: this delta's band does not abut the prior
        hi = lo + 9
        uri = f"memory://idx/delta{d_idx}.bin"
        blob = _build_bare_ivf_shard(master, ids, vecs)
        _write_shard_objects(uri, blob, ids, metas)
        did = _register(
            tenant, dataset, uri, vecs, level=1, parent_shard_id=base_id,
            quantizer_version=1, covered_lsn_lo=lo, covered_lsn_hi=hi,
            consolidated_lsn=hi,
            tombstone_int_ids=(tombstone_ids if d_idx == 0 else None),
        )
        delta_ids_db.append(did)
        all_ids += ids
        all_vecs.append(vecs)
        all_metas += metas
        prev_hi = hi
        next_id += n

    gen = _Generation(
        tenant, dataset, master, all_ids, np.vstack(all_vecs), all_metas
    )
    return gen, base_id, delta_ids_db


# --- flag OFF: byte-identical to today -----------------------------------------


def test_resolve_shard_flag_off_is_single_shard(monkeypatch):
    """Flag OFF: `_resolve_shard` sets only `resolved["shard"]=shards[0]`, no `shards`."""
    _make_generation()
    out = v1q._resolve_shard("t1", "ds", {})
    assert out is not None
    # Single-shard resolution preserved EXACTLY.
    shards = state_mod.list_shards("t1", "ds")
    assert out["shard"] == shards[0]
    assert "shards" not in out, "flag-off must NOT populate the multi-shard set"
    assert "delta_tombstone_int_ids" not in out


def test_resolve_shard_flag_off_no_shard_returns_none(monkeypatch):
    """Flag OFF: a dataset with no shard returns None (byte-identical no-shard signal)."""
    assert v1q._resolve_shard("t1", "empty", {}) is None


def test_watermark_for_shard_unchanged_flag_off():
    """`_watermark_for_shard` stays the single-shard reader (flag-off path)."""
    assert v1q._watermark_for_shard(None) == 0
    assert v1q._watermark_for_shard({}) == 0
    assert v1q._watermark_for_shard({"consolidated_lsn": 42}) == 42


# --- flag ON: multi-shard resolution -------------------------------------------


def test_resolve_shard_flag_on_returns_base_plus_deltas(monkeypatch):
    """Flag ON: `resolved["shards"] = [base] + deltas` ordered by covered_lsn_lo."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, base_id, delta_ids = _make_generation(n_per_delta=(10, 10))
    out = v1q._resolve_shard("t1", "ds", {})
    assert out is not None
    shard_ids = [s["id"] for s in out["shards"]]
    # base first, then deltas in covered_lsn_lo order.
    assert shard_ids[0] == base_id
    assert shard_ids[1:] == delta_ids
    # Ordering is by covered_lsn_lo ascending.
    los = [int(s["covered_lsn_lo"]) for s in out["shards"]]
    assert los == sorted(los)


def test_resolve_shard_flag_on_no_shard_returns_none(monkeypatch):
    """Flag ON: no-shard signal preserved byte-identical (return None)."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    assert v1q._resolve_shard("t1", "empty", {}) is None


def test_resolve_shard_flag_on_stashes_tombstones(monkeypatch):
    """Flag ON: the resolved deltas' tombstone_int_ids are stashed for suppression."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    tomb = [id_to_int64("b3"), id_to_int64("b7")]
    _make_generation(n_per_delta=(10,), tombstone_ids=tomb)
    out = v1q._resolve_shard("t1", "ds", {})
    assert set(out["delta_tombstone_int_ids"]) == set(tomb)


# --- frontier watermark --------------------------------------------------------


def test_frontier_watermark_contiguous(monkeypatch):
    """A contiguous base+delta cover yields the LAST contiguous hi."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    # base hi=100, delta0 [101,110], delta1 [111,120] → frontier 120.
    _make_generation(n_per_delta=(10, 10))
    out = v1q._resolve_shard("t1", "ds", {})
    assert v1q._frontier_watermark(out) == 120


def test_frontier_watermark_gap_clamps_before_gap(monkeypatch):
    """A GAP (a stale-cache missing delta) clamps to the contiguous max before it."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    # base hi=100, delta0 [101,110], delta1 [160,169] (gap at 111..159).
    # The contiguous frontier stops at delta0's hi=110 — NOT max()=169.
    _make_generation(n_per_delta=(10, 10), contiguous=False)
    out = v1q._resolve_shard("t1", "ds", {})
    wm = v1q._frontier_watermark(out)
    assert wm == 110, f"gap must clamp to the contiguous max (110), got {wm}"


def test_frontier_watermark_base_only(monkeypatch):
    """A base-only generation's frontier is the base's covered_lsn_hi."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    _make_generation(n_per_delta=())
    out = v1q._resolve_shard("t1", "ds", {})
    assert v1q._frontier_watermark(out) == 100


def test_frontier_watermark_legacy_base_anchors_on_consolidated_lsn(monkeypatch):
    """A legacy base (covered_lsn_hi==0) anchors at its consolidated_lsn."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    # Register a base whose covered_lsn_hi is the legacy 0 default but with a real
    # consolidated_lsn — the fallback must use the consolidated_lsn (55).
    state_mod.add_shard(
        "t1", "ds", "memory://idx/legacy.bin", checksum="x", vector_count=1,
        index_type="ivfflat", build_type="consolidate", consolidated_lsn=55,
        quantizer_version=1, level=0, covered_lsn_lo=0, covered_lsn_hi=0,
    )
    out = v1q._resolve_shard("t1", "ds", {})
    assert v1q._frontier_watermark(out) == 55


def test_frontier_watermark_no_shards_is_zero():
    """No resolved shards → watermark 0 (all recall rows qualify)."""
    assert v1q._frontier_watermark(None) == 0
    assert v1q._frontier_watermark({}) == 0
    assert v1q._frontier_watermark({"shards": []}) == 0


# --- loop-search union parity vs a monolithic index ----------------------------


def _ids_from_matches(matches):
    return [m["id"] for m in matches]


def test_search_union_matches_monolithic_ranking(monkeypatch):
    """The base+delta loop-search union == a single monolithic index (ids+ranking).

    Shared frozen quantizer (P0-C) makes the union lossless vs a monolith. We
    over-fetch top_k per shard and let `_merge_recall_and_consolidated` (with no
    recall) sort by L2² + truncate — the result must equal a single index built
    over ALL the vectors.
    """
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, _, _ = _make_generation(n_base=40, n_per_delta=(15, 15), seed=3)

    out = v1q._resolve_shard(gen.tenant, gen.dataset, {})
    top_k = 10
    # Query near a base vector so the answer spans base+deltas plausibly.
    q = gen.all_vecs[0].tolist()
    union_matches, mode = v1q._search_consolidated_shard(
        gen.tenant, gen.dataset, q, top_k, None, None, out
    )
    # Merge with an EMPTY recall set (consolidated-only union over shards) — this
    # is the sort+truncate the read path applies.
    union = v1q._merge_recall_and_consolidated(set(), [], union_matches, top_k)

    # Monolithic oracle: same nprobe (server default), same top_k.
    x = np.array([q], dtype=np.float32)
    sp, _ = v1q._ivf_search_params(gen.mono, None)
    kwargs = {"params": sp} if sp is not None else {}
    d, i = gen.mono.search(x, top_k, **kwargs)
    mono = v1q.map_hits_to_matches(i[0], d[0], gen.mono_sidecar, top_k)

    assert _ids_from_matches(union) == _ids_from_matches(mono), (
        "union ranking must equal the monolithic ranking (P0-C parity)"
    )
    # Scores align too.
    np.testing.assert_allclose(
        [m["score"] for m in union], [m["score"] for m in mono], rtol=1e-5, atol=1e-5
    )
    assert mode in ("hot", "cold")


def test_search_union_concatenates_across_shards(monkeypatch):
    """The union surfaces ids from the base AND from each delta (full fan-out)."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, _, _ = _make_generation(n_base=20, n_per_delta=(20, 20), seed=11)
    out = v1q._resolve_shard(gen.tenant, gen.dataset, {})
    # Large top_k so all three shards contribute.
    matches, _ = v1q._search_consolidated_shard(
        gen.tenant, gen.dataset, gen.all_vecs[0].tolist(), 60, None, None, out
    )
    got = set(_ids_from_matches(matches))
    assert any(i.startswith("b") for i in got), "base ids missing from union"
    assert any(i.startswith("d0_") for i in got), "delta0 ids missing from union"
    assert any(i.startswith("d1_") for i in got), "delta1 ids missing from union"


# --- re-upsert / overlap dedup (delta-wins precedence) -------------------------


def _make_overlap_generation(
    tenant="t1", dataset="ds", seed=4,
    base_uri="memory://idx/base.bin", delta_uri="memory://idx/delta0.bin",
):
    """Build a base + ONE delta that SHARE an id "x" with DIFFERENT vectors.

    The base holds id "x" with vector vA and metadata `{"v": "base"}`; a NEWER
    delta holds the SAME id "x" with vector vB (far from vA) and metadata
    `{"v": "delta"}`. This is the re-upsert scenario: PR-B folded the new copy
    of "x" into the delta but the OLD copy stays in the base, so the concatenated
    union would otherwise return "x" TWICE. A handful of disjoint base/delta
    neighbours fill out the shards. Returns `(_Generation-like dict, vA, vB)`.
    """
    train = _rng_vectors(64, seed=seed + 100)
    master = _master_quantizer(train)

    # Base: "x" with vA + a few disjoint neighbours.
    vA = _rng_vectors(1, seed=seed)[0]
    base_extra_ids = [f"b{i}" for i in range(6)]
    base_extra_vecs = _rng_vectors(6, seed=seed + 1)
    base_ids = ["x"] + base_extra_ids
    base_vecs = np.vstack([vA[None, :], base_extra_vecs])
    base_metas = [{"v": "base"}] + [{"src": "base"} for _ in base_extra_ids]
    base_blob = _build_bare_ivf_shard(master, base_ids, base_vecs)
    _write_shard_objects(base_uri, base_blob, base_ids, base_metas)
    base_id = _register(
        tenant, dataset, base_uri, base_vecs, level=0, parent_shard_id=None,
        quantizer_version=1, covered_lsn_lo=0, covered_lsn_hi=100,
        consolidated_lsn=100,
    )

    # Delta: the NEWER copy of "x" with vB (deliberately far from vA) + neighbours.
    vB = _rng_vectors(1, seed=seed + 50)[0]
    delta_extra_ids = [f"d{i}" for i in range(6)]
    delta_extra_vecs = _rng_vectors(6, seed=seed + 2)
    delta_ids = ["x"] + delta_extra_ids
    delta_vecs = np.vstack([vB[None, :], delta_extra_vecs])
    delta_metas = [{"v": "delta"}] + [{"src": "delta"} for _ in delta_extra_ids]
    delta_blob = _build_bare_ivf_shard(master, delta_ids, delta_vecs)
    _write_shard_objects(delta_uri, delta_blob, delta_ids, delta_metas)
    _register(
        tenant, dataset, delta_uri, delta_vecs, level=1, parent_shard_id=base_id,
        quantizer_version=1, covered_lsn_lo=101, covered_lsn_hi=110,
        consolidated_lsn=110,
    )
    return {"tenant": tenant, "dataset": dataset, "master": master}, vA, vB


def test_reupsert_dedups_to_single_newest_copy(monkeypatch):
    """Re-upsert: base "x"=vA + newer delta "x"=vB → union returns "x" ONCE, delta-wins.

    The OLD base copy and the NEW delta copy of "x" both surface from their own
    shard's top_k; the consolidated-vs-consolidated dedup must collapse them to
    EXACTLY ONE entry, keeping the NEWEST band (the delta). We query AT vA (the
    stale base vector) so a naive concat would rank the stale base copy first —
    proving the dedup keeps the delta's identity (metadata), not the base's.
    """
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, vA, vB = _make_overlap_generation()
    out = v1q._resolve_shard(gen["tenant"], gen["dataset"], {})
    top_k = 10
    matches, mode = v1q._search_consolidated_shard(
        gen["tenant"], gen["dataset"], vA.tolist(), top_k, None, None, out
    )
    ids = _ids_from_matches(matches)
    assert ids.count("x") == 1, f"re-upserted id must appear exactly once, got {ids}"
    # Delta-wins: the surviving copy carries the DELTA's metadata/identity.
    x_match = next(m for m in matches if m["id"] == "x")
    assert x_match["metadata"] == {"v": "delta"}, (
        "newest-band (delta) copy must win the dedup, not the stale base copy"
    )
    assert mode in ("hot", "cold")


def test_reupsert_dedup_through_merge_no_duplicate(monkeypatch):
    """End-to-end through the UNCHANGED merge: no duplicate "x", a real result kept.

    A naive concat would consume two of the top_k slots with the duplicate "x"
    and push a real neighbour out. After delta-wins dedup, the merge's sort +
    truncate yields a duplicate-free top_k.
    """
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, vA, vB = _make_overlap_generation(seed=17)
    out = v1q._resolve_shard(gen["tenant"], gen["dataset"], {})
    top_k = 10
    matches, _ = v1q._search_consolidated_shard(
        gen["tenant"], gen["dataset"], vA.tolist(), top_k, None, None, out
    )
    merged = v1q._merge_recall_and_consolidated(set(), [], matches, top_k)
    ids = _ids_from_matches(merged)
    assert len(ids) == len(set(ids)), f"merged union must have no duplicate ids: {ids}"
    # "x" appears AT MOST once — never the stale-base + new-delta pair. (It may
    # rank past top_k when vB is far from the query; dedup-correctness is the
    # no-duplicate invariant, not that the re-upserted id always survives.)
    assert ids.count("x") <= 1


def test_reupsert_dedup_composes_with_tombstone(monkeypatch):
    """Dedup + delete compose: a tombstoned re-upserted id is suppressed entirely.

    "x" is re-upserted (base vA + delta vB) AND tombstoned in the delta. The
    dedup keeps exactly its newest copy; tombstone suppression then drops that
    one copy — so "x" is ABSENT, with no lingering stale base duplicate.
    """
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    # Build the overlap generation, then re-register the delta WITH a tombstone on
    # "x" so the same delta both supersedes AND deletes the id.
    state_mod._MEM_SHARDS.clear()
    with storage._MEM_LOCK:
        storage._MEM_OBJECTS.clear()
    v1q.cache_clear()
    train = _rng_vectors(64, seed=204)
    master = _master_quantizer(train)
    vA = _rng_vectors(1, seed=4)[0]
    base_ids = ["x"] + [f"b{i}" for i in range(6)]
    base_vecs = np.vstack([vA[None, :], _rng_vectors(6, seed=5)])
    base_metas = [{"v": "base"}] + [{"src": "base"} for _ in range(6)]
    _write_shard_objects(
        "memory://idx/base.bin",
        _build_bare_ivf_shard(master, base_ids, base_vecs), base_ids, base_metas,
    )
    base_id = _register(
        "t1", "ds", "memory://idx/base.bin", base_vecs, level=0, parent_shard_id=None,
        quantizer_version=1, covered_lsn_lo=0, covered_lsn_hi=100, consolidated_lsn=100,
    )
    vB = _rng_vectors(1, seed=54)[0]
    delta_ids = ["x"] + [f"d{i}" for i in range(6)]
    delta_vecs = np.vstack([vB[None, :], _rng_vectors(6, seed=6)])
    delta_metas = [{"v": "delta"}] + [{"src": "delta"} for _ in range(6)]
    _write_shard_objects(
        "memory://idx/delta0.bin",
        _build_bare_ivf_shard(master, delta_ids, delta_vecs), delta_ids, delta_metas,
    )
    _register(
        "t1", "ds", "memory://idx/delta0.bin", delta_vecs, level=1,
        parent_shard_id=base_id, quantizer_version=1, covered_lsn_lo=101,
        covered_lsn_hi=110, consolidated_lsn=110,
        tombstone_int_ids=[id_to_int64("x")],
    )

    out = v1q._resolve_shard("t1", "ds", {})
    top_k = 10
    matches, _ = v1q._search_consolidated_shard("t1", "ds", vA.tolist(), top_k, None, None, out)
    # Dedup already collapsed "x" to one copy inside _search_consolidated_shard.
    assert _ids_from_matches(matches).count("x") <= 1
    suppress = v1q._tombstone_suppress_ids(out)
    assert "x" in suppress
    merged = v1q._merge_recall_and_consolidated(suppress, [], matches, top_k)
    assert "x" not in _ids_from_matches(merged), (
        "a re-upserted-then-deleted id must be absent (no stale base duplicate)"
    )


# --- tombstone suppression (cold-vs-cold delete) -------------------------------


def test_delta_tombstone_suppresses_cold_id(monkeypatch):
    """A delta's tombstone_int_ids suppresses the matching cold (base) id.

    Maps the int64 tombstones back to string ids via the per-shard sidecars and
    extends `recall_suppress_ids`. The merge then drops that cold id.
    """
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    # Tombstone base id "b0" (which is the query target, so it WOULD otherwise be
    # the top hit) via a delta carrying its int64.
    tomb = [id_to_int64("b0")]
    gen, _, _ = _make_generation(n_base=30, n_per_delta=(10,), seed=5, tombstone_ids=tomb)
    out = v1q._resolve_shard(gen.tenant, gen.dataset, {})
    top_k = 10
    q = gen.all_vecs[0].tolist()  # b0's vector
    matches, _ = v1q._search_consolidated_shard(
        gen.tenant, gen.dataset, q, top_k, None, None, out
    )
    # Build the suppression set from the resolved deltas' tombstones (the caller
    # logic PR-C adds to run_query); exercise the helper directly here.
    suppress = v1q._tombstone_suppress_ids(out)
    assert "b0" in suppress, "the int64 tombstone must map back to 'b0' via sidecars"

    merged = v1q._merge_recall_and_consolidated(suppress, [], matches, top_k)
    assert "b0" not in _ids_from_matches(merged), "tombstoned cold id must be suppressed"
    # A non-tombstoned neighbour still surfaces.
    assert len(merged) > 0


# --- unreadable frontier shard → 503 (run_query) -------------------------------


def test_unreadable_frontier_shard_raises_consolidated_error(monkeypatch):
    """A delta whose .bin is missing raises (the consolidated-error path) — no silent narrowing."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    gen, _, delta_ids = _make_generation(n_base=20, n_per_delta=(10,), seed=9)
    # Delete the delta's .bin (and its cache) to simulate an unreadable frontier shard.
    with storage._MEM_LOCK:
        del storage._MEM_OBJECTS["memory://idx/delta0.bin"]
    v1q.cache_clear()
    out = v1q._resolve_shard(gen.tenant, gen.dataset, {})
    with pytest.raises(Exception):
        v1q._search_consolidated_shard(
            gen.tenant, gen.dataset, gen.all_vecs[0].tolist(), 10, None, None, out
        )


def test_run_query_unreadable_frontier_shard_returns_503(monkeypatch):
    """End-to-end: an unreadable frontier delta → run_query 503 (recall on + delta on)."""
    from fastapi.responses import JSONResponse

    monkeypatch.setenv("RB_DELTA_TIER", "1")
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    gen, _, _ = _make_generation(n_base=20, n_per_delta=(10,), seed=13)
    with storage._MEM_LOCK:
        del storage._MEM_OBJECTS["memory://idx/delta0.bin"]
    v1q.cache_clear()
    # Recall succeeds cleanly; the consolidated (frontier) failure must still 503.
    monkeypatch.setattr(v1q, "recall_search", lambda *a, **k: (set(), []))

    parsed = v1q._ParsedQuery("ds", gen.all_vecs[0].tolist(), 10, None, {})
    out = v1q.run_query(gen.tenant, parsed)
    assert isinstance(out, JSONResponse)
    assert out.status_code == 503


# --- run_query end-to-end union (flag on) --------------------------------------


def test_run_query_delta_union_end_to_end(monkeypatch):
    """run_query (recall on + delta on) unions base+deltas and merges with recall."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    gen, _, _ = _make_generation(n_base=30, n_per_delta=(10, 10), seed=21)

    captured = {}

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        captured["watermark"] = watermark
        # A fresh recall row above the frontier + an override of a cold id.
        return (
            {"fresh"},
            [{"id": "fresh", "score": 0.0, "metadata": {"t": "hot"}, "deleted": False}],
        )

    monkeypatch.setattr(v1q, "recall_search", _fake_recall)
    parsed = v1q._ParsedQuery("ds", gen.all_vecs[0].tolist(), 10, None, {})
    out = v1q.run_query(gen.tenant, parsed)
    assert not hasattr(out, "status_code"), out
    # Recall scoped to the contiguous frontier (base 100 + 2 deltas → 120).
    assert captured["watermark"] == 120
    ids = _ids_from_matches(out["matches"])
    assert "fresh" in ids, "recall row must be unioned in"
    assert out["mode"] in ("hot", "cold")


def test_run_query_delta_union_tombstone_suppresses_cold(monkeypatch):
    """run_query: a delta tombstone suppresses the cold id end-to-end."""
    monkeypatch.setenv("RB_DELTA_TIER", "1")
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    tomb = [id_to_int64("b0")]
    gen, _, _ = _make_generation(n_base=30, n_per_delta=(10,), seed=33, tombstone_ids=tomb)
    monkeypatch.setattr(v1q, "recall_search", lambda *a, **k: (set(), []))

    parsed = v1q._ParsedQuery("ds", gen.all_vecs[0].tolist(), 10, None, {})
    out = v1q.run_query(gen.tenant, parsed)
    assert not hasattr(out, "status_code"), out
    assert "b0" not in _ids_from_matches(out["matches"]), (
        "the delta cold-vs-cold tombstone must suppress b0"
    )
