from __future__ import annotations

"""Delta-tier generation / watermark logic for the state adapter.

Extracted from `adapters.state.state` (behaviour-preserving). This module groups
a newest-first shard list into base+delta generations (migration 009) and derives
the per-dataset watermarks the recall→consolidated union and the sweep/grace-trim
depend on:

  * `_generations` — group shards into generations (newest first);
  * `live_generation` — the current `{"base", "deltas"}` generation;
  * `dataset_watermark` — highest `consolidated_lsn` over the live generation;
  * `grace_watermark` — the trim boundary (oldest still-live generation's frontier);
  * `superseded_shards` — shards outside the newest `keep` generations (sweepable).

These are pure functions of the shard rows; they read the catalog through
`_state.list_shards` at CALL time. `list_shards` currently lives in
`adapters.state.state`; referencing it via `_state.list_shards` (rather than a
direct import) keeps the seam stable when the catalog is later extracted too, and
keeps `monkeypatch`/`reload` of the state module honoured (see `pooling.py`).
`_generations` is re-exported from `adapters.state.state` so `v1_query`'s
`_state._generations` access keeps resolving.
"""

from typing import List, Optional

# `list_shards` is reached through the state module at call time so the seam
# stays stable across reloads/monkeypatches and across a future catalog split.
from adapters.state._lazy_state import state as _state  # lazy proxy: resolves the facade at call time (breaks the import cycle)


def _shard_level(shard: dict) -> int:
    return int(shard.get("level", 0) or 0)


def _shard_frontier(shard: dict) -> int:
    """The recall-LSN this shard covers up to (its `covered_lsn_hi`).

    Falls back to `consolidated_lsn` for legacy/base-only rows written before
    migration 009 (where `covered_lsn_hi` is its `0` default but the shard still
    carries a real watermark in `consolidated_lsn`).
    """
    hi = int(shard.get("covered_lsn_hi", 0) or 0)
    return hi if hi > 0 else int(shard.get("consolidated_lsn", 0) or 0)


def _generations(shards: List[dict]) -> List[List[dict]]:
    """Group a newest-first shard list into generations, newest generation first.

    A generation = one base (`level=0`) plus the deltas (`level=1`) whose
    `parent_shard_id` is that base AND whose `quantizer_version` matches it. Each
    base defines its own generation in `list_shards` order (newest base first);
    deltas attach to their base. Orphan deltas (parent already swept, or a
    quantizer-version mismatch) attach to NO generation, so they fall out of every
    live set and are swept — the desired behaviour.

    For a base-only dataset every shard is its own single-shard generation in
    newest-first order, so `_generations(shards)[:keep]` flattens back to
    `shards[:keep]` — preserving the pre-delta sweep/grace semantics exactly.
    """
    deltas_by_parent: dict = {}
    for s in shards:
        if _shard_level(s) != 0:
            deltas_by_parent.setdefault(s.get("parent_shard_id"), []).append(s)
    generations: List[List[dict]] = []
    for base in shards:
        if _shard_level(base) != 0:
            continue
        kids = [
            d for d in deltas_by_parent.get(base["id"], [])
            if int(d.get("quantizer_version", 0) or 0)
            == int(base.get("quantizer_version", 0) or 0)
        ]
        generations.append([base, *kids])
    return generations


def live_generation(tenant_id: str, dataset_name: str) -> Optional[dict]:
    """Return the CURRENT generation as `{"base": row, "deltas": [rows]}` or None.

    The base is the newest `level=0` shard; the deltas are its same-version
    children, ordered by `covered_lsn_lo` (oldest LSN band first) so the query
    path can verify a contiguous frontier. This REPLACES the `shards[0]` notion
    of "the current shard" for the delta-tier read/union path.
    """
    shards = _state.list_shards(tenant_id, dataset_name)
    generations = _generations(shards)
    if not generations:
        return None
    base, *deltas = generations[0]
    deltas.sort(key=lambda s: (int(s.get("covered_lsn_lo", 0) or 0), s["id"]))
    return {"base": base, "deltas": deltas}


def dataset_watermark(tenant_id: str, dataset_name: str) -> int:
    """Highest `consolidated_lsn` over the live generation (0 if no shard).

    The per-dataset high-water mark for carry-forward and recall-cap decisions.
    `get_latest_shard()["consolidated_lsn"]` is no longer safe to read directly:
    with deltas, `shards[0]` may be a delta and the max-over-generation is the
    correct frontier.
    """
    gen = live_generation(tenant_id, dataset_name)
    if not gen:
        return 0
    rows = [gen["base"], *gen["deltas"]]
    return max(int(r.get("consolidated_lsn", 0) or 0) for r in rows)


def grace_watermark(tenant_id: str, dataset_name: str, keep: int = 2) -> int:
    """The trim boundary: the oldest STILL-LIVE generation's frontier (I4).

    `recall_trim` may delete recall rows with `lsn <= grace_watermark`. The
    oldest live generation is the one a slow in-flight query might still resolve;
    its frontier is the highest recall LSN already folded into its shards, so
    rows at or below it are safely in the cold tier for that query. Returns 0
    until at least `keep` generations exist (nothing has aged into the grace
    window — e.g. the first consolidation), matching the legacy
    `list_shards()[1].consolidated_lsn` behaviour for base-only datasets.
    """
    keep = max(1, keep)
    generations = _generations(_state.list_shards(tenant_id, dataset_name))
    if len(generations) < keep:
        return 0
    oldest_live = generations[keep - 1]
    return max(_shard_frontier(s) for s in oldest_live)


def superseded_shards(
    tenant_id: str, dataset_name: str, keep: int = 2
) -> List[dict]:
    """Return shards eligible for sweeping — everything outside the newest `keep` generations.

    LIVENESS IS BY GENERATION MEMBERSHIP, not list position. A generation is a
    base plus its same-version deltas (`_generations`); the newest `keep`
    generations are retained (current + grace), everything else is swept. A live
    delta is NEVER swept even though it sorts to the HEAD of `list_shards` by
    `created_at` (deltas are written after their base) — the old
    `list_shards[keep:]` would have GC'd the live base out from under it.

    `keep=2` retains the current generation plus the immediately prior one (the
    grace buffer for an in-flight query that resolved the previous generation).
    For a base-only dataset this is identical to the legacy newest-`keep`-shards
    behaviour (each base is a single-shard generation).
    """
    keep = max(1, keep)
    shards = _state.list_shards(tenant_id, dataset_name)
    generations = _generations(shards)
    live_ids = {s["id"] for gen in generations[:keep] for s in gen}
    return [s for s in shards if s["id"] not in live_ids]
