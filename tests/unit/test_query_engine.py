"""Unit coverage for the query-engine fixes (`feat/engine-fix`).

Hermetic — no Docker, no network. Exercises three pieces of
`services.query_api.v1_query` directly:

  - nprobe tunability: `nprobe` is tunable via `RB_QUERY_NPROBE`, resolved into
    a per-search `SearchParametersIVF` (no shared-index mutation), clamped to
    `MAX_NPROBE`, and a harmless no-op on a flat index.
  - shard cache correctness: the in-memory shard cache returns the *same* index object
    across queries (a hit does not re-deserialise), is byte-budgeted by
    `RB_SHARD_CACHE_BYTES`, and a swept shard is evicted.
"""
from __future__ import annotations

import faiss  # type: ignore
import numpy as np

import services.query_api.v1_query as v1q


def _ivf_index(n=512, dim=16, nlist=8):
    """Build a small trained IVF index for nprobe assertions."""
    vecs = np.random.rand(n, dim).astype(np.float32)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(vecs)
    index.add(vecs)
    return index


# --- nprobe tunability ----------------------------------------------------


def test_query_nprobe_reads_env(monkeypatch):
    monkeypatch.setenv("RB_QUERY_NPROBE", "32")
    assert v1q.query_nprobe() == 32


def test_query_nprobe_floored_at_one(monkeypatch):
    monkeypatch.setenv("RB_QUERY_NPROBE", "0")
    assert v1q.query_nprobe() == 1


def test_ivf_search_params_resolves_default(monkeypatch):
    """`_ivf_search_params` returns per-search params, NOT a mutated index."""
    monkeypatch.setenv("RB_QUERY_NPROBE", "24")
    index = _ivf_index()
    assert index.nprobe == 1  # FAISS default
    params, applied = v1q._ivf_search_params(index)
    assert applied == 24
    assert isinstance(params, faiss.SearchParametersIVF)
    assert params.nprobe == 24
    # The shared index object is left untouched — no cross-query race.
    assert index.nprobe == 1


def test_ivf_search_params_override_wins(monkeypatch):
    """A per-request override beats the server default."""
    monkeypatch.setenv("RB_QUERY_NPROBE", "24")
    index = _ivf_index()
    params, applied = v1q._ivf_search_params(index, override=7)
    assert applied == 7
    assert params.nprobe == 7


def test_ivf_search_params_clamps_to_max(monkeypatch):
    """A huge per-query nprobe is clamped to MAX_NPROBE, never unbounded."""
    index = _ivf_index()
    params, applied = v1q._ivf_search_params(index, override=10_000_000)
    assert applied == v1q.MAX_NPROBE
    assert params.nprobe == v1q.MAX_NPROBE


def test_query_nprobe_clamped_to_max(monkeypatch):
    """The server-default nprobe is also clamped to MAX_NPROBE."""
    monkeypatch.setenv("RB_QUERY_NPROBE", str(v1q.MAX_NPROBE * 100))
    assert v1q.query_nprobe() == v1q.MAX_NPROBE


def test_ivf_search_params_noop_on_flat_index(monkeypatch):
    """A flat index has no IVF cells — params is None, no crash."""
    monkeypatch.setenv("RB_QUERY_NPROBE", "16")
    flat = faiss.IndexFlatL2(8)
    params, applied = v1q._ivf_search_params(flat)
    assert params is None
    assert applied == 16  # returns the value, no crash


# --- in-memory shard cache ------------------------------------------------


def test_cache_returns_same_object(monkeypatch):
    """A cache hit returns the *identical* index object — no re-deserialise."""
    v1q.cache_clear()
    index = _ivf_index()
    sidecar = {"1": {"id": "a", "metadata": {}}}
    v1q._cache_put("shard-1", index, sidecar)
    got_index, got_sidecar = v1q._cache_get("shard-1")
    assert got_index is index
    assert got_sidecar is sidecar


def test_cache_miss_is_none():
    v1q.cache_clear()
    assert v1q._cache_get("absent") is None


class _SizedSidecar(dict):
    """A sidecar dict whose measured footprint is a fixed `nbytes`.

    `_sidecar_nbytes` JSON-serialises the sidecar; padding the dict with a
    string of the right length gives a predictable, controllable footprint
    for byte-budget assertions without allocating real megabytes.
    """

    def __new__(cls, nbytes):
        return super().__new__(cls)

    def __init__(self, nbytes):
        super().__init__()
        # `json.dumps({"_pad": "x"*N})` is N + 12 chars; size the pad so the
        # serialised length is ~nbytes.
        self["_pad"] = "x" * max(0, nbytes - 12)


def test_cache_byte_budget_evicts_when_over(monkeypatch):
    """The cache is byte-budgeted — inserting past RB_SHARD_CACHE_BYTES evicts
    the LRU entry until the running total fits the budget."""
    # Budget fits two ~1 KiB entries but not three.
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_BYTES", 2400)
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_SIZE", 0)  # count cap disabled
    v1q.cache_clear()
    v1q._cache_put("s1", object(), _SizedSidecar(1000))
    v1q._cache_put("s2", object(), _SizedSidecar(1000))
    assert v1q._SHARD_CACHE_BYTES_USED <= 2400
    v1q._cache_put("s3", object(), _SizedSidecar(1000))  # over budget → evicts s1
    assert v1q._cache_get("s1") is None
    assert v1q._cache_get("s2") is not None
    assert v1q._cache_get("s3") is not None
    assert v1q._SHARD_CACHE_BYTES_USED <= 2400


def test_cache_oversized_entry_admitted_then_evicted(monkeypatch):
    """An entry larger than the whole budget is admitted then immediately
    evicted — usable for the current query, never retained, no infinite loop."""
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_BYTES", 500)
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_SIZE", 0)
    v1q.cache_clear()
    v1q._cache_put("huge", object(), _SizedSidecar(5000))  # alone > budget
    assert v1q._cache_get("huge") is None  # not retained
    assert v1q._SHARD_CACHE_BYTES_USED == 0


def test_cache_byte_budget_evicts_lru_first(monkeypatch):
    """Touching an entry makes it most-recent so it survives the next evict."""
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_BYTES", 2400)
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_SIZE", 0)
    v1q.cache_clear()
    v1q._cache_put("s1", object(), _SizedSidecar(1000))
    v1q._cache_put("s2", object(), _SizedSidecar(1000))
    v1q._cache_get("s1")  # s1 is now most-recent; s2 is LRU
    v1q._cache_put("s3", object(), _SizedSidecar(1000))  # evicts s2, not s1
    assert v1q._cache_get("s1") is not None
    assert v1q._cache_get("s2") is None


def test_cache_secondary_count_cap(monkeypatch):
    """The optional secondary count cap still bounds entry count when set."""
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_BYTES", 1 << 40)  # effectively unbounded
    monkeypatch.setattr(v1q, "RB_SHARD_CACHE_SIZE", 2)
    v1q.cache_clear()
    v1q._cache_put("s1", object(), {})
    v1q._cache_put("s2", object(), {})
    v1q._cache_put("s3", object(), {})  # over count cap → evicts the LRU (s1)
    assert v1q._cache_get("s1") is None
    assert v1q._cache_get("s2") is not None
    assert v1q._cache_get("s3") is not None


def test_cache_clear_resets_byte_total(monkeypatch):
    """cache_clear resets the running byte total to 0."""
    v1q.cache_clear()
    v1q._cache_put("s1", object(), _SizedSidecar(1000))
    assert v1q._SHARD_CACHE_BYTES_USED > 0
    v1q.cache_clear()
    assert v1q._SHARD_CACHE_BYTES_USED == 0


def test_evict_shard_decrements_byte_total(monkeypatch):
    """evict_shard subtracts the evicted entry's footprint from the total."""
    v1q.cache_clear()
    v1q._cache_put("s1", object(), _SizedSidecar(1000))
    used = v1q._SHARD_CACHE_BYTES_USED
    assert used > 0
    assert v1q.evict_shard("s1") is True
    assert v1q._SHARD_CACHE_BYTES_USED == 0


def test_evict_shard_drops_entry():
    """A swept shard's cache entry is dropped so it is never served."""
    v1q.cache_clear()
    v1q._cache_put("doomed", object(), {})
    assert v1q.evict_shard("doomed") is True
    assert v1q._cache_get("doomed") is None
    # Idempotent — evicting an absent shard is a harmless no-op.
    assert v1q.evict_shard("doomed") is False
