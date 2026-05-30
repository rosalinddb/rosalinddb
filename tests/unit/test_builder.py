import os
import faiss  # type: ignore
import numpy as np
import pytest

from services.index_builder.run import build_ivfflat, build_flat, _choose_nlist


@pytest.mark.skipif(
    os.environ.get("PY_MAJOR", str(os.sys.version_info.major)) == "3" and os.sys.version_info.minor >= 13,
    reason="faiss wheels may be unavailable on Python 3.13",
)
def test_build_ivfflat_serializes():
    """The recall touch-up: a non-tiny first ingest builds a serializable
    IVFFlat index (IVF coarse quantizer + raw, uncompressed float32 vectors)."""
    os.environ["DIMENSION"] = "8"
    vecs = np.random.rand(256, 8).astype(np.float32)
    blob = build_ivfflat(vecs)
    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) > 0


def test_build_ivfflat_produces_an_ivf_index_not_pq():
    """The serialized index must be an IVFFlat — an IVF index (so the
    query-time `nprobe` knob still works) that stores RAW vectors (no PQ)."""
    vecs = np.random.rand(512, 8).astype(np.float32)
    ids = np.arange(512, dtype=np.int64)
    blob = build_ivfflat(vecs, ids)
    index = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))
    inner = faiss.extract_index_ivf(index)  # raises if not an IVF index
    assert inner.nlist >= 4
    # IVFFlat stores raw float32 vectors — it is NOT an IVFPQ index.
    # `extract_index_ivf` returns a generic `IndexIVF` proxy; `downcast_index`
    # recovers the concrete subclass for the type check.
    concrete = faiss.downcast_index(faiss.extract_index_ivf(index))
    assert isinstance(concrete, faiss.IndexIVFFlat)
    assert not isinstance(concrete, faiss.IndexIVFPQ)
    assert index.ntotal == 512


def test_choose_nlist_follows_sqrt_rule():
    """`nlist` is sized by the FAISS `4*sqrt(N)` rule of thumb, clamped so
    k-means can always train (`<= N//8`) and capped by `IVF_NLIST`.

    Informed by OpenData Vector's ~100-vectors-per-cluster sizing target."""
    # 100k vectors → 4*sqrt(100000) ≈ 1264, well under N//8 (12500) and the
    # 4096 ceiling → ~80 vectors per cell, close to the ~100 target.
    assert _choose_nlist(100_000) == 1264
    # tiny batch → clamped to N//8 so every posting stays non-degenerate.
    assert _choose_nlist(64) == 8
    # the result is always >= 1.
    assert _choose_nlist(1) == 1


def test_choose_nlist_respects_ivf_nlist_ceiling(monkeypatch):
    """`IVF_NLIST` is a hard ceiling on the cell count."""
    monkeypatch.setenv("IVF_NLIST", "16")
    # 100k vectors would otherwise yield ~1264 cells; the ceiling caps it.
    assert _choose_nlist(100_000) == 16


def test_build_flat_serializes_small_batch():
    """A batch below the IVF training floor still builds via the exact flat
    path — the tiny-dataset fallback."""
    vecs = np.random.rand(100, 8).astype(np.float32)
    ids = np.arange(100, dtype=np.int64)
    blob = build_flat(vecs, ids)
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
