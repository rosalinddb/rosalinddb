"""Pure-unit parity tests for the embedded numpy recall memtable.

The all-in-one (single-process, no-docker) deployment routes the recall tier to
an in-process numpy "memtable" (`adapters.recall.memtable`) instead of pgvector,
selected when `RB_RECALL` is on AND no `RB_RECALL_DSN` is set. These tests pin
the memtable's semantics to the pgvector recall path EXACTLY (upsert / search /
delete / snapshot / trim / list / get), so the union/consolidation code that
calls the recall_* interface behaves identically against either backend.

All hermetic: numpy + stdlib only, no Docker, no pgvector. They drive the
memtable functions directly with `RB_RECALL_BACKEND=memory`.
"""
from __future__ import annotations

import importlib
import threading

import numpy as np
import pytest


@pytest.fixture
def mem(monkeypatch):
    """Embedded-recall memtable with a clean store and memory backend selected."""
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_BACKEND", "memory")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    import adapters.recall as recall_pkg
    importlib.reload(recall_pkg)
    from adapters.recall import memtable
    importlib.reload(memtable)
    memtable._reset()
    yield memtable
    memtable._reset()


# --- config seam (read-fresh) ---------------------------------------------


def test_recall_backend_read_fresh(monkeypatch):
    """`config.recall_backend()` reads RB_RECALL_BACKEND fresh (default 'auto')."""
    import adapters.config as config

    monkeypatch.delenv("RB_RECALL_BACKEND", raising=False)
    assert config.recall_backend() == "auto"
    monkeypatch.setenv("RB_RECALL_BACKEND", "memory")
    assert config.recall_backend() == "memory"  # no reload needed
    monkeypatch.setenv("RB_RECALL_BACKEND", "pgvector")
    assert config.recall_backend() == "pgvector"


def test_recall_backend_normalized(monkeypatch):
    """`recall_backend()` strips + lower-cases so casing/whitespace can't break
    the `_use_memory_backend()` token compare."""
    import adapters.config as config
    import adapters.recall as recall_pkg

    monkeypatch.setenv("RB_RECALL_BACKEND", "  MEMORY ")
    assert config.recall_backend() == "memory"
    monkeypatch.setenv("RB_RECALL_BACKEND", "Auto")
    assert config.recall_backend() == "auto"

    # And the normalisation actually flows through to backend selection: a
    # mixed-case " Memory " still routes to the embedded memtable.
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_BACKEND", " Memory ")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(recall_pkg)
    assert recall_pkg._use_memory_backend() is True


# --- enabled gate (headline embedded-mode change) -------------------------


def test_enabled_without_dsn(monkeypatch):
    """RB_RECALL on + backend=memory (or auto) + no DSN => recall_enabled() True."""
    import adapters.recall as recall_pkg

    # memory backend explicitly, no DSN -> enabled
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_BACKEND", "memory")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(recall_pkg)
    assert recall_pkg.recall_enabled() is True

    # auto backend, recall on, no DSN -> embedded selected -> enabled
    monkeypatch.setenv("RB_RECALL_BACKEND", "auto")
    importlib.reload(recall_pkg)
    assert recall_pkg.recall_enabled() is True

    # RB_RECALL off -> disabled regardless of backend
    monkeypatch.delenv("RB_RECALL", raising=False)
    importlib.reload(recall_pkg)
    assert recall_pkg.recall_enabled() is False

    # pgvector mode: DSN set, backend resolves to pgvector, still requires DSN.
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_BACKEND", "auto")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    importlib.reload(recall_pkg)
    assert recall_pkg.recall_enabled() is True  # on (pgvector path)
    assert recall_pkg._use_memory_backend() is False  # but NOT the memtable

    # explicit pgvector backend with no DSN -> NOT enabled (no store to write to)
    monkeypatch.setenv("RB_RECALL_BACKEND", "pgvector")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(recall_pkg)
    assert recall_pkg._use_memory_backend() is False
    assert recall_pkg.recall_enabled() is False


# --- upsert + search (read-your-writes) -----------------------------------


def _dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sum((a - b) ** 2))


def test_upsert_then_search_ryw(mem):
    n = mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [1.0, 0.0], "metadata": {}}])
    assert n == 1
    suppress, matches = mem.recall_search("t", "d", [1.0, 0.0], top_k=10, watermark=0)
    assert "x" in suppress
    assert len(matches) == 1
    assert matches[0]["id"] == "x"
    assert matches[0]["deleted"] is False
    # score == exact L2-squared distance
    assert matches[0]["score"] == pytest.approx(_dist([1.0, 0.0], [1.0, 0.0]))


def test_search_score_is_l2_squared_not_sqrt(mem):
    """The recall score MUST be L2-SQUARED (not plain L2/sqrt) so it ranks on the
    same scale as the FAISS L2-squared distances the cold tier produces.

    A distance of 2 along ONE axis distinguishes the two: squared = 4.0, while a
    sqrt impl would return 2.0. Pin the SQUARED value so a plain-L2 regression
    fails green here (this is the load-bearing ranking property of the union).
    """
    mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [0.0, 0.0], "metadata": {}}])
    suppress, matches = mem.recall_search("t", "d", [2.0, 0.0], top_k=10, watermark=0)
    assert len(matches) == 1
    # squared distance = (2-0)^2 + 0 = 4.0; a sqrt impl would give 2.0.
    assert matches[0]["score"] == pytest.approx(4.0)
    assert matches[0]["score"] != pytest.approx(2.0)
    # And a two-axis case where squared (8) and sqrt (~2.83) also differ.
    mem.recall_upsert_vectors("t", "d2", [{"id": "y", "values": [0.0, 0.0], "metadata": {}}])
    _, m2 = mem.recall_search("t", "d2", [2.0, 2.0], top_k=10, watermark=0)
    assert m2[0]["score"] == pytest.approx(8.0)


def test_search_returns_metadata_copy_not_alias(mem):
    """`recall_search` / `recall_list_rows` return DEFENSIVE COPIES of the stored
    metadata (the pgvector path returns a fresh JSONB decode per fetch). A caller
    mutating a returned dict must NOT corrupt the live memtable row."""
    mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [1.0, 0.0], "metadata": {"k": "v"}}])

    _, matches = mem.recall_search("t", "d", [1.0, 0.0], top_k=10, watermark=0)
    matches[0]["metadata"]["k"] = "MUTATED"
    matches[0]["metadata"]["injected"] = True
    # a fresh read is unaffected (the store kept the original)
    _, again = mem.recall_search("t", "d", [1.0, 0.0], top_k=10, watermark=0)
    assert again[0]["metadata"] == {"k": "v"}

    live_rows, _ = mem.recall_list_rows("t", "d", watermark=0)
    live_rows[0]["metadata"]["k"] = "MUTATED2"
    again_rows, _ = mem.recall_list_rows("t", "d", watermark=0)
    assert again_rows[0]["metadata"] == {"k": "v"}


def test_upsert_last_write_wins(mem):
    # Two separate upserts of the same id; plus an intra-batch duplicate.
    mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [1.0, 1.0], "metadata": {"v": 1}}])
    mem.recall_upsert_vectors(
        "t", "d",
        [
            {"id": "x", "values": [2.0, 2.0], "metadata": {"v": 2}},
            {"id": "x", "values": [3.0, 3.0], "metadata": {"v": 3}},  # intra-batch dup
        ],
    )
    max_lsn, rows = mem.recall_snapshot_for_consolidation("t", "d")
    # exactly one surviving row for id 'x' with the LAST values/metadata
    xrows = [r for r in rows if r["id"] == "x"]
    assert len(xrows) == 1
    assert xrows[0]["values"] == [3.0, 3.0]
    assert xrows[0]["metadata"] == {"v": 3}
    # strictly higher lsn than the first upsert (which got lsn 1)
    assert xrows[0]["lsn"] > 1


def test_search_single_snapshot_suppress_superset_matches(mem):
    # one row passes the filter & ranks in top_k, one fails the filter, one ranks
    # past top_k. ALL ids are in suppress; only the qualifying one is a match.
    mem.recall_upsert_vectors("t", "d", [
        {"id": "near_pass", "values": [0.0, 0.0], "metadata": {"k": "a"}},
        {"id": "near_fail", "values": [0.1, 0.0], "metadata": {"k": "b"}},
        {"id": "far_pass", "values": [9.0, 9.0], "metadata": {"k": "a"}},
    ])
    suppress, matches = mem.recall_search(
        "t", "d", [0.0, 0.0], top_k=1, watermark=0, flt={"k": "a"}
    )
    match_ids = {m["id"] for m in matches}
    assert match_ids.issubset(suppress)  # superset always
    assert suppress == {"near_pass", "near_fail", "far_pass"}
    assert match_ids == {"near_pass"}  # top_k=1, filter k=a, nearest


def test_search_watermark_scoping(mem):
    mem.recall_upsert_vectors("t", "d", [{"id": "a", "values": [0.0, 0.0], "metadata": {}}])  # lsn 1
    mem.recall_upsert_vectors("t", "d", [{"id": "b", "values": [0.0, 0.0], "metadata": {}}])  # lsn 2
    suppress, matches = mem.recall_search("t", "d", [0.0, 0.0], top_k=10, watermark=1)
    # lsn<=1 excluded from BOTH suppress and matches
    assert suppress == {"b"}
    assert {m["id"] for m in matches} == {"b"}


def test_tombstone_never_matches(mem):
    mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [5.0, 5.0], "metadata": {}}])
    mem.recall_delete_vector("t", "d", "x", dimension=2)
    # query exactly at the zero-vector placeholder: a tombstone sorts FIRST by raw
    # distance, but must never be a match.
    suppress, matches = mem.recall_search("t", "d", [0.0, 0.0], top_k=10, watermark=0)
    assert "x" in suppress
    assert "x" not in {m["id"] for m in matches}
    status, meta = mem.recall_get_vector("t", "d", "x", watermark=0)
    assert status == "tombstone"
    assert meta is None


def test_delete_allocates_fresh_lsn_above_max(mem):
    mem.recall_upsert_vectors("t", "d", [
        {"id": "a", "values": [1.0], "metadata": {}},
        {"id": "b", "values": [2.0], "metadata": {}},
    ])  # lsn 1, 2
    lsn = mem.recall_delete_vector("t", "d", "a", dimension=1)
    assert lsn > 2  # strictly above every prior lsn
    max_lsn, rows = mem.recall_snapshot_for_consolidation("t", "d")
    assert lsn == max(r["lsn"] for r in rows)


def test_snapshot_for_consolidation(mem):
    mem.recall_upsert_vectors("t", "d", [{"id": "a", "values": [1.0], "metadata": {"n": 1}}])
    mem.recall_upsert_vectors("t", "d", [{"id": "b", "values": [2.0], "metadata": {}}])
    mem.recall_delete_vector("t", "d", "b", dimension=1)
    max_lsn, rows = mem.recall_snapshot_for_consolidation("t", "d")
    # ascending by lsn, one row per id, values parsed to list[float]
    lsns = [r["lsn"] for r in rows]
    assert lsns == sorted(lsns)
    ids = {r["id"] for r in rows}
    assert ids == {"a", "b"}
    assert max_lsn == max(lsns)
    assert all(isinstance(r["values"], list) for r in rows)
    assert all(isinstance(v, float) for r in rows for v in r["values"])
    brow = [r for r in rows if r["id"] == "b"][0]
    assert brow["deleted"] is True

    # empty partition -> (0, [])
    em, er = mem.recall_snapshot_for_consolidation("t", "empty")
    assert em == 0
    assert er == []


def test_snapshot_excludes_later_write(mem):
    # A write that commits with a higher lsn AFTER the snapshot bound is excluded.
    mem.recall_upsert_vectors("t", "d", [{"id": "a", "values": [1.0], "metadata": {}}])
    max_lsn, rows = mem.recall_snapshot_for_consolidation("t", "d")
    assert {r["id"] for r in rows} == {"a"}
    # inject a later write
    mem.recall_upsert_vectors("t", "d", [{"id": "later", "values": [2.0], "metadata": {}}])
    # the previously-captured snapshot is unchanged (it was a copy)
    assert {r["id"] for r in rows} == {"a"}
    # a fresh snapshot now includes 'later' with a higher lsn
    m2, r2 = mem.recall_snapshot_for_consolidation("t", "d")
    assert {r["id"] for r in r2} == {"a", "later"}
    assert m2 > max_lsn


def test_trim_by_watermark(mem):
    mem.recall_upsert_vectors("t", "d", [{"id": "a", "values": [1.0], "metadata": {}}])  # lsn 1
    mem.recall_upsert_vectors("t", "d", [{"id": "b", "values": [2.0], "metadata": {}}])  # lsn 2
    mem.recall_upsert_vectors("t", "d", [{"id": "c", "values": [3.0], "metadata": {}}])  # lsn 3
    assert mem.recall_partition_count("t", "d") == 3
    # grace <= 0 -> no-op
    assert mem.recall_trim("t", "d", 0) == 0
    assert mem.recall_partition_count("t", "d") == 3
    # grace > 0 -> deletes rows lsn<=grace and returns count
    deleted = mem.recall_trim("t", "d", 2)
    assert deleted == 2
    assert mem.recall_partition_count("t", "d") == 1


def test_list_rows_single_snapshot(mem):
    mem.recall_upsert_vectors("t", "d", [
        {"id": "live1", "values": [1.0], "metadata": {"a": 1}},
        {"id": "live2", "values": [2.0], "metadata": {}},
    ])
    mem.recall_delete_vector("t", "d", "live2", dimension=1)  # tombstone live2
    mem.recall_upsert_vectors("t", "d", [{"id": "live3", "values": [3.0], "metadata": {}}])
    live_rows, suppress = mem.recall_list_rows("t", "d", watermark=0)
    live_ids = {r["id"] for r in live_rows}
    assert live_ids == {"live1", "live3"}  # tombstone excluded from live
    assert suppress == {"live1", "live2", "live3"}  # covers live + tombstoned
    assert live_ids.issubset(suppress)
    # metadata carried through
    l1 = [r for r in live_rows if r["id"] == "live1"][0]
    assert l1["metadata"] == {"a": 1}


def test_get_vector_tri_state(mem):
    mem.recall_upsert_vectors("t", "d", [{"id": "x", "values": [1.0, 2.0], "metadata": {"m": 1}}])
    status, meta = mem.recall_get_vector("t", "d", "x", watermark=0)
    assert status == "live"
    assert meta == {"m": 1}
    # with embedding
    status, meta, emb = mem.recall_get_vector_with_embedding("t", "d", "x", watermark=0)
    assert status == "live"
    assert emb == [1.0, 2.0]
    # missing id -> (None, None)
    assert mem.recall_get_vector("t", "d", "missing", watermark=0) == (None, None)
    # below-watermark row -> (None, None)
    assert mem.recall_get_vector("t", "d", "x", watermark=99) == (None, None)


def test_idle_partitions(mem):
    import datetime as dt

    # Freshly written partition -> NOT idle.
    mem.recall_upsert_vectors("t", "fresh", [{"id": "a", "values": [1.0], "metadata": {}}])
    assert ("t", "fresh") not in mem.recall_idle_partitions(idle_seconds=60)

    # Backdate a partition's created_at well past the cutoff.
    mem.recall_upsert_vectors("t", "old", [{"id": "a", "values": [1.0], "metadata": {}}])
    old_ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=3600)
    with mem._MemRecall._lock:
        for row in mem._MemRecall._parts[("t", "old")].values():
            row["created_at"] = old_ts
    idle = mem.recall_idle_partitions(idle_seconds=60)
    assert ("t", "old") in idle
    assert ("t", "fresh") not in idle
    # an empty (trimmed) partition is not returned
    mem.recall_trim("t", "old", 99999)
    assert ("t", "old") not in mem.recall_idle_partitions(idle_seconds=60)


def test_thread_safety_concurrent_upsert_and_snapshot(mem):
    errors = []
    n_threads = 8
    per_thread = 50

    def writer(tid):
        try:
            for i in range(per_thread):
                mem.recall_upsert_vectors(
                    "t", "d", [{"id": f"w{tid}-{i}", "values": [float(i)], "metadata": {}}]
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def snapper():
        try:
            for _ in range(per_thread):
                suppress, matches = mem.recall_search("t", "d", [0.0], top_k=5, watermark=0)
                # invariant: suppress is always a superset of match ids
                assert {m["id"] for m in matches}.issubset(suppress)
                mem.recall_snapshot_for_consolidation("t", "d")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    threads += [threading.Thread(target=snapper) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    # lsn strictly monotonic per partition: all distinct, count == writes
    max_lsn, rows = mem.recall_snapshot_for_consolidation("t", "d")
    lsns = [r["lsn"] for r in rows]
    assert len(lsns) == len(set(lsns))  # no duplicate lsns
    assert len(rows) == n_threads * per_thread
