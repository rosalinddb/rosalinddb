"""Integration regression for the single-snapshot recall scans.

Covers BOTH read-side recall scans that had the b1 over-suppression race:
`recall_search` (the query union, task #17) and its twin `recall_list_rows`
(the CRUD list union, task #30).

This is the read-side analogue of `test_visibility_gap_during_consolidation`:
it hammers concurrent cross-watermark re-UPSERTs WHILE running `recall_search`
against a REAL pgvector container, and asserts the suppress/match split is always
internally consistent — `match_ids ⊆ suppress_ids` — and that the union never
black-holes (returns 0 live matches while live ids exist above the watermark).
The `recall_list_rows` test below applies the identical hammer to the list path.

Why it needs a real database (and so lives here, not in the unit suite): the b1
bug is a READ-SIDE MVCC snapshot skew. The old two-scan shape ran two
`cur.execute`s on a default READ COMMITTED connection, each taking its OWN
snapshot; a re-UPSERT that committed BETWEEN the two scans made the SUPPRESS scan
see ids the MATCH scan missed → over-suppression → a transient all-rows-missing
blackout (bench case b1; root cause: bench-lab/analysis/b1-rootcause.md). Only a
real Postgres exhibits the inter-statement snapshot skew; a fake cursor cannot.

The fix is one SELECT over `lsn > W` returning `(id, deleted, score, metadata)`,
split in Python — ONE statement is ONE MVCC snapshot even under READ COMMITTED,
so the skew is structurally impossible. With the OLD two-scan code this test
flakes to a 0-match blackout under load; with the fix it is rock solid.

CI runs the integration suite with Docker — you need not run it locally.
"""
from __future__ import annotations

import threading

import psycopg2
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


pytestmark = pytest.mark.integration


_DIM = 8
_N_IDS = 50          # all ids stay live the whole test (re-upserted, never deleted)
_QUERY_ITERS = 400   # query hammering iterations
_REUPSERT_ITERS = 400


@pytest.fixture(scope="module")
def recall_url():
    """One pgvector container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the recall-snapshot suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _state_on(monkeypatch, recall_url):
    """Reload the state module with the recall tier ON, pointed at the container."""
    import importlib

    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_RECALL", "true")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    state_mod.migrate_recall(force=True)
    return state_mod


def _vec(i: int) -> list[float]:
    """A deterministic, distinct embedding for id `i`."""
    base = float(i % 10)
    return [base + j * 0.01 for j in range(_DIM)]


def _records():
    return [
        {"id": f"id{i}", "values": _vec(i), "metadata": {"n": i}}
        for i in range(_N_IDS)
    ]


def test_concurrent_cross_watermark_reupserts_never_black_hole(monkeypatch, recall_url):
    """Concurrent re-UPSERTs across the watermark never make the union return 0.

    All `_N_IDS` ids are live the whole test (re-upserted, never deleted). A
    background thread re-UPSERTs the full batch in a tight loop (each batch is one
    atomic multi-row UPSERT that lifts every id to a fresh, higher LSN). The main
    thread queries `recall_search` with `watermark=0` (so ALL recall rows qualify)
    in a tight loop. Two invariants must hold on EVERY observation:

      1. `match_ids ⊆ suppress_ids` — the suppress/match split came from ONE
         snapshot, so suppression can never reference an id the match split lacked.
      2. The union is never a black hole: because every id is live and above the
         watermark, `recall_search` must always return SOME live matches (capped
         at top_k). The b1 blackout was exactly `matches == []` while every id was
         live — the failure this test exists to catch.
    """
    state = _state_on(monkeypatch, recall_url)
    recs = _records()

    # Seed once so the very first query has rows.
    state.recall_upsert_vectors("t", "ds", recs)

    stop = threading.Event()
    errors: list[BaseException] = []

    def _reupsert_loop():
        try:
            for _ in range(_REUPSERT_ITERS):
                if stop.is_set():
                    break
                # One atomic multi-row UPSERT lifting EVERY id to a fresh LSN —
                # the exact write shape that drove the b1 blackout.
                state.recall_upsert_vectors("t", "ds", recs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer = threading.Thread(target=_reupsert_loop, daemon=True)
    writer.start()
    try:
        top_k = 10
        blackouts = 0
        leaks = 0
        for _ in range(_QUERY_ITERS):
            suppress_ids, matches = state.recall_search(
                "t", "ds", _vec(0), top_k=top_k, watermark=0
            )
            match_ids = {m["id"] for m in matches}
            # Invariant 1: the split is internally consistent.
            if not match_ids <= suppress_ids:
                leaks += 1
            # Invariant 2: no black hole — live ids exist, so matches must too.
            if len(matches) == 0:
                blackouts += 1
            # Matches are capped at top_k and are all live.
            assert len(matches) <= top_k
            assert all(m["deleted"] is False for m in matches)
        assert leaks == 0, f"{leaks} observations had match_ids ⊄ suppress_ids"
        assert blackouts == 0, (
            f"{blackouts}/{_QUERY_ITERS} queries returned 0 matches while every "
            "id was live above the watermark — the b1 over-suppression blackout"
        )
    finally:
        stop.set()
        writer.join(timeout=30)
    assert not errors, f"re-upsert loop raised: {errors[0]!r}"


def test_list_rows_concurrent_reupserts_never_drop_a_live_id(monkeypatch, recall_url):
    """`recall_list_rows` under concurrent re-UPSERTs never loses a live id.

    The b1 TWIN on the LIST path (task #30). `recall_list_rows` used to run TWO
    READ COMMITTED scans — a `NOT deleted` LIVE scan then a full SUPPRESS scan — so
    a re-UPSERT that committed BETWEEN them made the SUPPRESS scan return an id the
    LIVE scan never saw → `suppress_ids ⊋ live_ids`. At the list call site
    (`services/source_registry/main.py`) the cold copy of that id is DROPPED (it is
    in `suppress_ids`) AND no recall live row is appended (the live scan missed it),
    so the record TRANSIENTLY VANISHES from the list.

    Every one of `_N_IDS` ids stays live the whole test (re-upserted, never
    deleted). A background thread re-UPSERTs the full batch in a tight loop; the
    main thread calls `recall_list_rows(watermark=0)` (so ALL recall rows qualify)
    in a tight loop. Two invariants must hold on EVERY observation:

      1. `live_ids ⊆ suppress_ids` — the split came from ONE snapshot.
      2. No id vanishes: because every id is live above the watermark, the union
         the call site computes (cold copies minus `suppress_ids`, plus the recall
         `live` rows) must contain ALL `_N_IDS` ids on every list. With the OLD
         two-scan code an id transiently appears in `suppress_ids` but NOT in
         `live`, so it is dropped from cold AND not re-added → the record is gone.

    The fix is ONE SELECT over `lsn > W` returning `(id, deleted, metadata)`, split
    in Python — one statement is one MVCC snapshot, so the skew is impossible. With
    the OLD code this flakes (an id missing from the list); with the fix it is solid.
    """
    state = _state_on(monkeypatch, recall_url)
    recs = _records()
    all_ids = {f"id{i}" for i in range(_N_IDS)}

    # Seed once so the very first list has rows.
    state.recall_upsert_vectors("tl", "dsl", recs)

    stop = threading.Event()
    errors: list[BaseException] = []

    def _reupsert_loop():
        try:
            for _ in range(_REUPSERT_ITERS):
                if stop.is_set():
                    break
                state.recall_upsert_vectors("tl", "dsl", recs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer = threading.Thread(target=_reupsert_loop, daemon=True)
    writer.start()
    try:
        leaks = 0
        drops = 0
        # Simulate the call site's cold sidecar: every id has a (stale) cold copy.
        cold_ids = set(all_ids)
        for _ in range(_QUERY_ITERS):
            live, suppress_ids = state.recall_list_rows("tl", "dsl", watermark=0)
            live_ids = {r["id"] for r in live}
            # Invariant 1: the split is internally consistent.
            if not live_ids <= suppress_ids:
                leaks += 1
            # Invariant 2: reconstruct the list union exactly as main.py does —
            # cold ids not suppressed, PLUS the recall live ids — and assert no id
            # vanished. The b1-twin drop is precisely an id missing here.
            listed = (cold_ids - suppress_ids) | live_ids
            if listed != all_ids:
                drops += 1
        assert leaks == 0, f"{leaks} observations had live_ids ⊄ suppress_ids"
        assert drops == 0, (
            f"{drops}/{_QUERY_ITERS} lists dropped a live id (cold copy suppressed "
            "but no recall live row appended) — the b1-twin list blackout"
        )
    finally:
        stop.set()
        writer.join(timeout=30)
    assert not errors, f"re-upsert loop raised: {errors[0]!r}"


def test_steady_state_full_recall_under_no_writes(monkeypatch, recall_url):
    """With no concurrent writes, `recall_search` returns every live id exactly once.

    A sanity baseline: the single scan returns all live ids in `suppress_ids` and
    the top_k closest in `matches`, with no duplicates and no tombstones.
    """
    state = _state_on(monkeypatch, recall_url)
    # Clean slate for this test.
    conn = psycopg2.connect(recall_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("TRUNCATE recall_vectors")
            cur.execute("TRUNCATE recall_lsn_seq")
    finally:
        conn.close()

    recs = _records()
    state.recall_upsert_vectors("t2", "ds2", recs)

    suppress_ids, matches = state.recall_search(
        "t2", "ds2", _vec(0), top_k=_N_IDS, watermark=0
    )
    assert suppress_ids == {f"id{i}" for i in range(_N_IDS)}
    match_ids = [m["id"] for m in matches]
    assert len(match_ids) == _N_IDS, "every live id is a match when top_k covers all"
    assert len(set(match_ids)) == _N_IDS, "no duplicate matches"
    # Ascending by score.
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores), "matches must be ascending by L2-squared score"
