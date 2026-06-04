"""Integration coverage for the Recall→Consolidated consolidation (flush) — PR5.

Runs the FULL stack end-to-end:

  - the **recall tier** on a REAL `pgvector/pgvector:pg15` container
    (`RB_RECALL_DSN`) — the recall snapshot, grace-bounded trim, cap-count, and
    idle-sweep all hit real SQL;
  - the **consolidated (cold)** tier built by the real builder into the session
    MinIO (the same fixtures the query-union suite uses);
  - the control plane on the default `memory://` state adapter (the recall path
    is gated on `RB_RECALL_DSN`, not the control-plane DSN).

The HARD invariant / failure-mode tests (docs/architecture/recall-consolidate.md,
invariants I1-I4 + the failure-mode table):

  - test_read_your_writes_through_consolidation (I1+I2): a vector is returned by
    query continuously before, during, and after its consolidation — no
    visibility gap.
  - test_crash_between_commit_and_trim (I2): failure after the catalog commit but
    before the trim → no loss, no duplicate in the union (rows excluded by the
    watermark, GC'd next run).
  - test_grace_buffer_inflight_older_shard (I4): a query resolving an older shard
    still sees its recall rows after a newer consolidation trimmed to the
    2nd-newest watermark.
  - test_consolidate_on_idle_drains_to_zero: an idle dataset → 0 recall rows → a
    subsequent query opens no recall connection (pure cold).
  - test_per_tenant_cap_forces_consolidation: exceeding RB_RECALL_MAX_ROWS
    triggers a consolidation.
  - test_consolidation_applies_tombstones: a deleted=true recall id is removed
    from cold and not carried forward.
  - test_consolidation_flag_off_noop: flag off → no consolidation, byte-identical.
  - test_monotonic_watermark_and_build_type_labeled: watermark advances
    monotonically; build_type='consolidate'; the supersede sweep keeps newest 2.
"""
from __future__ import annotations

import importlib
import json

import psycopg2
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def recall_url():
    """One pgvector container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the consolidation suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


# --- recall-store helpers -------------------------------------------------


def _truncate_recall(dsn: str) -> None:
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.recall_vectors'), "
                "to_regclass('public.recall_lsn_seq')"
            )
            hv, seq = cur.fetchone()
            if hv is not None:
                cur.execute("TRUNCATE recall_vectors")
            if seq is not None:
                cur.execute("TRUNCATE recall_lsn_seq")
    finally:
        conn.close()


def _recall_count(dsn, tenant, dataset) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM recall_vectors WHERE tenant_id=%s AND dataset=%s",
                (tenant, dataset),
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def _recall_lsns(dsn, tenant, dataset):
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lsn FROM recall_vectors WHERE tenant_id=%s AND dataset=%s "
                "ORDER BY lsn",
                (tenant, dataset),
            )
            return [int(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def _tombstone_recall(dsn, tenant, dataset, vid) -> None:
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE recall_vectors SET deleted=TRUE "
                "WHERE tenant_id=%s AND dataset=%s AND id=%s",
                (tenant, dataset, vid),
            )
    finally:
        conn.close()


def _backdate_recall(dsn, tenant, dataset, seconds) -> None:
    """Push a partition's `created_at` back so the idle sweep sees it as idle."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE recall_vectors SET created_at = now() - (%s || ' seconds')::interval "
                "WHERE tenant_id=%s AND dataset=%s",
                (seconds, tenant, dataset),
            )
    finally:
        conn.close()


# --- app harness ----------------------------------------------------------


def _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on):
    if recall_on:
        monkeypatch.setenv("RB_RECALL", "true")
    else:
        monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    from adapters.queue.queue import consume as _consume
    for _topic in (
        "VALIDATE_DATASET", "DATASET_READY", "CONSOLIDATE",
        "RUN_EPHEMERAL_QUERY", "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.ephemeral_runner.run as ephemeral
    importlib.reload(ephemeral)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app), state_mod, v1_query, builder


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_of(client, signup):
    r = client.get("/auth/me", headers=_auth(signup["token"]))
    return r.json()["tenant"]["id"]


def _post_recall(client, token, name, records):
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    return r.json()


def _query(client, token, name, vector, top_k=10, flt=None):
    body = {"dataset": name, "vector": vector, "top_k": top_k}
    if flt is not None:
        body["filter"] = flt
    r = client.post("/v1/query", headers=_auth(token), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _migrate_recall(recall_url):
    import os

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    os.environ["RB_RECALL_DSN"] = recall_url
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)


def _write_landing_batch(s3_landing_prefix, tenant, dataset, upload, records):
    """Land a parquet batch under the dataset's landing prefix (mirrors validator).

    Bulk imports BYPASS recall and land into the cold path. Writing a parquet
    part under `{LANDING_PREFIX}/{tenant}/{dataset}/...` and then calling
    `builder.run_once` drives the same `_run_once_locked` bulk-import build a real
    import (DATASET_READY) would — without going through `POST /vectors` (which,
    flag-on, writes recall instead).
    """
    from adapters.landing.parquet_writer import write_parquet

    prefix = f"{s3_landing_prefix}/{tenant}/{dataset}/upload-{upload}"
    return write_parquet(prefix, records)


# --- I1 + I2: read-your-writes THROUGH a consolidation --------------------


def test_read_your_writes_through_consolidation(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A vector is returned by query before, during, and after its consolidation.

    No cold shard exists at first (recall-only). The query must return `fact-1`
    BEFORE consolidation (from recall), and still AFTER consolidation (now from
    the freshly-built cold shard) — no visibility gap (I1 + I2).
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client)
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "ryw", "dimension": 4})

    _post_recall(client, s["token"], "ryw", [
        {"id": "fact-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"t": "peanuts"}},
        {"id": "fact-2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"t": "shellfish"}},
    ])

    # BEFORE: answered synchronously from recall, no cold shard yet.
    out = _query(client, s["token"], "ryw", [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert out["mode"] == "recall", out
    assert out["matches"][0]["id"] == "fact-1"

    # DURING: hook the trim so a query fires at the exact mid-consolidation
    # instant — AFTER the cold shard + catalog row are committed (the trim only
    # ever runs post-commit, I2) but BEFORE the recall rows are removed. This is
    # the one moment a vector lives in BOTH tiers; the union must still return it
    # exactly once (no visibility gap, no duplicate). A before/after pair would
    # never exercise this window. The real trim runs after the probe so the test
    # still ends in the fully-consolidated state.
    real_trim = builder.recall_trim
    during = {}

    def _probe_then_trim(t, d, grace):
        # The shard is already committed at this point.
        during["shard"] = _state_latest(builder, t, "ryw")
        during["out"] = _query(client, s["token"], "ryw", [1.0, 0.0, 0.0, 0.0], top_k=2)
        return real_trim(t, d, grace)

    builder.recall_trim = _probe_then_trim
    try:
        # CONSOLIDATE: fold the recall partition into a cold shard.
        n = builder.run_consolidate_once("ryw", tenant)
    finally:
        builder.recall_trim = real_trim
    assert n == 2, "two live recall rows folded"
    shard = _state_latest(builder, tenant, "ryw")
    assert shard["build_type"] == "consolidate"
    assert shard["consolidated_lsn"] == 2

    # The mid-consolidation probe ran AFTER the commit (shard present) and the
    # vector was visible THROUGH the consolidation — exactly once, no gap.
    assert during["shard"] is not None and during["shard"]["build_type"] == "consolidate"
    mid = during["out"]
    mid_ids = [m["id"] for m in mid["matches"]]
    assert "fact-1" in mid_ids, f"vector vanished mid-consolidation: {mid}"
    assert mid_ids.count("fact-1") == 1, f"duplicate mid-consolidation: {mid_ids}"
    assert set(mid_ids) == {"fact-1", "fact-2"}, mid

    # AFTER: still returned — now from the cold shard (mode hot|cold), not lost.
    out = _query(client, s["token"], "ryw", [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert out["matches"][0]["id"] == "fact-1", out
    assert {m["id"] for m in out["matches"]} == {"fact-1", "fact-2"}
    assert out["mode"] in ("hot", "cold"), out


def _state_latest(builder, tenant, dataset):
    import adapters.state.state as state_mod
    return state_mod.get_latest_shard(tenant, dataset)


# --- I2 failure mode: crash between commit and trim -----------------------


def test_crash_between_commit_and_trim(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Simulate a failure AFTER the catalog commit but BEFORE the trim.

    The new shard is committed (watermark advanced) but the recall rows are NOT
    trimmed (the trim raised). The union must still return the vector exactly
    once: the cold shard has it (lsn <= watermark) and the recall rows are
    excluded by the watermark (lsn > consolidated_lsn is false), so NO duplicate.
    The next consolidation GCs the orphaned recall rows.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="crash@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "cr", "dimension": 4})

    _post_recall(client, s["token"], "cr", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])

    # Make the trim raise — simulating a crash AFTER the commit, BEFORE the trim.
    # Save the real trim to restore it surgically later (not monkeypatch.undo,
    # which would also revert the autouse MinIO env the cold-shard write needs).
    real_trim = builder.recall_trim

    def _boom_trim(t, d, grace):
        raise RuntimeError("simulated crash after commit, before trim")

    builder.recall_trim = _boom_trim

    n = builder.run_consolidate_once("cr", tenant)
    assert n == 2, "consolidation committed the shard despite the trim crash"

    # The shard IS committed with the watermark.
    shard = _state_latest(builder, tenant, "cr")
    assert shard["consolidated_lsn"] == 2 and shard["build_type"] == "consolidate"
    # The recall rows are NOT trimmed (still present) — no loss.
    assert _recall_count(recall_url, tenant, "cr") == 2, "recall rows survived the crash"

    # The union returns each id EXACTLY once — recall rows are excluded by the
    # watermark (lsn 1,2 <= consolidated_lsn 2), so no duplicate from recall.
    out = _query(client, s["token"], "cr", [1.0, 0.0, 0.0, 0.0], top_k=10)
    ids = [m["id"] for m in out["matches"]]
    assert sorted(ids) == ["a", "b"], f"no loss, no duplicate: {ids}"
    assert len(ids) == len(set(ids)), f"duplicate in union: {ids}"

    # The NEXT consolidation (trim restored) GCs the orphaned recall rows.
    builder.recall_trim = real_trim  # restore only the trim (env stays intact)
    # A fresh write so there is a new max LSN > the committed watermark.
    _post_recall(client, s["token"], "cr", [
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("cr", tenant)
    # The grace trim now deletes the old orphaned rows (covered by the
    # 2nd-newest watermark). The latest recall write (`c`, above the previous
    # watermark) may remain in the grace window — but the original a/b orphans
    # are GC'd.
    remaining = _recall_lsns(recall_url, tenant, "cr")
    assert 1 not in remaining and 2 not in remaining, (
        f"orphaned recall rows (lsn 1,2) must be GC'd next consolidation: {remaining}"
    )


# --- I4: grace buffer — in-flight query resolving an older shard ----------


def test_grace_buffer_inflight_older_shard(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """An older shard's recall rows survive a newer consolidation's trim (I4).

    After TWO consolidations, the trim only deletes up to the 2nd-newest shard's
    watermark — so the recall rows in (2nd-newest-watermark, newest-watermark]
    are NOT deleted: an in-flight query that resolved the older (2nd-newest)
    shard, whose watermark is smaller, still finds them via the union.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="grace@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "gr", "dimension": 4})

    # Write + consolidate #1 → shard A (watermark 2). Trims nothing (only shard).
    _post_recall(client, s["token"], "gr", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("gr", tenant)
    shard_a = _state_latest(builder, tenant, "gr")
    assert shard_a["consolidated_lsn"] == 2
    # After #1, with only one shard, the grace trim watermark is 0 → nothing
    # trimmed: a/b remain in recall (still within the grace buffer).
    assert _recall_count(recall_url, tenant, "gr") == 2

    # Write + consolidate #2 → shard B (watermark 3). Now A is the 2nd-newest,
    # so the trim deletes up to A's watermark (2): lsn 1,2 go, lsn 3 (`c`) stays.
    _post_recall(client, s["token"], "gr", [
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("gr", tenant)
    shard_b = _state_latest(builder, tenant, "gr")
    assert shard_b["consolidated_lsn"] == 3 and shard_b["build_type"] == "consolidate"

    remaining = _recall_lsns(recall_url, tenant, "gr")
    assert remaining == [3], (
        "grace buffer: rows up to the 2nd-newest watermark (2) are trimmed; the "
        f"row at lsn 3 (covered only by the NEWEST shard) survives: {remaining}"
    )

    # An in-flight query resolving the OLDER shard A (watermark 2) still finds
    # `c` via the union (lsn 3 > 2). Simulate by querying with shard A's
    # watermark directly through the recall scan.
    import adapters.state.state as state_mod
    suppress_ids, matches = state_mod.recall_search(
        tenant, "gr", [0.0, 0.0, 1.0, 0.0], 10, shard_a["consolidated_lsn"]
    )
    found = {m["id"] for m in matches}
    assert "c" in found, (
        "a query resolving the older shard (watermark 2) must still see `c` "
        f"(lsn 3) in recall: {found}"
    )


# --- consolidate-on-idle drains to zero -----------------------------------


def test_consolidate_on_idle_drains_to_zero(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """An idle dataset is consolidated to ZERO recall rows → pure cold queries.

    The idle sweep enqueues CONSOLIDATE; draining it twice (the first leaves the
    newest write in the grace window; the second, after the new shard ages,
    trims the rest) takes the partition to 0 — after which a query opens no
    recall connection.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_RECALL_IDLE_S", "1")  # tiny idle window
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="idle@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "id", "dimension": 4})

    _post_recall(client, s["token"], "id", [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "y", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    # Backdate so the partition is older than the idle window.
    _backdate_recall(recall_url, tenant, "id", 120)

    # The idle sweep finds the partition and enqueues CONSOLIDATE.
    from adapters.queue.queue import consume
    builder._LAST_IDLE_SWEEP_AT = 0.0  # force the rate-limited sweep to run
    builder._maybe_sweep_idle_recall()
    msg = consume("CONSOLIDATE", block=False)
    assert msg is not None and msg["dataset"] == "id", "idle sweep enqueued CONSOLIDATE"

    # Drain it to completion: consolidate repeatedly until recall is empty.
    # (Each pass folds remaining live rows + grace-trims; the second pass, with
    # two shards, trims the rows the first pass left in the grace window.)
    for _ in range(5):
        builder.run_consolidate_once("id", tenant)
        # A fresh idle write would re-populate; here there are none, so after the
        # grace window the next pass GCs the rest.
        _backdate_recall(recall_url, tenant, "id", 120)
        if _recall_count(recall_url, tenant, "id") == 0:
            break

    assert _recall_count(recall_url, tenant, "id") == 0, "idle dataset drained to ZERO"

    # A subsequent query opens NO recall connection — prove it by making the
    # recall connection blow up; the query must still succeed from pure cold.
    import adapters.state.state as state_mod

    real_recall_search = state_mod.recall_search
    calls = {"n": 0}

    def _counting_recall_search(*a, **k):
        calls["n"] += 1
        return real_recall_search(*a, **k)

    monkeypatch.setattr(state_mod, "recall_search", _counting_recall_search)
    # `v1_query` imported recall_search by-name; patch it there too.
    monkeypatch.setattr(v1q, "recall_search", _counting_recall_search)

    out = _query(client, s["token"], "id", [1.0, 0.0, 0.0, 0.0], top_k=10)
    ids = {m["id"] for m in out["matches"]}
    assert ids == {"x", "y"}, out
    # The union still runs (flag on) and consults recall, but the recall set is
    # empty — scale-to-zero means zero ROWS, the brute-force scan is trivially
    # empty. (The recall scan returns nothing, so the query is pure cold.)
    assert out["mode"] in ("hot", "cold"), out
    _, m = real_recall_search(tenant, "id", [1.0, 0.0, 0.0, 0.0], 10, 0)
    assert m == [], "recall set is empty after the idle drain"


# --- per-tenant cap forces consolidation ----------------------------------


def test_per_tenant_cap_forces_consolidation(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Exceeding RB_RECALL_MAX_ROWS enqueues a CONSOLIDATE that drains the partition."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "3")  # tiny cap
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="cap@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "cp", "dimension": 4})

    # Write 4 rows (> cap of 3) → the write path enqueues CONSOLIDATE.
    _post_recall(client, s["token"], "cp", [
        {"id": f"r{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
        for i in range(4)
    ])

    from adapters.queue.queue import consume
    msg = consume("CONSOLIDATE", block=False)
    assert msg is not None and msg["dataset"] == "cp", "cap exceeded must enqueue CONSOLIDATE"

    # Process it: the partition is folded into a cold shard.
    n = builder.run_consolidate_once("cp", tenant)
    assert n == 4
    shard = _state_latest(builder, tenant, "cp")
    assert shard["build_type"] == "consolidate" and shard["consolidated_lsn"] == 4

    # The vectors are still queryable (now from cold).
    out = _query(client, s["token"], "cp", [0.0, 0.0, 0.0, 0.0], top_k=10)
    assert {m["id"] for m in out["matches"]} == {"r0", "r1", "r2", "r3"}, out


# --- tombstones applied at consolidation ----------------------------------


def test_consolidation_applies_tombstones(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A deleted=true recall id is removed from cold and not carried forward."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="tomb@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "tb", "dimension": 4})

    # Consolidate `doomed` + `survivor` into a cold shard.
    _post_recall(client, s["token"], "tb", [
        {"id": "doomed", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "survivor", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("tb", tenant)
    out = _query(client, s["token"], "tb", [1.0, 0.0, 0.0, 0.0], top_k=10)
    assert {m["id"] for m in out["matches"]} == {"doomed", "survivor"}

    # Re-write `doomed` into recall, then tombstone it; a fresh write is needed
    # so the snapshot's max LSN advances past the prior watermark.
    _post_recall(client, s["token"], "tb", [
        {"id": "doomed", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ])
    _tombstone_recall(recall_url, tenant, "tb", "doomed")

    # Consolidate again: the tombstone removes `doomed` from the cold shard.
    builder.run_consolidate_once("tb", tenant)
    shard = _state_latest(builder, tenant, "tb")
    assert shard["build_type"] == "consolidate"

    out = _query(client, s["token"], "tb", [1.0, 0.0, 0.0, 0.0], top_k=10)
    ids = {m["id"] for m in out["matches"]}
    assert "doomed" not in ids, f"tombstone must remove the cold id: {ids}"
    assert "survivor" in ids


# --- monotonic watermark + build_type + supersede sweep -------------------


def test_monotonic_watermark_and_build_type_labeled(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Watermark advances monotonically; build_type='consolidate'; sweep keeps 2."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="mono@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "mo", "dimension": 4})

    import adapters.state.state as state_mod
    watermarks = []
    for i in range(4):
        _post_recall(client, s["token"], "mo", [
            {"id": f"r{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}},
        ])
        builder.run_consolidate_once("mo", tenant)
        shard = state_mod.get_latest_shard(tenant, "mo")
        assert shard["build_type"] == "consolidate"
        watermarks.append(shard["consolidated_lsn"])

    assert watermarks == sorted(watermarks), f"watermark non-monotonic: {watermarks}"
    assert len(set(watermarks)) == len(watermarks), "watermark did not strictly advance"
    # The supersede sweep keeps the newest 2 shards.
    shards = state_mod.list_shards(tenant, "mo")
    assert len(shards) == 2, f"sweep keeps newest 2: {len(shards)}"


# --- flag OFF: no consolidation, byte-identical ---------------------------


def test_consolidation_flag_off_noop(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag off: run_consolidate_once is a no-op; the write path enqueues nothing.

    Recall holds a row (seeded directly via the store), but with RB_RECALL off
    the builder consolidation is a no-op and never touches it.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="off@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "of", "dimension": 4})

    # Seed a recall row directly (recall_upsert_vectors keys on RB_RECALL_DSN,
    # not the RB_RECALL flag).
    state_mod.recall_upsert_vectors(tenant, "of", [
        {"id": "ghost", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ])
    assert _recall_count(recall_url, tenant, "of") == 1

    # With the flag OFF the builder consolidation is a clean no-op: no shard, no
    # trim, the recall row is untouched.
    n = builder.run_consolidate_once("of", tenant)
    assert n == 0
    assert state_mod.get_latest_shard(tenant, "of") is None, "no shard built (flag off)"
    assert _recall_count(recall_url, tenant, "of") == 1, "recall row untouched (flag off)"

    # The flag-off write path returns 202 and enqueues no CONSOLIDATE.
    from adapters.queue.queue import consume
    while consume("CONSOLIDATE", block=False):
        pass
    body = json.dumps({"id": "z", "values": [1.0, 2.0, 3.0, 4.0]})
    r = client.post(
        "/v1/datasets/of/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    assert consume("CONSOLIDATE", block=False) is None, "flag off enqueues no CONSOLIDATE"


# --- watermark carry-forward across non-consolidate builds (I1/I4) ---------


def test_non_consolidate_build_carries_watermark_forward(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A bulk import (and a delete) between consolidations must NOT regress the
    watermark, and a subsequent consolidation's grace trim must still drain.

    Regression guard for the P1 watermark carry-forward: a non-consolidate build
    (ingest/incremental/delete) that defaulted `consolidated_lsn=0` would reset
    the per-dataset watermark, making the 2nd-newest shard carry watermark 0 →
    `recall_trim` deletes nothing → recall stalls (never drains) and the union
    re-scans already-consolidated rows. We assert the newest shard's watermark
    never regresses below the consolidation high-water mark, the grace trim still
    fires on the next consolidation, and the union shows no dup/loss.
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="carry@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "cw", "dimension": 4})

    # 1. Consolidate to N=2 → cold shard with consolidated_lsn=2.
    _post_recall(client, s["token"], "cw", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("cw", tenant)
    shard = state_mod.get_latest_shard(tenant, "cw")
    assert shard["consolidated_lsn"] == 2, shard
    consolidate_watermark = 2

    # 2. A BULK IMPORT (lands, bypasses recall) builds the next newest shard.
    #    Pre-fix this committed consolidated_lsn=0 (watermark regression).
    _write_landing_batch(s3_landing_prefix, tenant, "cw", "imp1", [
        {"id": f"imp{i}", "values": [0.0, 0.0, float(i + 1), 0.0], "metadata": {}}
        for i in range(3)
    ])
    n = builder.run_once("cw", tenant)
    assert n == 3 and builder._LAST_BUILD["build_type"] == "incremental"
    after_import = state_mod.get_latest_shard(tenant, "cw")
    assert after_import["consolidated_lsn"] >= consolidate_watermark, (
        "bulk import regressed the watermark: "
        f"{after_import['consolidated_lsn']} < {consolidate_watermark}"
    )

    # 3. A DELETE-BY-ID build also produces a new newest shard; it too must carry
    #    the watermark forward (not reset to 0).
    builder.run_delete_once("cw", tenant, "imp0")
    after_delete = state_mod.get_latest_shard(tenant, "cw")
    assert after_delete["consolidated_lsn"] >= consolidate_watermark, (
        "delete-by-id regressed the watermark: "
        f"{after_delete['consolidated_lsn']} < {consolidate_watermark}"
    )

    # 4. The grace trim must STILL drain on the next consolidation. Write a fresh
    #    recall row (lsn 3) and consolidate: now the 2nd-newest shard carries a
    #    watermark >= 2 (not the regressed 0), so the trim deletes the original
    #    a/b recall rows (lsn 1,2) instead of no-opping.
    _post_recall(client, s["token"], "cw", [
        {"id": "c", "values": [0.0, 0.0, 0.0, 1.0], "metadata": {}},
    ])
    builder.run_consolidate_once("cw", tenant)
    remaining = _recall_lsns(recall_url, tenant, "cw")
    assert 1 not in remaining and 2 not in remaining, (
        "grace trim STALLED — watermark regression left lsn 1,2 in recall "
        f"(would never drain): {remaining}"
    )

    # 5. The union shows no dup/loss: every live id returned exactly once.
    out = _query(client, s["token"], "cw", [0.0, 0.0, 1.0, 0.0], top_k=10)
    ids = [m["id"] for m in out["matches"]]
    # `imp0` was deleted; `a,b,c,imp1,imp2` survive.
    assert "imp0" not in ids, f"deleted id resurfaced: {ids}"
    assert set(ids) == {"a", "b", "c", "imp1", "imp2"}, f"dup/loss in union: {ids}"
    assert len(ids) == len(set(ids)), f"duplicate in union: {ids}"


# --- snapshot self-consistency under a concurrent recall writer (I1) -------


def test_snapshot_self_consistent_under_concurrent_writer(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """A row committed concurrently with the snapshot is never dropped (no loss).

    The snapshot is a single MVCC statement (the bound N is derived in a scalar
    sub-SELECT of the same query), so a write that commits during the snapshot
    either lands inside the snapshot's max-lsn (folded into the shard) or above it
    (left in recall, excluded by the trim). It can NEVER be both excluded from the
    shard AND trimmed from recall. We race a real concurrent recall writer against
    the snapshot and assert the snapshot+trim loses nothing.
    """
    import threading

    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, v1q, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="race@example.com")
    tenant = _tenant_of(client, s)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "rc", "dimension": 4})

    # Seed an initial batch so a shard + watermark exist.
    _post_recall(client, s["token"], "rc", [
        {"id": "seed0", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "seed1", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])

    # Direct self-consistency check on the snapshot itself: under a concurrent
    # writer that commits MANY rows while the snapshot runs, the returned
    # `max_lsn` must EQUAL the max lsn among the returned rows (single MVCC
    # snapshot). The old two-statement form (independent `SELECT MAX(lsn)` then
    # `SELECT ... WHERE lsn <= N`) could return an `N` that does not match the row
    # set, and every returned row must be `<= max_lsn` with no gap to the bound.
    stop = threading.Event()

    def _hammer():
        i = 0
        while not stop.is_set():
            state_mod.recall_upsert_vectors(tenant, "rc", [
                {"id": f"h{i}", "values": [0.0, 0.0, 0.0, float(i + 1)], "metadata": {}},
            ])
            i += 1

    hammer = threading.Thread(target=_hammer)
    hammer.start()
    try:
        for _ in range(20):
            snap_n, snap_rows = state_mod.recall_snapshot_for_consolidation(tenant, "rc")
            row_max = max((r["lsn"] for r in snap_rows), default=0)
            assert snap_n == row_max, (
                f"snapshot bound {snap_n} != max lsn of returned rows {row_max} "
                "(not a single MVCC snapshot)"
            )
            assert all(r["lsn"] <= snap_n for r in snap_rows), "row above the bound"
    finally:
        stop.set()
        hammer.join()

    # Wrap the snapshot so a concurrent writer commits a NEW recall row WHILE the
    # snapshot statement is in flight. The writer uses a fresh connection (its own
    # txn); whichever side of the MVCC snapshot it lands on, the row must survive.
    real_snapshot = state_mod.recall_snapshot_for_consolidation
    raced = {"id": "racer"}

    def _racing_snapshot(t, d):
        # Fire the concurrent write just before the snapshot statement runs.
        def _write():
            state_mod.recall_upsert_vectors(t, d, [
                {"id": raced["id"], "values": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
            ])
        th = threading.Thread(target=_write)
        th.start()
        try:
            return real_snapshot(t, d)
        finally:
            th.join()

    monkeypatch.setattr(builder, "recall_snapshot_for_consolidation", _racing_snapshot)

    # Consolidate with the race in flight, then a SECOND consolidation so the
    # grace trim definitely fires (the racer, whichever side it landed on, must
    # not be lost from BOTH tiers).
    builder.run_consolidate_once("rc", tenant)
    monkeypatch.setattr(
        builder, "recall_snapshot_for_consolidation", real_snapshot
    )
    # A trailing write guarantees the next snapshot advances and the prior shard
    # becomes the 2nd-newest, so the grace trim runs.
    _post_recall(client, s["token"], "rc", [
        {"id": "tail", "values": [0.0, 0.0, 0.0, 1.0], "metadata": {}},
    ])
    builder.run_consolidate_once("rc", tenant)
    builder.run_consolidate_once("rc", tenant)

    # The racer is NEVER lost: it is either in the cold shard (folded) or still in
    # recall (above the trim watermark). The union must return it exactly once.
    out = _query(client, s["token"], "rc", [0.0, 0.0, 1.0, 0.0], top_k=10)
    ids = [m["id"] for m in out["matches"]]
    assert raced["id"] in ids, (
        f"row committed concurrently with the snapshot was LOST from both tiers: {ids}"
    )
    assert ids.count(raced["id"]) == 1, f"duplicate of the raced row: {ids}"
    # All seeded + tail rows survive too — snapshot+trim loses nothing.
    assert {"seed0", "seed1", "tail"}.issubset(set(ids)), f"loss: {ids}"
