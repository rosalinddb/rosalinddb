"""Tests for metadata filtering on `POST /v1/query`.

`filter` is a flat object of field->value with AND-of-equals semantics:
a result is kept only if, for EVERY key in `filter`, the record's metadata
contains that key with an exactly-equal value (same type — no coercion).

These tests drive the full ingest pipeline inline (validator + builder) to
produce a real FAISS shard with metadata sidecar, then exercise the filter
through the public `POST /v1/query` route.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient over source_registry + mounted v1_query router.

    Mirrors the fixture in `test_query_api.py`: per-test MinIO landing/index
    prefixes + a local FAISS shard cache, reset in-memory state, pipeline
    modules reloaded so module-level env reads pick up the patched values.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny test fixtures use flat

    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY"):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
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
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _run_pipeline_once():
    from adapters.queue.queue import consume
    from services.validator_worker.run import process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


def _upload(client, token, name, records):
    body = "\n".join(json.dumps(r) for r in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()


def _make_indexed_dataset(client, token, name="test", dimension=4, records=None):
    r = client.post(
        "/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension}
    )
    assert r.status_code == 201, r.text
    assert records is not None, "records required"
    _upload(client, token, name, records)
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds
    return records


def _query(client, token, **body):
    return client.post("/v1/query", headers=_auth(token), json={"dataset": "test", **body})


def _tenant_id(client, token):
    """Resolve the tenant_id for a signed-up user via /auth/me."""
    r = client.get("/auth/me", headers=_auth(token))
    assert r.status_code == 200, r.text
    return r.json()["tenant"]["id"]


# A fixture set spanning two metadata keys and mixed value types.
def _records():
    return [
        {"id": "doc-0", "values": [0.0, 0.0, 0.0, 0.0], "metadata": {"category": "books", "year": 2024}},
        {"id": "doc-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "books", "year": 2023}},
        {"id": "doc-2", "values": [2.0, 0.0, 0.0, 0.0], "metadata": {"category": "music", "year": 2024}},
        {"id": "doc-3", "values": [3.0, 0.0, 0.0, 0.0], "metadata": {"category": "music", "year": 2023}},
        {"id": "doc-4", "values": [4.0, 0.0, 0.0, 0.0], "metadata": {"category": "books"}},  # no year
        {"id": "doc-5", "values": [5.0, 0.0, 0.0, 0.0], "metadata": {}},  # no metadata keys
    ]


# --- filter matches a subset ---------------------------------------------


def test_filter_matches_subset(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"category": "books"})
    assert r.status_code == 200, r.text
    ids = {m["id"] for m in r.json()["matches"]}
    # doc-0, doc-1, doc-4 have category=books; doc-2/3/5 do not.
    assert ids == {"doc-0", "doc-1", "doc-4"}


# --- AND semantics across two keys ---------------------------------------


def test_filter_and_semantics(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"category": "books", "year": 2024})
    assert r.status_code == 200, r.text
    ids = {m["id"] for m in r.json()["matches"]}
    # Only doc-0 has BOTH category=books AND year=2024.
    assert ids == {"doc-0"}


# --- a record missing a filtered key is excluded -------------------------


def test_filter_missing_key_excluded(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    r = _query(client, s["token"], vector=[4.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"year": 2024})
    assert r.status_code == 200, r.text
    ids = {m["id"] for m in r.json()["matches"]}
    # doc-4 (no year key) and doc-5 (empty metadata) must be excluded.
    assert "doc-4" not in ids
    assert "doc-5" not in ids
    assert ids == {"doc-0", "doc-2"}


# --- type mismatch does not match ----------------------------------------


def test_filter_type_mismatch_does_not_match(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    # metadata year is int 2024; filter value string "2024" must NOT match.
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"year": "2024"})
    assert r.status_code == 200, r.text
    assert r.json()["matches"] == []


def test_filter_null_value_never_matches(client):
    """A `null` filter VALUE is accepted but never matches (review nit 2)."""
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    # `{"category": null}` passes request validation (not a dict/list) but
    # must exclude every record — null is not a meaningful equality target.
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"category": None})
    assert r.status_code == 200, r.text
    assert r.json()["matches"] == []


# --- empty / absent filter leaves results unchanged ----------------------


def test_empty_filter_unchanged(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    baseline = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10)
    assert baseline.status_code == 200, baseline.text
    base_ids = {m["id"] for m in baseline.json()["matches"]}

    empty = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10, filter={})
    assert empty.status_code == 200, empty.text
    assert {m["id"] for m in empty.json()["matches"]} == base_ids

    null = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10, filter=None)
    assert null.status_code == 200, null.text
    assert {m["id"] for m in null.json()["matches"]} == base_ids

    # All 6 records come back when nothing is filtered.
    assert len(base_ids) == 6


# --- filter matching nothing returns an empty list, not an error ---------


def test_filter_matches_nothing_returns_empty(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"category": "nonexistent"})
    assert r.status_code == 200, r.text
    assert r.json()["matches"] == []
    assert r.json()["mode"] in ("hot", "cold")


# --- over-fetch preserves correct top-K ordering among matches -----------


def test_filter_over_fetch_topk_ordering(client):
    """With a selective filter, the K nearest *matching* records come back
    in ascending-distance order — over-fetch must not drop near matches."""
    s = _signup(client)
    # 20 records; even ids tagged keep=yes, interleaved by distance.
    records = [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i % 2 == 0 else "no"},
        }
        for i in range(20)
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=3,
               filter={"keep": "yes"})
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert len(matches) == 3
    # The 3 nearest keep=yes records to origin are v0, v2, v4 in that order.
    assert [m["id"] for m in matches] == ["v0", "v2", "v4"]
    # Scores ascending (FAISS L2 distance, nearest first).
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores)


def test_filter_truncates_to_top_k(client):
    """When more records match than top_k, only top_k nearest are returned."""
    s = _signup(client)
    records = [
        {"id": f"d{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"g": "a"}}
        for i in range(15)
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=5,
               filter={"g": "a"})
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert len(matches) == 5
    assert [m["id"] for m in matches] == ["d0", "d1", "d2", "d3", "d4"]


# --- metadata roundtrip survives the pipeline ----------------------------


def test_metadata_roundtrip_supports_filtering(client):
    """Proves metadata uploaded -> shard sidecar -> query is intact enough
    to filter on, including numeric values which must keep their JSON type."""
    s = _signup(client)
    records = [
        {"id": "x", "values": [0.0, 0.0, 0.0, 0.0], "metadata": {"n": 7, "s": "hi"}},
        {"id": "y", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"n": 9, "s": "bye"}},
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=2,
               filter={"n": 7})
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["id"] == "x"
    assert matches[0]["metadata"] == {"n": 7, "s": "hi"}


# --- ephemeral path: filter must apply on the brute-force runner too -----
#
# The ephemeral runner serves a `POST /v1/query` against a dataset with no
# shard yet (still indexing). It does a brute-force search over the dataset
# and previously ignored `filter` entirely — a filtered query silently
# returned UNFILTERED results. These tests drive `handle()` directly with a
# `filter` argument; they fail until the runner honours it.


def _ephemeral_handle(client, token, vector, top_k=10, flt=None):
    """Run the ephemeral runner's `handle()` inline against dataset 'test'.

    Returns the runner's `matches` list, mirroring the production path where
    the runner consumes RUN_EPHEMERAL_QUERY and computes the result.
    """
    from services.ephemeral_runner.run import handle as ephemeral_handle

    tenant = _tenant_id(client, token)
    return ephemeral_handle(tenant, "test", vector, top_k, flt)


def test_ephemeral_filter_matches_subset(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10, flt={"category": "books"}
    )
    ids = {m["id"] for m in matches}
    # doc-0, doc-1, doc-4 have category=books; doc-2/3/5 do not.
    assert ids == {"doc-0", "doc-1", "doc-4"}


def test_ephemeral_filter_and_semantics(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10,
        flt={"category": "books", "year": 2024},
    )
    ids = {m["id"] for m in matches}
    # Only doc-0 has BOTH category=books AND year=2024.
    assert ids == {"doc-0"}


def test_ephemeral_filter_missing_key_excluded(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    matches = _ephemeral_handle(
        client, s["token"], [4.0, 0.0, 0.0, 0.0], top_k=10, flt={"year": 2024}
    )
    ids = {m["id"] for m in matches}
    # doc-4 (no year key) and doc-5 (empty metadata) must be excluded.
    assert "doc-4" not in ids
    assert "doc-5" not in ids
    assert ids == {"doc-0", "doc-2"}


def test_ephemeral_filter_matches_nothing_returns_empty(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10,
        flt={"category": "nonexistent"},
    )
    # A selective filter may legitimately return zero — not an error.
    assert matches == []


def test_ephemeral_empty_filter_unchanged(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"], records=_records())
    baseline = _ephemeral_handle(client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10)
    base_ids = {m["id"] for m in baseline}
    # All 6 records come back when nothing is filtered.
    assert len(base_ids) == 6

    empty = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10, flt={}
    )
    assert {m["id"] for m in empty} == base_ids

    null = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10, flt=None
    )
    assert {m["id"] for m in null} == base_ids


def test_ephemeral_filter_truncates_to_top_k(client):
    """When more records match than top_k, only top_k nearest are returned."""
    s = _signup(client)
    records = [
        {"id": f"d{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"g": "a"}}
        for i in range(15)
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=5, flt={"g": "a"}
    )
    assert len(matches) == 5
    assert [m["id"] for m in matches] == ["d0", "d1", "d2", "d3", "d4"]


# --- filtered-query under-count regression (fix/filtered-query-undercount) ---
#
# A filtered `top_k` query must return exactly `min(top_k, total_matching)`;
# the result count must NOT depend on `top_k`. The old hot path post-filtered
# a fixed over-fetch (`max(top_k*10, 100)` candidates). On a 200-vector
# dataset whose filter-matching records are interleaved deep into the distance
# order, the nearest 100 candidates contained only ~8 of the 21 matches, so
# `top_k=10` silently under-returned 8 while `top_k=25` (over-fetch 250 →
# whole shard) returned all 21. These tests fail on the old over-fetch code.


def _undercount_records(n_total=200, near=8, far=13, far_start=120):
    """Build `n_total` records along axis 0 at distance == index.

    `keep:yes` is set on the `near` nearest indices (0..near-1) AND on `far`
    indices starting at `far_start` — so the matching records are NOT the
    global nearest neighbours; they straddle rank 100. Total matches =
    `near + far`. Every other record is `keep:no`.
    """
    keep = set(range(near)) | set(range(far_start, far_start + far))
    return [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i in keep else "no"},
        }
        for i in range(n_total)
    ]


def test_filtered_query_topk_less_than_matches_returns_exactly_topk(client):
    """top_k < total_matching -> returns exactly top_k, all matching.

    The 21 matching records straddle rank 100, so the old fixed over-fetch of
    100 candidates only saw ~8 of them and returned 8 for top_k=10.
    """
    s = _signup(client)
    records = _undercount_records()  # 21 matches: 8 near + 13 far
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
               filter={"keep": "yes"})
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    # Exactly top_k, all matching the filter, nearest-first.
    assert len(matches) == 10
    assert all(m["metadata"].get("keep") == "yes" for m in matches)
    assert [m["id"] for m in matches] == [f"v{i}" for i in range(8)] + [
        "v120", "v121"
    ]
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores)


def test_filtered_query_count_independent_of_topk(client):
    """The result count must not change with top_k for the same dataset.

    Old code: top_k=10 -> 8 results, top_k=25 -> 21. New code: top_k=10 -> 10,
    top_k=25 -> 21 (== total matches). The under-fetch dependency is gone.
    """
    s = _signup(client)
    records = _undercount_records()
    _make_indexed_dataset(client, s["token"], records=records)

    r10 = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
                 filter={"keep": "yes"})
    r25 = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=25,
                 filter={"keep": "yes"})
    assert r10.status_code == 200 and r25.status_code == 200
    assert len(r10.json()["matches"]) == 10  # min(10, 21)
    assert len(r25.json()["matches"]) == 21  # min(25, 21) == total matches


def test_filtered_query_topk_greater_than_matches_returns_match_count(client):
    """top_k > total_matching -> returns exactly the match count."""
    s = _signup(client)
    records = _undercount_records()  # 21 matches
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=50,
               filter={"keep": "yes"})
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert len(matches) == 21
    assert all(m["metadata"].get("keep") == "yes" for m in matches)


def test_unfiltered_query_unchanged(client):
    """Regression guard: an unfiltered query still returns exactly top_k."""
    s = _signup(client)
    records = _undercount_records()
    _make_indexed_dataset(client, s["token"], records=records)
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10)
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert len(matches) == 10
    # The 10 nearest to origin are v0..v9, regardless of metadata.
    assert [m["id"] for m in matches] == [f"v{i}" for i in range(10)]


def test_ephemeral_filtered_query_topk_less_than_matches(client):
    """Ephemeral path: top_k < total_matching -> exactly top_k matches."""
    s = _signup(client)
    records = _undercount_records()
    _make_indexed_dataset(client, s["token"], records=records)
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=10, flt={"keep": "yes"}
    )
    assert len(matches) == 10
    assert all(m["metadata"].get("keep") == "yes" for m in matches)


# --- IVF cell-coverage: filtered search must scan every cell ----------------
#
# The hot path's index is FAISS IVFFlat. `index.search` only scans `nprobe`
# cells (server default 64). A filter-matching vector in an UNPROBED cell is
# invisible no matter how large `fetch_k` is. A correct filtered search sets
# `nprobe = nlist` so every cell is scanned. This needs an IVF index whose
# `nlist` exceeds the default `nprobe` (64): `_choose_nlist` gives N//8 for
# N>512, so ~600 vectors yields nlist ~75.


@pytest.fixture
def ivf_client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Like `client` but builds a real IVFFlat index (INDEX_TYPE=ivfflat)."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")

    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY"):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
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
    return TestClient(main_mod.app)


def test_ivf_filtered_query_scans_all_cells(ivf_client):
    """On an IVFFlat index, a filtered query must return ALL matches even
    when matching vectors land in cells the default nprobe would not scan.

    The dataset is large enough that `nlist > 64` (default nprobe). The 21
    matching vectors are spread across the whole coordinate range so they
    fall into many different IVF cells; only `nprobe = nlist` (full cell
    coverage) guarantees every one is found.
    """
    client = ivf_client
    s = _signup(client)
    n_total = 600
    # 21 matches spread evenly across the full index range so they scatter
    # into many IVF cells (cells partition the coordinate space).
    match_idx = {int(round(i * (n_total - 1) / 20)) for i in range(21)}
    assert len(match_idx) == 21
    records = [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i in match_idx else "no"},
        }
        for i in range(n_total)
    ]
    _make_indexed_dataset(client, s["token"], dimension=4, records=records)

    import faiss
    import services.query_api.v1_query as v1_query
    # top_k smaller than the 21 matches -> still must return exactly top_k,
    # and a larger top_k must return all 21.
    r_small = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=10,
                     filter={"keep": "yes"})
    r_all = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=30,
                   filter={"keep": "yes"})
    assert r_small.status_code == 200 and r_all.status_code == 200
    # Now the index is cached; assert it is genuinely IVF with nlist > 64.
    index = list(v1_query._SHARD_CACHE.values())[0][0]
    ivf = faiss.extract_index_ivf(index)
    assert ivf.nlist > v1_query.DEFAULT_NPROBE, (
        f"test needs nlist > {v1_query.DEFAULT_NPROBE}; got nlist={ivf.nlist}"
    )
    assert len(r_small.json()["matches"]) == 10
    all_matches = r_all.json()["matches"]
    assert len(all_matches) == 21, (
        "filtered query under-returned on an IVF index — cell coverage gap"
    )
    assert {m["id"] for m in all_matches} == {f"v{i}" for i in sorted(match_idx)}


def test_ivf_ephemeral_filtered_query_scans_all_cells(ivf_client):
    """Ephemeral path on an IVFFlat index: filtered search scans all cells."""
    client = ivf_client
    s = _signup(client)
    n_total = 600
    match_idx = {int(round(i * (n_total - 1) / 20)) for i in range(21)}
    records = [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i in match_idx else "no"},
        }
        for i in range(n_total)
    ]
    _make_indexed_dataset(client, s["token"], dimension=4, records=records)
    matches = _ephemeral_handle(
        client, s["token"], [0.0, 0.0, 0.0, 0.0], top_k=30, flt={"keep": "yes"}
    )
    assert len(matches) == 21, (
        "ephemeral filtered query under-returned on an IVF index"
    )
    assert {m["id"] for m in matches} == {f"v{i}" for i in sorted(match_idx)}


# --- coverage: flat-index and top_k > shard-size filtered queries ----------
#
# Two coverage cases for the exhaustive-when-filtered path (review nit):
#   (a) A TINY dataset (< 64 vectors) builds a FLAT index, not IVFFlat —
#       `_ivf_search_params` returns `search_params is None` for it. The
#       filtered query must still return exactly `min(top_k, total_matching)`.
#   (b) An IVF-indexed dataset queried with `top_k` LARGER than the shard's
#       vector count must return exactly the total filter-matching count —
#       no error, no `-1` padding leaking into the result set.


def test_filtered_query_tiny_flat_index_returns_exactly_topk(client):
    """A < 64-vector dataset builds a flat index (search_params is None);
    a filtered query still returns exactly `min(top_k, total_matching)`."""
    s = _signup(client)
    # 30 vectors: 12 tagged keep=yes (interleaved), 18 keep=no. Tiny enough
    # that the builder produces a flat index, exercising the flat branch.
    n_total = 30
    keep = set(range(0, 24, 2))  # 12 even indices in 0..22
    assert len(keep) == 12
    records = [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i in keep else "no"},
        }
        for i in range(n_total)
    ]
    _make_indexed_dataset(client, s["token"], records=records)

    import faiss
    import services.query_api.v1_query as v1_query

    # top_k < total_matching -> exactly top_k; nearest-first.
    r_small = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=5,
                     filter={"keep": "yes"})
    # top_k > total_matching -> exactly the 12 matches.
    r_all = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=50,
                   filter={"keep": "yes"})
    assert r_small.status_code == 200 and r_all.status_code == 200, r_all.text

    # Confirm the cached index is genuinely FLAT (not IVF): the flat branch
    # of `_ivf_search_params` (search_params is None) is what is exercised.
    index = list(v1_query._SHARD_CACHE.values())[0][0]
    try:
        faiss.extract_index_ivf(index)
        is_ivf = True
    except Exception:
        is_ivf = False
    assert not is_ivf, "tiny dataset should yield a flat index, not IVFFlat"

    small = r_small.json()["matches"]
    assert len(small) == 5  # min(5, 12)
    assert all(m["metadata"].get("keep") == "yes" for m in small)
    assert [m["id"] for m in small] == ["v0", "v2", "v4", "v6", "v8"]
    all_matches = r_all.json()["matches"]
    assert len(all_matches) == 12  # min(50, 12) == total matching
    assert {m["id"] for m in all_matches} == {f"v{i}" for i in keep}


def test_ivf_filtered_query_topk_exceeds_shard_size(ivf_client):
    """On an IVF index, a filtered `top_k` larger than the whole shard
    returns exactly the filter-matching count — no error, no `-1` padding."""
    client = ivf_client
    s = _signup(client)
    n_total = 600
    match_idx = {int(round(i * (n_total - 1) / 20)) for i in range(21)}
    records = [
        {
            "id": f"v{i}",
            "values": [float(i), 0.0, 0.0, 0.0],
            "metadata": {"keep": "yes" if i in match_idx else "no"},
        }
        for i in range(n_total)
    ]
    _make_indexed_dataset(client, s["token"], dimension=4, records=records)

    import faiss
    import services.query_api.v1_query as v1_query

    # top_k far exceeds the 600-vector shard — must not error or pad with -1.
    r = _query(client, s["token"], vector=[0.0, 0.0, 0.0, 0.0], top_k=1000,
               filter={"keep": "yes"})
    assert r.status_code == 200, r.text

    index = list(v1_query._SHARD_CACHE.values())[0][0]
    ivf = faiss.extract_index_ivf(index)
    assert ivf.nlist > v1_query.DEFAULT_NPROBE, (
        f"test needs nlist > {v1_query.DEFAULT_NPROBE}; got nlist={ivf.nlist}"
    )
    matches = r.json()["matches"]
    # Exactly the 21 filter-matching vectors, no -1-derived padding entries.
    assert len(matches) == 21
    assert {m["id"] for m in matches} == {f"v{i}" for i in sorted(match_idx)}
    assert all(m["metadata"].get("keep") == "yes" for m in matches)
