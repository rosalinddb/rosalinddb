"""Unit tests for the first-party mem0 RosalindDB adapter (method->endpoint
mapping, L2^2->similarity conversion, filter passthrough, OutputData shapes).

These run against a MOCKED RosalindDB REST client — no network, no server.

CI safety: the whole module is skipped when ``mem0`` is not installed (the
core CI installs only ``requirements.txt``, which does NOT include ``mem0ai``),
so ``pytest -m unit`` stays green without the optional dependency.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mem0")  # optional dep — skip the whole module without it

# The adapter lives outside the importable package tree (integrations/mem0/);
# put that dir on sys.path so `import rosalinddb` resolves.
_ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "integrations",
    "mem0",
)
if _ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _ADAPTER_DIR)

import rosalinddb as rb  # noqa: E402
from rosalinddb_client import VectorNotFoundError  # noqa: E402


@pytest.fixture
def store():
    """A RosalindDB adapter wired to a MagicMock client."""
    client = MagicMock()
    client.create_dataset.return_value = {"name": "col", "dimension": 4}
    s = rb.RosalindDB("col", 4, client=client)
    # Reset the create-on-init call so per-test assertions start clean.
    client.create_dataset.reset_mock()
    return s


# -- L2^2 -> similarity conversion -----------------------------------------


def test_l2_squared_to_similarity_exact_match_is_one():
    assert rb.l2_squared_to_similarity(0.0) == 1.0


def test_l2_squared_to_similarity_is_monotonic_decreasing():
    # Smaller distance MUST yield a strictly higher similarity.
    sims = [rb.l2_squared_to_similarity(d) for d in (0.0, 0.5, 1.0, 4.0, 100.0)]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 < s <= 1.0 for s in sims)
    assert sims[0] > sims[1] > sims[2] > sims[3] > sims[4]


def test_search_converts_distance_to_similarity_and_preserves_order(store):
    store.client.query.return_value = {
        "matches": [
            {"id": "a", "score": 0.0, "metadata": {"v": "x"}},
            {"id": "b", "score": 0.25, "metadata": {}},
            {"id": "c", "score": 9.0, "metadata": {}},
        ],
        "mode": "hot",
    }
    res = store.search("q", [0.1, 0.2, 0.3, 0.4], top_k=3)
    assert [r.id for r in res] == ["a", "b", "c"]
    assert res[0].score == 1.0  # exact match
    # Higher-is-better: nearer (smaller L2^2) -> higher similarity.
    assert res[0].score > res[1].score > res[2].score
    assert all(0.0 < r.score <= 1.0 for r in res)


# -- method -> endpoint mapping --------------------------------------------


def test_create_col_calls_create_dataset_and_ignores_distance(store):
    store.create_col("col2", 8, distance="cosine")
    store.client.create_dataset.assert_called_once_with("col2", 8)


def test_create_col_is_idempotent_on_dataset_exists(store):
    err = rb.RosalindDB.__init__  # noqa: F841 - keep import warm
    exc = type("E", (Exception,), {})()
    exc.code = "dataset_exists"
    store.client.create_dataset.side_effect = exc
    # Must NOT raise.
    store.create_col("col", 4)


def test_insert_maps_to_ndjson_upsert(store):
    store.insert(
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        payloads=[{"data": "p1"}, {"data": "p2"}],
        ids=["id1", "id2"],
    )
    args = store.client.upsert.call_args
    assert args.args[0] == "col"
    records = args.args[1]
    assert records == [
        {"id": "id1", "values": [1, 0, 0, 0], "metadata": {"data": "p1"}},
        {"id": "id2", "values": [0, 1, 0, 0], "metadata": {"data": "p2"}},
    ]


def test_update_reupserts_with_passed_vector_and_payload(store):
    store.update("id1", vector=[1, 1, 1, 1], payload={"data": "new"})
    rec = store.client.upsert.call_args.args[1][0]
    assert rec == {"id": "id1", "metadata": {"data": "new"}, "values": [1, 1, 1, 1]}


def test_update_preserves_metadata_when_payload_omitted(store):
    store.client.get.return_value = {"id": "id1", "metadata": {"data": "old"}}
    store.update("id1", vector=[2, 2, 2, 2])
    rec = store.client.upsert.call_args.args[1][0]
    assert rec["metadata"] == {"data": "old"}
    assert rec["values"] == [2, 2, 2, 2]


def test_metadata_only_update_preserves_embedding_no_zero_vector(store):
    # vector=None must NOT clobber the embedding: the adapter reads it back via
    # include_values=true and re-upserts the REAL vector unchanged.
    store.client.get.return_value = {
        "id": "id1",
        "metadata": {"data": "old", "user_id": "u1"},
        "embedding": [0.1, 0.2, 0.3, 0.4],
    }
    store.update("id1", payload={"data": "new"})
    # Must request the stored values.
    assert store.client.get.call_args.kwargs.get("include_values") is True
    rec = store.client.upsert.call_args.args[1][0]
    # The real embedding is re-upserted unchanged — never a zero placeholder.
    assert rec["values"] == [0.1, 0.2, 0.3, 0.4]
    assert rec["values"] != [0.0, 0.0, 0.0, 0.0]
    # Metadata is last-write-wins merged over the existing.
    assert rec["metadata"] == {"data": "new", "user_id": "u1"}


def test_metadata_only_update_none_payload_keeps_existing_metadata(store):
    store.client.get.return_value = {
        "id": "id1",
        "metadata": {"data": "keep"},
        "embedding": [1.0, 0.0, 0.0, 0.0],
    }
    store.update("id1")  # vector=None, payload=None -> pure no-op-ish refresh
    rec = store.client.upsert.call_args.args[1][0]
    assert rec["values"] == [1.0, 0.0, 0.0, 0.0]
    assert rec["metadata"] == {"data": "keep"}


def test_metadata_only_update_consolidated_vector_raises_not_zeros(store):
    # Cold-only id: include_values returns no embedding. The adapter must REFUSE
    # (raise) rather than write a placeholder vector.
    store.client.get.return_value = {"id": "id1", "metadata": {"data": "x"}}
    with pytest.raises(ValueError, match="consolidated vector requires passing vector"):
        store.update("id1", payload={"data": "y"})
    store.client.upsert.assert_not_called()


def test_metadata_only_update_unknown_id_raises(store):
    store.client.get.side_effect = VectorNotFoundError("not_found", "gone")
    with pytest.raises(ValueError):
        store.update("missing", payload={"data": "y"})
    store.client.upsert.assert_not_called()


def test_delete_maps_to_delete_endpoint(store):
    store.delete("id9")
    store.client.delete.assert_called_once_with("col", "id9")


def test_get_returns_outputdata(store):
    store.client.get.return_value = {"id": "id1", "metadata": {"k": "v"}}
    out = store.get("id1")
    assert out.id == "id1"
    assert out.payload == {"k": "v"}
    assert out.score is None


def test_get_missing_returns_none(store):
    store.client.get.side_effect = VectorNotFoundError("not_found", "gone")
    assert store.get("missing") is None


def test_list_returns_double_wrapped_outputdata(store):
    store.client.list.return_value = {
        "vectors": [
            {"id": "a", "metadata": {"u": 1}},
            {"id": "b", "metadata": {}},
        ],
        "next_cursor": None,
    }
    result = store.list(filters={"user_id": "u1"}, top_k=50)
    # mem0 unwraps one level: list(...)[0] -> the rows.
    assert isinstance(result, list) and len(result) == 1
    rows = result[0]
    assert [r.id for r in rows] == ["a", "b"]
    assert all(r.score is None for r in rows)
    assert rows[0].payload == {"u": 1}
    # limit (top_k) is forwarded as the list `limit`.
    assert store.client.list.call_args.kwargs["limit"] == 50


def test_list_cols_maps_to_list_datasets(store):
    store.client.list_datasets.return_value = [{"name": "a"}, {"name": "b"}]
    assert store.list_cols() == ["a", "b"]


def test_col_info_maps_to_get_dataset(store):
    store.client.get_dataset.return_value = {"name": "col", "row_count": 3}
    assert store.col_info() == {"name": "col", "row_count": 3}


def test_delete_col_maps_to_delete_dataset(store):
    store.delete_col()
    store.client.delete_dataset.assert_called_once_with("col")


def test_reset_deletes_then_recreates(store):
    store.reset()
    store.client.delete_dataset.assert_called_once_with("col")
    store.client.create_dataset.assert_called_once_with("col", 4)


# -- filter passthrough -----------------------------------------------------


def test_search_passes_flat_filters_through(store):
    store.client.query.return_value = {"matches": [], "mode": "hot"}
    store.search("q", [0, 0, 0, 0], top_k=5, filters={"user_id": "u1", "agent_id": "a1"})
    assert store.client.query.call_args.kwargs["filter"] == {
        "user_id": "u1",
        "agent_id": "a1",
    }


def test_list_passes_flat_filters_through(store):
    store.client.list.return_value = {"vectors": [], "next_cursor": None}
    store.list(filters={"run_id": "r1"})
    assert store.client.list.call_args.kwargs["filter"] == {"run_id": "r1"}


def test_empty_filter_becomes_none(store):
    store.client.query.return_value = {"matches": [], "mode": "hot"}
    store.search("q", [0, 0, 0, 0], filters={})
    assert store.client.query.call_args.kwargs["filter"] is None


def test_none_filter_values_are_dropped(store):
    store.client.query.return_value = {"matches": [], "mode": "hot"}
    store.search("q", [0, 0, 0, 0], filters={"user_id": "u1", "agent_id": None})
    assert store.client.query.call_args.kwargs["filter"] == {"user_id": "u1"}


# -- keyword search unsupported --------------------------------------------


def test_keyword_search_returns_none_not_raises(store):
    # mem0's Memory.search calls keyword_search unconditionally and guards with
    # `if result is not None`, so it MUST return None (not raise) — a raise would
    # crash every Memory.search against this adapter.
    assert store.keyword_search("hello") is None
    assert store.keyword_search("hi", top_k=3, filters={"user_id": "u1"}) is None


def test_search_batch_loops_over_search(store):
    store.client.query.return_value = {
        "matches": [{"id": "a", "score": 0.0, "metadata": {}}],
        "mode": "hot",
    }
    results = store.search_batch(
        ["q1", "q2"], [[0, 0, 0, 0], [1, 1, 1, 1]], top_k=1
    )
    assert len(results) == 2
    assert results[0][0].id == "a"
    assert store.client.query.call_count == 2
