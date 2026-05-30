"""Unit coverage for the ephemeral-query result store (`result_store`).

Hermetic — no Docker, no Redis. Exercises the in-process fallback used when
`REDIS_URL` is unset (the unit-test / single-process mode):

  - a stored result is readable back, with the v1 status shape preserved;
  - an unknown / not-yet-ready job_id reads back as None;
  - `clear()` drops everything;
  - `v1_query`'s `_RESULTS` alias still points at the fallback dict so the
    existing test fixtures' `v1_query._RESULTS.clear()` keeps working.

The genuine cross-replica behaviour fix (a result written by one `query_api`
instance is visible to a status poll on another) needs a real shared Redis and
lives in `tests/integration/test_result_store_redis.py`.
"""
from __future__ import annotations

import services.query_api.result_store as result_store
import services.query_api.v1_query as v1_query


def test_store_and_get_round_trips():
    """A stored result reads back with its fields intact."""
    result_store.clear()
    result_store.store_result(
        "job_abc",
        {"correlation_id": "job_abc", "matches": [{"id": "x"}], "latency_ms": 7},
    )
    res = result_store.get_result("job_abc")
    assert res is not None
    assert res["matches"] == [{"id": "x"}]
    assert res["latency_ms"] == 7


def test_unknown_job_reads_back_none():
    """An unknown / not-yet-ready job_id is absent — None, not an error."""
    result_store.clear()
    assert result_store.get_result("job_never_stored") is None


def test_clear_drops_everything():
    """`clear()` empties the store."""
    result_store.store_result("job_1", {"matches": []})
    result_store.clear()
    assert result_store.get_result("job_1") is None


def test_v1_query_results_alias_points_at_fallback():
    """`v1_query._RESULTS` is the same dict the fallback store writes to.

    The existing query/quota test fixtures reset the store via
    `v1_query._RESULTS.clear()`; that must keep emptying the store the
    `RESULT_READY` consumer writes to.
    """
    assert v1_query._RESULTS is result_store._RESULTS
    result_store.store_result("job_alias", {"matches": []})
    assert "job_alias" in v1_query._RESULTS
    v1_query._RESULTS.clear()
    assert result_store.get_result("job_alias") is None
