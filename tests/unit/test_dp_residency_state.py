"""`adapters/state/state.py` — residency registry functions.

The four state-adapter functions:

  - `register_dp_shard_warm(dp_id, shard_uri, warm_since, last_query_at)`
    — UPSERT. First write inserts; subsequent writes refresh `last_query_at`
    (and intentionally leave `warm_since` alone — it is set once on first
    admit and never moved, so an operator can read it as "how long has this
    DP held the shard cached").
  - `unregister_dp_shard_warm(dp_id, shard_uri)` — DELETE, idempotent.
  - `list_dp_residency_for_shard(shard_uri)` — routing primitive: given a
    shard, which DPs hold it warm?
  - `list_dp_residency_for_dp(dp_id)` — operator / observability primitive:
    what is this DP currently holding?

The tests run on the in-memory state adapter (the default for the `unit`
suite — `DATABASE_URL=memory://local` from `tests/conftest.py`). The PG
branch is exercised in the integration suite (out of scope here); the
memory adapter must mirror the PG semantics so the two branches stay
swappable.
"""
from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def state():
    """Fresh in-memory state module with no residency rows."""
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    # Defensive: the reload re-evaluates module-level dict defaults so the
    # store is empty, but a test that fails partway through could leave
    # rows behind for a sibling. Explicit clear for clarity.
    if hasattr(state_mod, "_MEM_DP_RESIDENCY"):
        state_mod._MEM_DP_RESIDENCY.clear()
    return state_mod


def test_register_inserts_first_row(state):
    """A first `register_dp_shard_warm` inserts the (dp_id, shard_uri) row.

    The row is visible to both lookup primitives immediately — no
    transaction the test has to commit.
    """
    state.register_dp_shard_warm(
        "dp-a", "memory://b/shard-1.bin",
        warm_since=100.0, last_query_at=100.0,
    )
    by_shard = state.list_dp_residency_for_shard("memory://b/shard-1.bin")
    assert by_shard == [("dp-a", 100.0, 100.0)]
    by_dp = state.list_dp_residency_for_dp("dp-a")
    assert by_dp == [("memory://b/shard-1.bin", 100.0, 100.0)]


def test_register_is_upsert_refreshing_last_query_at(state):
    """A second `register` for the same (dp_id, shard_uri) refreshes only `last_query_at`.

    `warm_since` is set once on first admit and is intentionally left
    alone on subsequent calls — that is the contract the operator reads as
    "how long has this DP held the shard". Refreshing it on every hit
    would defeat that purpose.
    """
    uri = "memory://b/shard-1.bin"
    state.register_dp_shard_warm("dp-a", uri, warm_since=100.0, last_query_at=100.0)
    state.register_dp_shard_warm("dp-a", uri, warm_since=200.0, last_query_at=205.0)

    rows = state.list_dp_residency_for_shard(uri)
    assert len(rows) == 1
    dp_id, warm_since, last_query_at = rows[0]
    assert dp_id == "dp-a"
    assert warm_since == 100.0  # NOT moved — first-write wins for warm_since
    assert last_query_at == 205.0  # refreshed by the second call


def test_unregister_removes_row(state):
    """`unregister_dp_shard_warm` removes the row; lookups see no entry."""
    uri = "memory://b/shard-1.bin"
    state.register_dp_shard_warm("dp-a", uri, warm_since=100.0, last_query_at=100.0)
    state.unregister_dp_shard_warm("dp-a", uri)

    assert state.list_dp_residency_for_shard(uri) == []
    assert state.list_dp_residency_for_dp("dp-a") == []


def test_unregister_is_idempotent(state):
    """Removing a row that does not exist is a clean no-op (no raise)."""
    # Never inserted — the call must succeed silently.
    state.unregister_dp_shard_warm("dp-ghost", "memory://b/nope.bin")
    # And removing the same row twice must also be safe.
    uri = "memory://b/shard-1.bin"
    state.register_dp_shard_warm("dp-a", uri, warm_since=10.0, last_query_at=10.0)
    state.unregister_dp_shard_warm("dp-a", uri)
    state.unregister_dp_shard_warm("dp-a", uri)
    assert state.list_dp_residency_for_shard(uri) == []


def test_list_for_shard_returns_all_dps(state):
    """A shard held warm by multiple DPs returns every (dp_id, warm_since, last_query_at).

    This is the routing primitive's primary read path: "given a shard the
    request needs, which DPs already have it cached?". Sort order is not
    part of the contract — the router picks a winner from the set.
    """
    uri = "memory://b/popular-shard.bin"
    state.register_dp_shard_warm("dp-a", uri, warm_since=10.0, last_query_at=10.0)
    state.register_dp_shard_warm("dp-b", uri, warm_since=20.0, last_query_at=25.0)
    state.register_dp_shard_warm("dp-c", uri, warm_since=30.0, last_query_at=35.0)

    rows = state.list_dp_residency_for_shard(uri)
    assert sorted(rows) == sorted([
        ("dp-a", 10.0, 10.0),
        ("dp-b", 20.0, 25.0),
        ("dp-c", 30.0, 35.0),
    ])


def test_list_for_dp_returns_all_shards(state):
    """A DP that holds multiple shards returns every (shard_uri, warm_since, last_query_at).

    The operator / observability primitive: "what is this DP currently
    holding?". Used for dashboards and the future ssd-cache admin
    surfaces.
    """
    state.register_dp_shard_warm("dp-a", "memory://b/s1.bin", warm_since=1.0, last_query_at=1.0)
    state.register_dp_shard_warm("dp-a", "memory://b/s2.bin", warm_since=2.0, last_query_at=2.0)
    state.register_dp_shard_warm("dp-a", "memory://b/s3.bin", warm_since=3.0, last_query_at=3.0)
    # A different DP must not bleed in.
    state.register_dp_shard_warm("dp-b", "memory://b/other.bin", warm_since=99.0, last_query_at=99.0)

    rows = state.list_dp_residency_for_dp("dp-a")
    assert sorted(rows) == sorted([
        ("memory://b/s1.bin", 1.0, 1.0),
        ("memory://b/s2.bin", 2.0, 2.0),
        ("memory://b/s3.bin", 3.0, 3.0),
    ])


def test_dp_id_is_part_of_primary_key(state):
    """Two DPs registering the SAME shard are independent rows.

    PK is `(dp_id, shard_uri)` — the same shard URI under two DPs is two
    rows, and unregistering one does not affect the other. If the PK
    were just `shard_uri`, dp-b's register would clobber dp-a's row and
    the routing primitive would lose dp-a.
    """
    uri = "memory://b/shared-shard.bin"
    state.register_dp_shard_warm("dp-a", uri, warm_since=10.0, last_query_at=10.0)
    state.register_dp_shard_warm("dp-b", uri, warm_since=20.0, last_query_at=20.0)
    # Unregister only dp-a.
    state.unregister_dp_shard_warm("dp-a", uri)

    rows = state.list_dp_residency_for_shard(uri)
    assert rows == [("dp-b", 20.0, 20.0)]


def test_lookups_return_empty_when_no_rows(state):
    """Lookups for a nonexistent shard / DP return an empty list, not None.

    The empty-list contract lets callers write `for dp_id, _, _ in
    list_dp_residency_for_shard(...)` without a None-check.
    """
    assert state.list_dp_residency_for_shard("memory://nope.bin") == []
    assert state.list_dp_residency_for_dp("dp-ghost") == []
