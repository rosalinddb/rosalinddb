"""Unit coverage: the recall tier is OFF by default and never connects.

The recall tier ships behind `RB_RECALL_DSN` (and `RB_RECALL`),
default off. The headline safety property of this PR is that a flag-off deploy
behaves byte-identically to today: nothing connects to a recall store, and the
control-plane migrate path is unchanged. These hermetic tests assert exactly
that — no Docker, no pgvector, just the gate logic in the state adapter.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def state(monkeypatch):
    """Fresh state module with `RB_RECALL_DSN` explicitly cleared (recall tier off)."""
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    yield state_mod
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(state_mod)


def test_recall_dsn_unset_is_none(state):
    """`_recall_dsn()` is None when `RB_RECALL_DSN` is unset — the off signal."""
    assert state._recall_dsn() is None


def test_recall_dsn_blank_is_treated_as_off(state, monkeypatch):
    """A blank/whitespace `RB_RECALL_DSN` is treated as unset (cannot enable it).

    A compose default that resolves to empty must NOT silently turn the tier on.
    """
    monkeypatch.setenv("RB_RECALL_DSN", "")
    importlib.reload(state)
    assert state._recall_dsn() is None
    monkeypatch.setenv("RB_RECALL_DSN", "   ")
    importlib.reload(state)
    assert state._recall_dsn() is None


def test_migrate_recall_noop_and_never_connects(state, monkeypatch):
    """`migrate_recall()` returns False and opens no connection when the tier is off.

    `psycopg2.connect` is monkeypatched to explode — if `migrate_recall()` tried
    to reach a recall store it would raise; the no-op contract means it never does.
    """
    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("migrate_recall connected to a DB while recall tier off")

    monkeypatch.setattr(state.psycopg2, "connect", _boom)
    assert state.migrate_recall(force=True) is False


def test_recall_dsn_set_is_returned(state, monkeypatch):
    """When `RB_RECALL_DSN` IS set, `_recall_dsn()` returns it (the on signal).

    This only checks the gate flips on — it does NOT connect (no force/apply).
    """
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    importlib.reload(state)
    assert state._recall_dsn() == "postgresql://u:p@recall:5432/recall"
