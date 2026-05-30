"""Rendezvous-hash (HRW) DP routing in the CP.

Rendezvous hashing for CP→DP routing ships additively behind
`RB_ROUTING_RENDEZVOUS=true`. The existing static env-based map
(`QUERY_DP_URL` for the shared pool, `QUERY_DP_URL_<TENANT>` for
per-tenant overrides — one URL each, see the top-of-doc
"Today's routing is single-DP-per-pool" framing) remains the default.
When the flag is on AND the resolved pool config carries >1 URL
(comma-separated), the router picks the DP via Highest-Random-Weight (HRW)
hashing over the URL list, walking HRW rank order to skip unhealthy DPs.

This file covers the pure-function HRW primitive, the multi-URL parser, the
gated `resolve_dp_base_url` / `pick_dp_url` behaviour (including the
unhealthy-DP fallback chain and the all-unhealthy fall-back to the static
behaviour), and the strict default-off contract: with the flag unset a
comma-separated URL must produce a clear startup-style WARNING so an
operator misconfiguration surfaces immediately rather than silently
single-URL-ing the first segment.

Routing key: `f"{tenant}|{dataset}"`. The v1 simplification uses
`(tenant, dataset)` as the HRW key — same-dataset queries are stable per-DP
across version changes. The residency-hint override (via the
`dp_shard_residency` table) is not yet wired into routing.
"""
from __future__ import annotations

import logging
from collections import Counter

import pytest


# --- shared fixture: reset the module-global misconfig-warning dedup set --
#
# `_MULTI_URL_WARNED` is a process-local set inside `query_proxy` that
# dedupes the "comma-separated URL without RB_ROUTING_RENDEZVOUS" WARNING
# to one log line per (pool, env_key) pair. Without this autouse fixture,
# the first test that triggers the WARNING populates the set, and the
# second test that triggers it for the same key gets nothing — flaky
# across test order. The reset costs microseconds.


@pytest.fixture(autouse=True)
def _reset_multi_url_warned():
    import services.query_api.query_proxy as qp

    with qp._MULTI_URL_WARNED_LOCK:
        qp._MULTI_URL_WARNED.clear()
    yield
    with qp._MULTI_URL_WARNED_LOCK:
        qp._MULTI_URL_WARNED.clear()


# --- pure HRW primitive ---------------------------------------------------


def test_pick_dp_is_deterministic_for_fixed_inputs():
    """`pick_dp(key, urls)` returns the same URL on repeated calls.

    HRW is a pure function of (key, urls); a stable input must yield a
    stable output. This is the load-bearing property — same dataset always
    lands on the same DP.
    """
    import services.query_api.query_proxy as qp

    urls = [
        "http://dp-1:8090",
        "http://dp-2:8090",
        "http://dp-3:8090",
    ]
    key = "ten_alpha|dataset_books"
    first = qp._hrw_pick(key, urls)
    for _ in range(50):
        assert qp._hrw_pick(key, urls) == first


def test_pick_dp_returns_a_url_from_the_pool():
    """The HRW pick is always one of the configured URLs (never a fresh string)."""
    import services.query_api.query_proxy as qp

    urls = [
        "http://dp-1:8090",
        "http://dp-2:8090",
        "http://dp-3:8090",
    ]
    picked = qp._hrw_pick("ten_a|ds_x", urls)
    assert picked in urls


def test_pick_dp_url_order_independent():
    """HRW is order-independent — shuffling the URL list yields the same pick."""
    import services.query_api.query_proxy as qp

    urls = ["http://dp-1:8090", "http://dp-2:8090", "http://dp-3:8090"]
    reordered = ["http://dp-3:8090", "http://dp-1:8090", "http://dp-2:8090"]
    key = "ten_z|ds_q"
    assert qp._hrw_pick(key, urls) == qp._hrw_pick(key, reordered)


def test_pick_dp_distributes_roughly_uniformly_across_pool():
    """Over many distinct keys, each DP wins ~1/N of the picks.

    Rough fairness check: with N=4 DPs and 1000 keys, each DP should land
    in the [0.7 * 1/N, 1.3 * 1/N] band — well inside HRW's natural variance.
    """
    import services.query_api.query_proxy as qp

    urls = [f"http://dp-{i}:8090" for i in range(1, 5)]
    n_keys = 1000
    counts: Counter[str] = Counter(
        qp._hrw_pick(f"ten_x|ds_{i}", urls) for i in range(n_keys)
    )
    expected = n_keys / len(urls)
    lo, hi = expected * 0.7, expected * 1.3
    for url in urls:
        assert lo <= counts[url] <= hi, (
            f"{url} got {counts[url]} picks; expected {lo:.0f}..{hi:.0f}"
        )


def test_pick_dp_single_url_is_trivial():
    """A 1-element URL list always returns that URL (HRW of N=1)."""
    import services.query_api.query_proxy as qp

    urls = ["http://only-dp:8090"]
    for i in range(20):
        assert qp._hrw_pick(f"ten_a|ds_{i}", urls) == "http://only-dp:8090"


def test_removing_a_dp_only_reshuffles_about_one_over_n_keys():
    """The HRW minimal-disruption property — removing 1-of-N reshuffles ~1/N keys.

    This is the property that makes HRW worth its keep over a static modulo
    hash. When DP-4 is removed from a 4-DP pool, only the keys that were
    landing on DP-4 should move; the rest stay on the same DP. So ~25% of
    keys move, not 75%.
    """
    import services.query_api.query_proxy as qp

    urls_before = [f"http://dp-{i}:8090" for i in range(1, 5)]
    urls_after = urls_before[:-1]  # drop dp-4
    keys = [f"ten_x|ds_{i}" for i in range(1000)]

    moved = 0
    for key in keys:
        before = qp._hrw_pick(key, urls_before)
        after = qp._hrw_pick(key, urls_after)
        if before != after:
            moved += 1

    # Expected move rate ≈ 1/4 = 250/1000. Allow a generous band for variance:
    # 15%–35% (well above the 0% a perfect HRW would give for non-DP-4 keys
    # and well below the ~75% a naïve modulo hash would produce).
    assert 150 <= moved <= 350, (
        f"removing 1-of-4 DPs moved {moved}/1000 keys; expected ~250 "
        f"(15–35%); higher means HRW's minimal-disruption property is broken"
    )


def test_adding_a_dp_only_reshuffles_about_one_over_n_keys():
    """The HRW minimal-disruption property — adding 1-of-N reshuffles ~1/(N+1) keys.

    Symmetric to the remove case: adding DP-4 to a 3-DP pool should pull
    ~25% of keys onto the new DP, leaving the other ~75% where they were.
    """
    import services.query_api.query_proxy as qp

    urls_before = [f"http://dp-{i}:8090" for i in range(1, 4)]
    urls_after = urls_before + ["http://dp-4:8090"]
    keys = [f"ten_x|ds_{i}" for i in range(1000)]

    moved = 0
    for key in keys:
        before = qp._hrw_pick(key, urls_before)
        after = qp._hrw_pick(key, urls_after)
        if before != after:
            moved += 1

    # Expected move rate ≈ 1/4 = 250/1000.
    assert 150 <= moved <= 350, (
        f"adding 1-of-3 -> 4 DPs moved {moved}/1000 keys; expected ~250 "
        f"(15–35%)"
    )


# --- HRW rank order (for the unhealthy-DP fallback chain) -----------------


def test_pick_dp_ranked_returns_all_urls_in_stable_order():
    """`_hrw_rank(key, urls)` returns every URL once, ordered by HRW weight.

    The unhealthy-DP fallback walks this list: try rank 0; if it is sick,
    try rank 1; etc. So the function must be a permutation of the input.
    """
    import services.query_api.query_proxy as qp

    urls = [f"http://dp-{i}:8090" for i in range(1, 6)]
    key = "ten_alpha|ds_books"
    ranked = qp._hrw_rank(key, urls)
    assert sorted(ranked) == sorted(urls)
    # First element must agree with `_hrw_pick` — same algorithm, same key.
    assert ranked[0] == qp._hrw_pick(key, urls)


def test_pick_dp_ranked_is_deterministic():
    """Repeated `_hrw_rank` calls for the same key/urls return the same order."""
    import services.query_api.query_proxy as qp

    urls = [f"http://dp-{i}:8090" for i in range(1, 6)]
    key = "ten_x|ds_y"
    first = qp._hrw_rank(key, urls)
    for _ in range(20):
        assert qp._hrw_rank(key, urls) == first


# --- resolve_dp_base_url multi-URL parsing (flag on) ----------------------


def test_multi_url_config_with_flag_on_picks_via_hrw(monkeypatch):
    """With the flag on, a comma-separated `QUERY_DP_URL` is HRW-routed.

    `pick_dp_url(pool, routing_key)` is the new public entry point: it
    resolves the pool to the list of configured URLs, then either delegates
    to the static map (flag off / single URL) or runs HRW + health-fallback
    (flag on, >1 URL).
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    monkeypatch.setenv(
        "QUERY_DP_URL",
        "http://dp-1:8090,http://dp-2:8090,http://dp-3:8090",
    )
    # All DPs healthy by default — the resolver should pick whichever URL
    # `_hrw_pick` selects for this key.
    picked = qp.pick_dp_url("shared", "ten_a|ds_books")
    expected = qp._hrw_pick(
        "ten_a|ds_books",
        ["http://dp-1:8090", "http://dp-2:8090", "http://dp-3:8090"],
    )
    assert picked == expected


def test_multi_url_config_strips_whitespace_and_trailing_slashes(monkeypatch):
    """Whitespace and trailing slashes around each URL are normalised."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    monkeypatch.setenv(
        "QUERY_DP_URL",
        " http://dp-1:8090/ , http://dp-2:8090 , http://dp-3:8090/ ",
    )
    picked = qp.pick_dp_url("shared", "ten_x|ds_y")
    # No trailing slash on the returned URL, no leading whitespace.
    assert picked.startswith("http://")
    assert not picked.endswith("/")
    assert " " not in picked


def test_multi_url_dedicated_pool_picks_via_hrw(monkeypatch):
    """`'dedicated-<tenant>'` with a comma-separated env var is HRW-routed."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    monkeypatch.setenv(
        "QUERY_DP_URL_TEN_ABC",
        "http://ded-1:9000,http://ded-2:9000",
    )
    picked = qp.pick_dp_url("dedicated-ten_abc", "ten_abc|ds_x")
    assert picked in ("http://ded-1:9000", "http://ded-2:9000")


def test_single_url_with_flag_on_behaves_unchanged(monkeypatch):
    """A 1-element list with the flag on still returns that single URL.

    HRW of N=1 is trivial — the gate-on path with a single URL must keep
    identical behaviour to the flag-off / static-map path.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    monkeypatch.setenv("QUERY_DP_URL", "http://only-dp:8090")
    picked = qp.pick_dp_url("shared", "ten_a|ds_books")
    assert picked == "http://only-dp:8090"


# --- default-off rollback contract (flag unset) ---------------------------


def test_flag_off_single_url_returns_static_map(monkeypatch):
    """With the flag unset, the static map's behaviour is byte-identical.

    No HRW math runs; `pick_dp_url("shared", key)` must return the same
    URL `resolve_dp_base_url("shared")` returns today.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_ROUTING_RENDEZVOUS", raising=False)
    monkeypatch.setenv("QUERY_DP_URL", "http://dp.internal:9000")
    assert qp.pick_dp_url("shared", "ten_a|ds_x") == "http://dp.internal:9000"
    # And `resolve_dp_base_url` must keep returning the single URL unchanged.
    assert qp.resolve_dp_base_url("shared") == "http://dp.internal:9000"


def test_flag_off_multi_url_logs_warning(monkeypatch, caplog):
    """A comma-separated URL with the flag unset emits a clear WARNING.

    Operator-misconfiguration safety net: if someone sets the multi-URL
    config but forgets the gate flag, they see the mismatch immediately
    rather than silently routing every query to the first URL.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_ROUTING_RENDEZVOUS", raising=False)
    monkeypatch.setenv(
        "QUERY_DP_URL",
        "http://dp-1:8090,http://dp-2:8090,http://dp-3:8090",
    )
    with caplog.at_level(logging.WARNING, logger=qp.__name__):
        qp.pick_dp_url("shared", "ten_a|ds_books")

    warned = [
        rec
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and "RB_ROUTING_RENDEZVOUS" in rec.getMessage()
    ]
    assert warned, (
        "expected a WARNING mentioning RB_ROUTING_RENDEZVOUS when a "
        "comma-separated QUERY_DP_URL is configured but the gate flag is off"
    )


def test_flag_off_multi_url_still_returns_a_reachable_url(monkeypatch):
    """Even when misconfigured (multi-URL, flag off), the router returns SOME URL.

    Today the resolver returns the raw env-var contents — so a misconfigured
    comma-separated URL would today route to a literal `"http://dp-1:8090,
    http://dp-2:8090"` host, which is broken. The flag-off path must
    degrade gracefully to a single, reachable URL (the first segment) so
    queries keep flowing while the operator fixes the config.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_ROUTING_RENDEZVOUS", raising=False)
    monkeypatch.setenv(
        "QUERY_DP_URL",
        "http://dp-1:8090,http://dp-2:8090",
    )
    picked = qp.pick_dp_url("shared", "ten_a|ds_books")
    # Reachable URL — no comma in the host string.
    assert "," not in picked
    assert picked.startswith("http://")


# --- unhealthy-DP fallback (flag on) --------------------------------------


def test_unhealthy_top_rank_falls_to_next(monkeypatch):
    """When the HRW-elected DP is unhealthy, the next-rank DP wins.

    The router queries an `is_dp_healthy(url) -> bool` hook (defaults to
    True). When the top-rank DP reports unhealthy, the router walks HRW
    rank order and returns the first healthy URL.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    urls = ["http://dp-1:8090", "http://dp-2:8090", "http://dp-3:8090"]
    monkeypatch.setenv("QUERY_DP_URL", ",".join(urls))

    key = "ten_a|ds_books"
    ranked = qp._hrw_rank(key, urls)
    top = ranked[0]
    second = ranked[1]

    # Mark only the top-rank DP unhealthy.
    monkeypatch.setattr(qp, "_is_dp_healthy", lambda url: url != top)

    assert qp.pick_dp_url("shared", key) == second


def test_unhealthy_top_two_falls_to_third(monkeypatch):
    """Two-deep walk through HRW rank order to find a healthy DP."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    urls = ["http://dp-1:8090", "http://dp-2:8090", "http://dp-3:8090"]
    monkeypatch.setenv("QUERY_DP_URL", ",".join(urls))

    key = "ten_a|ds_books"
    ranked = qp._hrw_rank(key, urls)
    sick = {ranked[0], ranked[1]}
    expected = ranked[2]

    monkeypatch.setattr(qp, "_is_dp_healthy", lambda url: url not in sick)

    assert qp.pick_dp_url("shared", key) == expected


def test_all_unhealthy_falls_back_to_static_behaviour(monkeypatch):
    """All DPs unhealthy -> fall back to the static-map "best-effort even when sick".

    Today the static map returns the configured URL regardless of health —
    the "best-effort even when sick" pattern. When every DP in the
    rendezvous pool is sick, the router preserves that pattern by
    returning the first URL in the configured list (a deterministic,
    config-order choice, NOT an HRW pick that would change as keys change).
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_ROUTING_RENDEZVOUS", "true")
    urls = ["http://dp-1:8090", "http://dp-2:8090", "http://dp-3:8090"]
    monkeypatch.setenv("QUERY_DP_URL", ",".join(urls))

    monkeypatch.setattr(qp, "_is_dp_healthy", lambda url: False)

    # Every key falls back to the first configured URL — config order,
    # not HRW order, so the fallback is operator-predictable.
    for i in range(20):
        assert qp.pick_dp_url("shared", f"ten_x|ds_{i}") == urls[0]


def test_healthy_hook_defaults_to_true(monkeypatch):
    """`_is_dp_healthy` is currently a no-op returning True (no active health polling yet).

    The hook must default-True so HRW picks are honoured. Active /healthz
    polling that would feed `_is_dp_healthy` with real health data is not
    yet wired into routing.
    """
    import services.query_api.query_proxy as qp

    assert qp._is_dp_healthy("http://anything:1234") is True


# --- routing key shape ----------------------------------------------------


def test_routing_key_builder_uses_tenant_and_dataset():
    """The routing key is `f"{tenant}|{dataset}"`.

    Exposing the helper keeps the contract testable — callers that wire HRW
    into `_proxy` must agree on this key shape so same-dataset queries land
    on the same DP across version changes.
    """
    import services.query_api.query_proxy as qp

    assert qp._routing_key("ten_a", "books") == "ten_a|books"
    assert qp._routing_key("ten_x", "ds with spaces") == "ten_x|ds with spaces"
