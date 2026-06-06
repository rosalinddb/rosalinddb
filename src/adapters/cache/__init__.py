"""Shared, service-agnostic cache primitives.

This package holds the catalog-listing cache and the local shard fetch /
download-coalescing logic that `services.query_api.v1_query` and
`services.ephemeral_runner.run` previously kept as two near-verbatim copies
(with a "the two copies MUST keep their public shape identical" sync comment).

The logic lives here exactly once. It depends only on the standard library
(plus, at call time, callables the caller injects), so it sits cleanly in the
``adapters`` layer and respects the one-way ``adapters`` -> (never) ->
``services`` import rule.

State ownership is deliberately PER CALLER, not global:

  * Each service constructs its own :class:`CatalogCache` instance, so the two
    services keep independent caches exactly as the two inline copies did
    (a test that monkeypatches one service's ``_now`` / ``list_shards`` must
    not perturb the other's cache).
  * :func:`ensure_cached` takes the in-flight registry, its lock, the coalesce
    deadline and the timeout class as parameters, so each service keeps owning
    its own module-level single-flight state (which tests reset / monkeypatch
    by module attribute).

The injected-callable shape (``list_shards_fn`` / ``now_fn`` passed per call)
exists so a caller's wrapper can resolve those names from its OWN module
namespace at call time — preserving the existing contract that
``monkeypatch.setattr(v1q, "list_shards", ...)`` / ``..., "_now", ...)`` retunes
the cache without touching this module.
"""

from adapters.cache.catalog_cache import CatalogCache
from adapters.cache.shard_fetch import ensure_cached

__all__ = ["CatalogCache", "ensure_cached"]
