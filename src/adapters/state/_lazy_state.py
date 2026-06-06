"""Lazy forwarding accessor for the ``adapters.state.state`` facade.

The split state submodules (``pooling``, ``migrations``, ``catalog``,
``generations``, ``quota``) and the ``adapters.recall`` sub-adapter own no
process-wide state themselves — the mutable globals (``_MEMORY_MODE``, the
connection pools, the ``_MEM_*`` stores, the migration flags) live in
``adapters.state.state`` and are reached via ``_state.X`` at *call* time.

Binding that with a plain ``import adapters.state.state as _state`` at module top
used to create an order-sensitive import cycle: ``state.state`` re-exports these
submodules at the END of its own module, so importing a submodule *first* in a
fresh interpreter hit a partially-initialised module
(``ImportError: cannot import name ... from partially initialized module``).
Every real entrypoint reaches the tier via ``adapters.state.state`` so it never
fired in production, but it was a latent footgun.

This proxy removes the cycle by deferring the import to first *attribute* access.
Because the submodules only ever touch ``_state.X`` at call time (never at import
time), nothing resolves the facade while a submodule is still importing, so
importing any submodule first is now safe (in either order). ``monkeypatch``
and ``importlib.reload(state)`` are honoured because every read/write/delete
resolves the live module from ``sys.modules`` on each access (no cached ref).

Note: ``_state`` is a *forwarding proxy object*, NOT the module itself. Attribute
get/set/del forward correctly, but it is not a ``ModuleType`` — do not
type-introspect it (``isinstance(_state, ModuleType)``, ``inspect.getmodule``),
compare its identity, or ``importlib.reload(_state)``. Use ``adapters.state.state``
directly if you need the real module object. The submodules only ever do
``_state.X`` (get/set), which is exactly what this supports.
"""
from __future__ import annotations


class _LazyStateModule:
    """Attribute proxy that resolves to ``adapters.state.state`` on each access.

    Reads (``_state.X``), writes (``_state.X = v``) and deletes (``del _state.X``)
    all forward to the live facade module, looked up fresh from ``sys.modules``
    each time (a cheap dict hit after first import — the module is not
    re-executed), so reloads/monkeypatches are always observed.
    """

    __slots__ = ()

    def __getattr__(self, name):
        import adapters.state.state as _s

        return getattr(_s, name)

    def __setattr__(self, name, value):
        import adapters.state.state as _s

        setattr(_s, name, value)

    def __delattr__(self, name):
        import adapters.state.state as _s

        delattr(_s, name)


# The singleton imported as ``_state`` across the state submodules.
state = _LazyStateModule()
