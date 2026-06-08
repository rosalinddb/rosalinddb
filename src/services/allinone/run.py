from __future__ import annotations

"""All-in-one single-process RosalindDB entrypoint.

Runs the WHOLE database in one OS process with ZERO external infrastructure —
no Postgres, no Redis, no S3/MinIO, no pgvector. The eval / quickstart / laptop
deployment: `python -m services.allinone` (or the `rosalinddb` console script)
and you have a working vector DB with read-your-writes out of the box.

How it composes the full system in-process:

  1. EVAL-FRIENDLY ENV DEFAULTS (set FIRST, before any service/adapter import).
     `state.py`, `queue.py`, and `v1_query.py` capture some values at IMPORT time
     (`_MEMORY_MODE`, `_REDIS_URL` + the in-process queues, `CACHE_DIR` /
     `_MMAP_ENABLED` / shard-cache sizes). So the env MUST be set via
     `os.environ.setdefault` at the very top of this module, before the first
     `from services...` / `from adapters...` import, or memory mode / the
     in-process queue / recall-on would not take effect:
       - DATABASE_URL=memory://local            -> in-memory catalog + tenants
       - LANDING/INDEXES/STAGING_PREFIX=memory://… -> dict-backed object store
       - (no REDIS_URL)                          -> in-process queue.Queue fallback
       - RB_RECALL=true + RB_RECALL_BACKEND=memory (RB_RECALL_DSN left UNSET)
                                                 -> embedded numpy recall memtable
       - CACHE_DIR=<writable tmp dir>            -> FAISS shard cache (the default
                                                    /var/cache/shards is not
                                                    writable on a dev box)
     All via `setdefault`, so an operator can still override any of them.

  2. OBSERVABILITY pinned to `rosalinddb-allinone` (idempotent; first call wins),
     so the resolved service name is this app's even though importing
     source_registry would otherwise set its own.

  3. THE CONTROL-PLANE APP is reused wholesale from `services.source_registry.main`
     — CORS, the auth router, the request-scoped-connection middleware, the v1
     exception handlers + rate-limit handler, `/healthz`, the `/v1/datasets*` +
     vectors CRUD surface, and its startup hook that bootstraps the default
     tenant. `cp_app.py` mounts `query_proxy.router` on this same app (the CP->DP
     HTTP hop); the all-in-one instead mounts the REAL in-process query router
     (`v1_query.router`) so `POST /v1/query` runs FAISS + recall union IN-PROCESS
     — no proxy hop, no separate DP node.

  4. THE THREE PIPELINE WORKERS (validator, index builder, ephemeral runner) run
     on DAEMON THREADS in this same process, wired to the HTTP side through the
     in-process queue. The index builder hosts the reaper + idle-recall sweep and
     consumes DATASET_READY / DELETE_VECTORS / CONSOLIDATE, so the full
     recall->consolidated lifecycle runs locally.

Metrics servers (validator 9101, builder 9100, ephemeral 9102) bind locally and
are fine on a dev box; set `RB_ALLINONE_DISABLE_METRICS=1` to skip them if a port
clashes (e.g. a second all-in-one instance).
"""

import os
import tempfile
import threading

# --- 1. EVAL-FRIENDLY ENV DEFAULTS — set BEFORE any service/adapter import ----
#
# These MUST precede every `from services...` / `from adapters...` import below,
# because several modules capture these values at IMPORT time (see module
# docstring). `setdefault` leaves an operator-provided override untouched.
os.environ.setdefault("DATABASE_URL", "memory://local")
os.environ.setdefault("LANDING_PREFIX", "memory://rosalinddb/landing")
os.environ.setdefault("INDEXES_PREFIX", "memory://rosalinddb/indexes")
os.environ.setdefault("STAGING_PREFIX", "memory://rosalinddb/staging")
# Recall ON via the embedded in-process numpy memtable. RB_RECALL_DSN is left
# UNSET on purpose — the memory backend needs no recall store, and leaving it
# unset is what makes `auto` resolve to the embedded backend too.
os.environ.setdefault("RB_RECALL", "true")
os.environ.setdefault("RB_RECALL_BACKEND", "memory")
# FAISS shard cache. The production default (/var/cache/shards) is not writable
# on a dev box; the query path writes fetched memory:// shard bytes here before
# `read_index`. Use a per-process writable temp dir.
os.environ.setdefault("CACHE_DIR", os.path.join(tempfile.gettempdir(), "rosalinddb-allinone-cache"))

# --- 2. OBSERVABILITY — pin the service name before source_registry imports ----
from adapters.observability import init_observability  # noqa: E402

init_observability("rosalinddb-allinone")

# --- 3. THE CONTROL-PLANE APP (+ in-process query router) ---------------------
from adapters import config  # noqa: E402
from adapters.state import state as state_mod  # noqa: E402
from services.source_registry.main import app  # noqa: E402  (reused wholesale)
from services.query_api.v1_query import router as v1_query_router  # noqa: E402

# Mount the REAL in-process query router (NOT query_proxy). This replaces the
# CP->DP HTTP hop: POST /v1/query runs FAISS search + the recall union directly
# in this process.
app.include_router(v1_query_router)

__all__ = ["app", "main"]


# Guard so the worker daemon threads start exactly once per process even if
# `_start_workers()` / `main()` is called more than once (e.g. an embedding test
# harness importing the app).
_WORKERS_STARTED = threading.Event()
_WORKERS_LOCK = threading.Lock()


def _worker_target(main_loop):
    """Wrap a worker's `main_loop` so its metrics server can be skipped.

    Each worker's `main_loop` already calls `config.validate()` + `migrate()` +
    `install_signal_handlers()` + `start_metrics_server()`. `install_signal_handlers`
    is a no-op off the main thread (it swallows the `ValueError` from
    `signal.signal`), so the redundancy is harmless. The metrics servers bind
    fixed ports (9100/9101/9102); when `RB_ALLINONE_DISABLE_METRICS` is set we
    monkeypatch the worker module's `start_metrics_server` to a no-op for the
    duration of the loop so a port clash cannot crash the thread.
    """
    if not config.truthy(os.getenv("RB_ALLINONE_DISABLE_METRICS")):
        return main_loop

    module = __import__(main_loop.__module__, fromlist=["start_metrics_server"])

    def _runner():
        original = getattr(module, "start_metrics_server", None)
        if original is not None:
            module.start_metrics_server = lambda *a, **k: None  # type: ignore[assignment]
        try:
            main_loop()
        finally:
            if original is not None:
                module.start_metrics_server = original  # type: ignore[assignment]

    return _runner


def _start_workers() -> None:
    """Start the three pipeline workers on daemon threads (once per process).

    validator_worker, index_builder, and ephemeral_runner each run their blocking
    `main_loop` on a daemon thread, wired to the HTTP side via the in-process
    queue. Daemon threads die with the process, so SIGINT/SIGTERM to the main
    (uvicorn) thread tears everything down without a join.
    """
    with _WORKERS_LOCK:
        if _WORKERS_STARTED.is_set():
            return
        # Imported here (not at module top) so a caller that only wants `app`
        # (e.g. a TestClient boot test) does not pull in the worker modules until
        # workers are actually started.
        from services.validator_worker import run as validator_worker
        from services.index_builder import run as index_builder
        from services.ephemeral_runner import run as ephemeral_runner

        for run_module in (validator_worker, index_builder, ephemeral_runner):
            thread = threading.Thread(
                target=_worker_target(run_module.main_loop),
                name=f"allinone-{run_module.__name__}",
                daemon=True,
            )
            thread.start()

        # Start the RESULT_READY consumer the data plane uses. The query path's
        # ephemeral fallback (a dataset with no shard yet enqueues a
        # RUN_EPHEMERAL_QUERY) publishes its result on the RESULT_READY queue; a
        # `GET /v1/query/status/{job_id}` poll only finds it once this consumer
        # has stashed it in the shared `result_store`. Without it the fallback
        # hangs at `ready:false` forever. Mirrors `dp_app.on_start()` which calls
        # the same `start_result_consumer()` symbol. It is itself a daemon thread
        # started idempotently (guarded by its own module-level flag).
        from services.query_api.v1_query import start_result_consumer

        start_result_consumer()
        _WORKERS_STARTED.set()


class _AllInOneServer:
    """A `uvicorn.Server` subclass that also trips the workers' stop event.

    uvicorn installs its OWN SIGTERM/SIGINT handlers (setting `should_exit` to
    drain in-flight HTTP requests), but those never call
    `queue.shutdown.request_stop()`, so the worker daemon threads' consume loops
    (which poll `should_stop()` at the top of each iteration) are never told to
    exit. The daemon threads die with the process regardless, so this is not
    required for a clean exit, but tripping the event lets a worker holding a
    queue message ack/nack it promptly (avoiding a reaper reclaim) instead of
    being killed mid-consume.

    Rather than fight uvicorn's handler ordering, we OWN the signal handler:
    `install_signal_handlers()` is uvicorn's hook (called on the main thread when
    the server starts). We override it to install a handler that BOTH trips our
    queue stop event AND drives uvicorn's own `should_exit` shutdown, so a single
    SIGINT/SIGTERM tears down HTTP and the workers together with no hang.
    """

    def __init__(self, config):  # noqa: ANN001
        import uvicorn

        # Build via the real Server so all of uvicorn's machinery is intact.
        self._inner = uvicorn.Server(config)

    def install_signal_handlers(self) -> None:
        import signal

        from adapters.queue import shutdown as queue_shutdown

        def _handler(signum, _frame):  # noqa: ANN001
            queue_shutdown.request_stop()
            # Mirror uvicorn's own handler: ask the HTTP server to drain+exit.
            self._inner.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Off the main thread — nothing to wire (daemon threads die
                # with the process anyway).
                pass

    def run(self) -> None:
        # Substitute our signal-handler installer onto the inner server, then run.
        self._inner.install_signal_handlers = self.install_signal_handlers  # type: ignore[method-assign]
        self._inner.run()


def main() -> None:
    """Run the all-in-one server: bootstrap, start workers, serve HTTP.

    `migrate()` in memory mode is a pure bootstrap-default-tenant no-op (no
    schema); each worker `main_loop` also calls it (idempotent). uvicorn binds
    `0.0.0.0:$PORT` (default 8080). On SIGTERM/SIGINT the server drains HTTP and
    trips the workers' shutdown event (see `_AllInOneServer`).
    """
    import uvicorn

    config.validate()
    state_mod.migrate()
    _start_workers()
    server = _AllInOneServer(
        uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
    )
    server.run()
