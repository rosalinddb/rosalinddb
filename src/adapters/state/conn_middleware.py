from __future__ import annotations

"""Request-scoped Postgres connection ASGI middleware.

Each `state.pooled_conn()` block historically did its own
`getconn`/`commit`/`putconn`, so an HTTP request that called five state
functions cycled the pool five times. The k6 load sweeps showed that churn
dominating a request and exhausting the pool under burst.

`RequestScopedConnectionMiddleware` fixes that: per HTTP request it checks
ONE connection out of the pool, binds it to the `state._REQUEST_CONN`
contextvar, commits on a clean response / rolls back on an exception, and
returns the connection to the pool exactly once at request end. Every
`pooled_conn()` block inside the request transparently reuses that single
connection — one checkout, not N.

This is a **pure ASGI middleware**, NOT a `starlette.BaseHTTPMiddleware`.
That is deliberate and load-bearing: `BaseHTTPMiddleware` runs the
downstream app in a *separate* anyio task, so a `contextvars` value set in
the middleware would NOT be visible to the route handler. A pure ASGI
middleware calls the downstream app inside its OWN coroutine (same context),
so the `_REQUEST_CONN` contextvar bound here IS visible to the handler — and,
because `contextvars` are copied into the threads `asyncio.to_thread` /
`run_in_threadpool` spawn, it is visible inside off-loop sync state calls too.

Threading discipline:
  - The contextvar bind/unbind run in the middleware's own coroutine, so the
    binding reaches the handler and the `Token` is reset in the context it
    was set in.
  - The blocking work — the pool checkout (block-with-timeout, may sleep) and
    the commit/rollback + `putconn` — runs off the event loop via
    `run_in_threadpool`, so a burst on one request never stalls other
    coroutines.

Scope notes:
  - Only `http` scopes get a request connection. `websocket` / `lifespan`
    scopes pass straight through untouched.
  - Memory mode (`state._MEMORY_MODE`) has no pool: the checkout returns
    `None`, the contextvar is bound to `None`, and `pooled_conn()` falls
    through to its standalone (no-op) path — the middleware adds no behaviour
    and no measurable overhead in `memory://` tests.
"""

import json
from typing import Any, Awaitable, Callable

from starlette.concurrency import run_in_threadpool

from adapters.state import state as _state
from adapters.state.state import PoolCheckoutTimeout

# v1 error envelope code for a sustained pool exhaustion (HTTP 503). Kept in
# sync with the exception handler installed by `install_pool_exhaustion_handler`
# in `services/auth/auth.py`.
POOL_EXHAUSTED_CODE = "service_unavailable"
POOL_EXHAUSTED_MESSAGE = "Service temporarily unavailable, please retry"

# An ASGI app is `async (scope, receive, send) -> None`.
ASGIApp = Callable[
    [dict, Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


class RequestScopedConnectionMiddleware:
    """Bind one pooled Postgres connection per HTTP request.

    Wrap an ASGI app: `app.add_middleware(RequestScopedConnectionMiddleware)`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive, send) -> None:
        # Non-HTTP scopes (lifespan, websocket) never touch the request-scoped
        # connection — pass them straight through.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # Skip the request-scoped checkout for paths whose requests are long
        # but NOT Postgres-bound. Holding ONE pooled connection for the whole
        # request would pin it idle across the slow non-DB work and starve the
        # pool under load (a trace showed a 14.79s query holding its connection
        # the entire time while touching Postgres for ~1s of it):
        #
        #   /healthz                  - a pure liveness probe; must do NO DB
        #     round-trip (N4). A pool checkout here also risks a cascade: a
        #     saturated pool fails healthz, the platform restarts the machine,
        #     and the survivors take more load.
        #
        #   /v1/query, /v1/query/status/{id}  - on the CP these are reverse-
        #     proxied: the request is dominated by the multi-second CP->DP hop,
        #     with Postgres touched only briefly (resolve dp_pool, consume
        #     quota). On the DP the request is dominated by the FAISS search /
        #     shard download. Either way, request-scoping pins a connection
        #     idle for the slow part. Skipping it here means the handlers' brief
        #     DB calls each take a short *standalone* `pooled_conn()` checkout
        #     and release it immediately — so a connection is held only for the
        #     ~ms of real DB work, not the whole request, and the pool stays
        #     free during the hop. (`try_consume_query` already does this via
        #     `standalone=True`; the others fall through to the standalone path
        #     because no request connection is bound.)
        #
        #   /v1/datasets/{name}/vectors (POST)  - the NDJSON vector-upload path.
        #     The request is dominated by reading a multi-MB body, parsing every
        #     NDJSON line, and writing a ~9 MB landing object to object storage
        #     — all NOT Postgres work. Postgres is touched only briefly
        #     (`get_dataset` lookup, `try_consume_vectors` quota write, which
        #     already uses `standalone=True`). A k6 load sweep at OpenAI-
        #     embedding size (1536-dim, ~9 MB batches) showed 10 concurrent
        #     ingests pinning all 10 pooled connections idle across their slow
        #     S3 writes, so concurrent small `POST /v1/datasets` creates timed
        #     out the 2.5s pool checkout and returned 503 `service_unavailable`.
        #     Skipping the request scope here means the ingest's brief DB calls
        #     take short standalone checkouts and release immediately, keeping
        #     the pool free during the slow body-read + landing write.
        #
        # These paths pass straight through with no pool interaction.
        _path = scope.get("path") or ""
        _method = scope.get("method") or ""
        _is_vector_upload = (
            _method == "POST"
            and _path.startswith("/v1/datasets/")
            and _path.endswith("/vectors")
        )
        if (
            _path == "/healthz"
            or _path == "/v1/query"
            or _path.startswith("/v1/query/status/")
            or _is_vector_upload
        ):
            await self.app(scope, receive, send)
            return

        # Block-with-timeout pool checkout — blocking, possibly sleeping on a
        # saturated pool — runs off the event loop. In memory mode this is a
        # no-op and returns None.
        #
        # A `PoolCheckoutTimeout` here escapes BEFORE the app runs, so the
        # app's own exception handlers (which live inside `self.app`) cannot
        # see it — it would otherwise surface as a bare ASGI-server 500. The
        # middleware therefore emits the v1 503 envelope itself.
        try:
            conn = await run_in_threadpool(_state._checkout_request_conn)
        except PoolCheckoutTimeout:
            await _send_503(send)
            return

        # Bind in THIS coroutine's context so the route handler sees it.
        token = _state.bind_request_conn(conn)
        failed = False
        try:
            await self.app(scope, receive, send)
        except BaseException:
            # The handler raised (or a PoolCheckoutTimeout escaped). The
            # single request transaction must roll back; re-raise so the
            # app's exception handlers still run and shape the 5xx envelope.
            failed = True
            raise
        finally:
            # Commit/rollback + return-to-pool is blocking I/O — off-loop it.
            await run_in_threadpool(_state.finish_request_conn, conn, failed=failed)
            # Reset the contextvar in the same context it was set in.
            _state.unbind_request_conn(token)


async def _send_503(send) -> None:
    """Emit a v1 `{"error": {...}}` 503 envelope directly on the ASGI `send`.

    Used only for a `PoolCheckoutTimeout` raised by the middleware's own pool
    checkout — that happens before the app runs, so the app's exception
    handlers cannot shape it. A `PoolCheckoutTimeout` raised *inside* a route
    handler is shaped by the app-level handler instead (see
    `install_pool_exhaustion_handler`).
    """
    body = json.dumps(
        {"error": {"code": POOL_EXHAUSTED_CODE, "message": POOL_EXHAUSTED_MESSAGE}}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
