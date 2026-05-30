"""Unit coverage for the unauthenticated `/healthz` liveness probes.

Every RosalindDB HTTP service exposes `/healthz`. It must be reachable
without credentials, return 200 with the shared
`{"status": "ok", "service": <name>}` body, and do no DB/storage I/O —
it is the cheap "is the process up and routing" check `make smoke` and any
post-deploy health gate hit first.
"""
from fastapi.testclient import TestClient

from services.query_api.main import app as query_app
from services.source_registry.main import app as source_registry_app


def test_query_api_healthz():
    """query_api `/healthz` → 200 with the shared shape, no auth header."""
    c = TestClient(query_app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "query_api"}


def test_source_registry_healthz():
    """The source_registry app `/healthz` → 200 with the shared shape.

    Regression guard: this app hosts the `/v1/datasets*`, `/v1/query`,
    `/v1/imports` and `/auth/*` routes and historically returned 404 for
    `/healthz`. It must answer the probe without credentials.

    The reported `service` is `control_plane`: this app IS the Control Plane
    surface (the CP reuses it wholesale — see `services/control_plane/cp_app.py`).
    The `source_registry` module path is an internal implementation detail.
    """
    c = TestClient(source_registry_app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "control_plane"}


def test_query_dp_healthz():
    """The Query-DP app `/healthz` → 200 with the shared shape.

    The DP app (`services/query_api/dp_app.py`) is the private data-plane
    process group. It must answer the liveness probe without credentials,
    with the shared `{"status", "service"}` body so a health gate treats it
    like every other RosalindDB service.
    """
    from services.query_api.dp_app import app as query_dp_app

    c = TestClient(query_dp_app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "query_dp"}
