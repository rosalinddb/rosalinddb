import importlib
import os
import threading
import time

import pytest

from adapters.queue.queue import publish


def _start_thread(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


@pytest.mark.skipif(
    os.environ.get("PY_MAJOR", str(os.sys.version_info.major)) == "3" and os.sys.version_info.minor >= 13,
    reason="skip integration on Python 3.13 due to optional wheels",
)
def test_end_to_end_hot_then_ephemeral(tmp_path, s3_landing_prefix, s3_indexes_prefix):
    """Full validator -> builder -> query path with tenant auth.

    Signs up, creates a dataset via the v1 surface, ingests NDJSON, runs
    the in-process pipeline, and confirms the index is queryable.
    Landing + index shards live in real MinIO.
    """
    os.environ["DIMENSION"] = "4"
    os.environ["LANDING_PREFIX"] = s3_landing_prefix
    os.environ["INDEXES_PREFIX"] = s3_indexes_prefix
    os.environ["CACHE_DIR"] = str(tmp_path / "cache")
    os.environ["TENANT_PREFIX"] = "true"
    os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")
    os.environ["DATABASE_URL"] = "memory://test"

    # Reload modules so the patched LANDING_PREFIX / INDEXES_PREFIX win.
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._MEM_DATASETS.clear()
    state_mod._MEM_TENANTS.clear()
    state_mod._MEM_TENANTS_BY_EMAIL.clear()
    state_mod._MEM_API_KEYS.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as source_main
    importlib.reload(source_main)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.query_api.main as query_main
    importlib.reload(query_main)
    import services.ephemeral_runner.run as ephemeral
    importlib.reload(ephemeral)

    from fastapi.testclient import TestClient
    c = TestClient(source_main.app)
    s = c.post("/auth/signup", json={"email": "e2e@example.com", "password": "password123"}).json()
    token = s["token"]
    c.post("/v1/datasets", headers={"Authorization": f"Bearer {token}"}, json={"name": "sample", "dimension": 4})
    body = '{"id":"a","values":[0,0,0,0]}\n'
    r = c.post(
        "/v1/datasets/sample/vectors",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text

    # Drain the queue synchronously.
    from adapters.queue.queue import consume
    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        validator.process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
        pending.append(msg)
    for msg in pending:
        builder.run_once(msg["dataset"], msg["tenant"])

    # Query the dataset via the legacy /query route (scoped to caller's tenant).
    qc = TestClient(query_main.app)
    r = qc.post(
        "/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"dataset": "sample", "vector": [0, 0, 0, 0], "top_k": 1, "mode": "ephemeral"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "ephemeral"
