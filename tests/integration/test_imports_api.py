"""Integration tests for the async bulk-ingest import-job flow.

These run against a *real* MinIO container (via ``testcontainers``) so the
presigned-PUT upload path is exercised end-to-end: the test does an actual
HTTP PUT to MinIO with the raw file as the request body, exactly as a browser
(or the bench client) would. The validator + index builder then run inline.

Covered: NDJSON happy path, Parquet happy path, complete-before-upload 400,
and the re-homed size cap (the import worker `head`s the staged object and
fails an oversized job — a presigned PUT URL, unlike presigned POST, cannot
reject an oversized upload server-side).

Note — MinIO supports both presigned POST and presigned PUT, so this suite
cannot exercise the gap where a backend lacks presigned POST support. Verify
against such a backend separately.
"""
from __future__ import annotations

import importlib
import io
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient with per-test MinIO landing/index prefixes."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    monkeypatch.delenv("RB_TEST_VECTOR_QUOTA", raising=False)

    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY"):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS",
                 "_MEM_API_KEYS_BY_HASH", "_MEM_DATASETS", "_MEM_IMPORTS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_dataset(client, token, name="bulk", dim=4):
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dim})
    assert r.status_code == 201, r.text


def _put_to_minio(target: dict, data: bytes) -> int:
    """Do a real HTTP PUT to MinIO with the raw file as the request body.

    A presigned PUT has no multipart form and no fields — the client just PUTs
    the bytes. This mirrors exactly what a browser (or the bench client) does
    against S3 / MinIO / Cloudflare R2.
    """
    assert target["method"] == "PUT"
    # The presigned URL is signed for an exact `Content-Type`; the client must
    # send precisely that header or MinIO/S3/R2 answer `403`.
    resp = requests.put(
        target["url"],
        data=data,
        headers={"Content-Type": target["content_type"]},
        timeout=30,
    )
    return resp.status_code


def _ndjson(rows) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def _parquet_bytes(rows, dim) -> bytes:
    ids = [r["id"] for r in rows]
    vectors = [np.array(r["values"], dtype=np.float32) for r in rows]
    metas = [r.get("metadata") or {} for r in rows]
    vec = pa.FixedSizeListArray.from_arrays(np.concatenate(vectors), dim)
    table = pa.table({"id": ids, "values": vec, "metadata": metas})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _drain():
    from adapters.queue.queue import consume
    from services.validator_worker.run import finalize_import, process_import, process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            if msg.get("import_id"):
                process_import(msg["import_id"])
            else:
                process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])
        if msg.get("import_id"):
            finalize_import(msg["import_id"])


def test_ndjson_import_end_to_end_via_minio(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    rows = [{"id": f"r{i}", "values": [0.1 * i, 0.2, 0.3, 0.4]} for i in range(8)]
    status = _put_to_minio(imp["upload"], _ndjson(rows))
    assert status in (200, 204), status

    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 202, r.text

    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "completed", st
    assert st["records_accepted"] == 8
    assert st["records_rejected"] == 0


def test_parquet_import_end_to_end_via_minio(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    rows = [{"id": f"p{i}", "values": [float(i), 0.2, 0.3, 0.4], "metadata": {"n": i}}
            for i in range(8)]
    status = _put_to_minio(imp["upload"], _parquet_bytes(rows, dim=4))
    assert status in (200, 204), status

    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 202, r.text

    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "completed", st
    assert st["records_accepted"] == 8


def test_parquet_import_indexed_exactly_once_via_minio(client, s3_landing_prefix):
    """End-to-end on MinIO: a Parquet import is indexed exactly once.

    The raw `upload.parquet` is staged in the staging root, never the landing
    prefix, so the index builder's recursive landing scan sees exactly one
    `.parquet` part (the validator's produced landing part).
    """
    from adapters.queue.queue import consume
    from adapters.landing.parquet_reader import list_landing_parts
    from services.validator_worker.run import process_import
    import services.index_builder.run as builder

    s = _signup(client)
    _make_dataset(client, s["token"])
    tenant = s["tenant"]["id"]
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    rows = [{"id": f"p{i}", "values": [float(i), 0.2, 0.3, 0.4], "metadata": {"n": i}}
            for i in range(8)]
    status = _put_to_minio(imp["upload"], _parquet_bytes(rows, dim=4))
    assert status in (200, 204), status
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )

    msg = consume("VALIDATE_DATASET", block=False)
    process_import(msg["import_id"])

    landing_prefix = f"{s3_landing_prefix.rstrip('/')}/{tenant}/bulk"
    parts = list_landing_parts(landing_prefix)
    assert len(parts) == 1, f"expected exactly 1 landing part, got {parts}"
    assert not any("upload.parquet" in p for p in parts)

    added = builder.run_once(msg["dataset"], msg["tenant"])
    assert added == 8, f"expected 8 vectors, got {added} (double-indexed?)"


def test_malformed_parquet_rejected_via_minio(client):
    """A columns-present-but-malformed Parquet fails the import on MinIO."""
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    bad = pa.table({
        "id": ["a", "b"],
        "values": pa.array([["x", "y", "z", "w"], ["1", "2", "3", "4"]],
                            type=pa.list_(pa.string())),
    })
    buf = io.BytesIO()
    pq.write_table(bad, buf)
    status = _put_to_minio(imp["upload"], buf.getvalue())
    assert status in (200, 204), status
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert st["error_message"], st


def test_complete_before_upload_400(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 400, r.text


def test_oversized_upload_rejected_by_worker(client, monkeypatch):
    """The re-homed size cap, end-to-end on real MinIO.

    A presigned PUT carries no upload policy, so MinIO happily accepts the
    oversized object. The import worker then `head`s the staged object and
    fails the job because it exceeds `IMPORT_MAX_BYTES`. (MinIO supports
    presigned PUT, so this test passes here — backends that lack presigned
    POST must be verified separately.)
    """
    import services.validator_worker.run as validator
    monkeypatch.setattr(validator, "IMPORT_MAX_BYTES", 64)

    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    # 2 KiB payload against a 64-byte cap. The PUT itself succeeds.
    status = _put_to_minio(imp["upload"], b"x" * 2048)
    assert status in (200, 204), status

    # complete succeeds — the object is present — and enqueues validation.
    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 202, r.text

    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert "size limit" in (st["error_message"] or ""), st


def test_presigned_put_rejects_wrong_content_type(client):
    """The presigned PUT URL is signed for an EXACT `Content-Type`.

    This is the exact failure mode the presigned-POST -> presigned-PUT fix is
    about: the SigV4 signature covers the `Content-Type` header, so a PUT that
    sends any other (or no) `Content-Type` is rejected `403
    SignatureDoesNotMatch` by S3/MinIO/R2. This test pins the contract that the
    client MUST echo the signed `content_type` verbatim.
    """
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    target = imp["upload"]
    assert target["method"] == "PUT"
    assert target["content_type"] == "application/octet-stream"

    body = _ndjson([{"id": "r0", "values": [0.1, 0.2, 0.3, 0.4]}])

    # A WRONG Content-Type — the signature was computed for
    # `application/octet-stream`, so MinIO rejects the PUT with 403.
    wrong = requests.put(
        target["url"], data=body,
        headers={"Content-Type": "text/plain"}, timeout=30,
    )
    assert wrong.status_code == 403, wrong.text

    # A MISSING Content-Type (requests defaults to a non-matching value for a
    # raw-bytes body) is likewise rejected — not a 2xx.
    missing = requests.put(target["url"], data=body, timeout=30)
    assert missing.status_code == 403, missing.text

    # Sanity: echoing the signed Content-Type verbatim succeeds.
    ok = _put_to_minio(target, body)
    assert ok in (200, 204), ok


def test_failed_import_deletes_staged_object_via_minio(client, monkeypatch):
    """A failed import deletes its staged upload object directly (Nit 1).

    The staged raw upload must not orphan in object storage. `_fail` deletes
    it directly rather than relying on `index_builder`'s post-build landing
    sweep — which never runs for a tenant whose first/only import fails. Here
    an oversized import fails; afterwards the staged object is gone from MinIO.
    """
    import services.validator_worker.run as validator
    from adapters.storage import storage as storage_mod
    monkeypatch.setattr(validator, "IMPORT_MAX_BYTES", 64)

    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    status = _put_to_minio(imp["upload"], b"x" * 2048)
    assert status in (200, 204), status

    job = validator.get_import_job_by_id(imp["import_id"])
    upload_uri = job["upload_uri"]
    assert storage_mod.exists(upload_uri), "staged object should exist post-PUT"

    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert not storage_mod.exists(upload_uri), (
        "failed import must delete its staged upload object, not orphan it"
    )
