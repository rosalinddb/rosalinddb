"""Unit tests for the async bulk-ingest import-job flow.

Covers the `POST /v1/datasets/{name}/imports*` HTTP surface and the validator
worker's `process_import` path, all on the `memory://` storage + state
adapters so the suite stays hermetic (no Docker, no network).

The flow under test:

  create   -> POST /v1/datasets/{name}/imports        (201, awaiting_upload)
  upload   -> client "uploads" through the presigned target (memory:// fake)
  complete -> POST .../imports/{id}/complete           (202, validating)
  drain    -> run the validator + builder synchronously
  status   -> GET .../imports/{id}                     (200, completed)

The presigned upload for `memory://` is a faithful fake: `presign_put`
returns a `{url, method}` whose `url` is the `memory://...` object key. There
is no `fields` dict — a presigned PUT has no upload policy. The test helper
`_upload` writes the bytes straight through the storage adapter, exactly as a
browser PUT to MinIO/R2 would land the object. The bulk-upload size cap
(`content-length-range` on the old presigned POST) is now re-homed to the
import worker, which `head`s the staged object — see
`test_oversized_upload_rejected_by_worker`.
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


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh FastAPI TestClient with reset in-memory state + memory:// landing."""
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.delenv("RB_TEST_VECTOR_QUOTA", raising=False)
    # The vector-quota subsystem is opt-in (`RB_ENABLE_QUOTAS`). Several tests
    # in this module assert the 429 admission/settlement responses, so the
    # fixture turns it on for every test here.
    monkeypatch.setenv("RB_ENABLE_QUOTAS", "true")

    from adapters.storage import storage as storage_mod
    storage_mod.memory_reset()

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


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_dataset(client, token, name="bulk", dim=4):
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dim})
    assert r.status_code == 201, r.text


def _upload(target: dict, data: bytes) -> None:
    """Faithful client-side 'upload' through a presigned-PUT target.

    For the `memory://` fake the `url` is the object key. A presigned PUT has
    no form fields — the client just PUTs the raw bytes — so we write the
    bytes straight through the storage adapter exactly as a browser PUT to
    MinIO/R2 would land the object.
    """
    from adapters.storage import storage as storage_mod

    assert target["method"] == "PUT"
    storage_mod.write_bytes(target["url"], data)


def _ndjson(rows) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def _parquet_bytes(rows, dim) -> bytes:
    """Build a RosalindDB internal-landing-schema Parquet blob."""
    ids = [r["id"] for r in rows]
    vectors = [np.array(r["values"], dtype=np.float32) for r in rows]
    metas = [r.get("metadata") or {} for r in rows]
    vec = pa.FixedSizeListArray.from_arrays(
        pa.array(np.concatenate(vectors)) if vectors else pa.array([], type=pa.float32()),
        dim,
    )
    table = pa.table({"id": ids, "values": vec, "metadata": metas})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _drain():
    """Run validator + builder synchronously over the queued messages.

    Mirrors the production wiring: the validator's `process_import` advances
    the job, the builder's `run_once` builds the shard, and `finalize_import`
    flips the job to `completed` — exactly what `index_builder.main_loop`
    does on a `DATASET_READY` carrying an `import_id`.
    """
    from adapters.queue.queue import consume
    from services.validator_worker.run import (
        finalize_import, process_import, process_uri,
    )
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


# --- create ---------------------------------------------------------------


def test_create_import_returns_usable_presigned_target(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    r = client.post(
        "/v1/datasets/bulk/imports",
        headers=_auth(s["token"]),
        json={"format": "ndjson"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["import_id"].startswith("imp_")
    assert body["dataset"] == "bulk"
    assert body["status"] == "awaiting_upload"
    assert body["format"] == "ndjson"
    assert body["error_mode"] == "continue"
    assert body["max_bad_records"] is None
    up = body["upload"]
    assert up["method"] == "PUT"
    assert up["url"]
    # A presigned PUT carries no multipart form fields; it does carry the
    # exact Content-Type the URL is signed for.
    assert "fields" not in up
    assert up["content_type"] == "application/octet-stream"
    assert up["max_bytes"] > 0
    assert up["expires_at"]
    # The target must be usable: an upload through it lands in storage.
    _upload(up, _ndjson([{"id": "r0", "values": [1, 2, 3, 4]}]))


def test_create_import_bad_format_400(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    r = client.post(
        "/v1/datasets/bulk/imports",
        headers=_auth(s["token"]),
        json={"format": "csv"},
    )
    assert r.status_code == 400, r.text


def test_create_import_bad_error_mode_400(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    r = client.post(
        "/v1/datasets/bulk/imports",
        headers=_auth(s["token"]),
        json={"format": "ndjson", "error_mode": "explode"},
    )
    assert r.status_code == 400, r.text


def test_create_import_unknown_dataset_404(client):
    s = _signup(client)
    r = client.post(
        "/v1/datasets/missing/imports",
        headers=_auth(s["token"]),
        json={"format": "ndjson"},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- complete -------------------------------------------------------------


def test_complete_fails_if_nothing_uploaded(client):
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


def test_complete_unknown_import_404(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    r = client.post(
        "/v1/datasets/bulk/imports/imp_nope/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 404, r.text


def test_complete_twice_409(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    _upload(imp["upload"], _ndjson([{"id": "r0", "values": [1, 2, 3, 4]}]))
    r1 = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r1.status_code == 202, r1.text
    r2 = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r2.status_code == 409, r2.text


# --- happy path: NDJSON ---------------------------------------------------


def test_ndjson_import_happy_path(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    rows = [{"id": f"r{i}", "values": [0.1 * i, 0.2, 0.3, 0.4]} for i in range(5)]
    _upload(imp["upload"], _ndjson(rows))
    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "validating"

    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "completed", st
    assert st["records_processed"] == 5
    assert st["records_accepted"] == 5
    assert st["records_rejected"] == 0
    assert st["percent_complete"] == 100
    assert st["rejected_records_url"] is None
    assert st["error_message"] is None
    assert st["completed_at"]


def test_oversized_upload_rejected_by_worker(client, monkeypatch):
    """The re-homed size cap: a presigned-PUT URL cannot enforce a size limit
    server-side, so the import worker `head`s the staged object and fails the
    job when it exceeds `IMPORT_MAX_BYTES`."""
    import services.validator_worker.run as validator
    monkeypatch.setattr(validator, "IMPORT_MAX_BYTES", 64)

    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    # 4 KiB payload against a 64-byte cap. The PUT itself succeeds (no policy);
    # the worker rejects it.
    rows = [{"id": f"r{i}", "values": [0.1, 0.2, 0.3, 0.4]} for i in range(200)]
    _upload(imp["upload"], _ndjson(rows))
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


def test_failed_import_deletes_staged_object(client, monkeypatch):
    """A failed import deletes its staged upload object directly (Nit 1).

    `_fail` deletes the staged raw upload itself rather than leaving it for
    `index_builder`'s post-build landing sweep — which never runs for a tenant
    whose first/only import fails, so the object would otherwise orphan in
    object storage indefinitely. Holds for ALL failure paths; an oversized
    import is the case exercised here.
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
    rows = [{"id": f"r{i}", "values": [0.1, 0.2, 0.3, 0.4]} for i in range(200)]
    _upload(imp["upload"], _ndjson(rows))

    job = validator.get_import_job_by_id(imp["import_id"])
    upload_uri = job["upload_uri"]
    assert storage_mod.exists(upload_uri), "staged object should exist post-upload"

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


# --- happy path: Parquet --------------------------------------------------


def test_parquet_import_happy_path(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    rows = [{"id": f"p{i}", "values": [float(i), 0.2, 0.3, 0.4], "metadata": {"k": "v"}}
            for i in range(6)]
    _upload(imp["upload"], _parquet_bytes(rows, dim=4))
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
    assert st["records_accepted"] == 6
    assert st["records_rejected"] == 0


# --- bad records: continue mode ------------------------------------------


def test_continue_mode_drops_bad_records(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson", "error_mode": "continue"},
    ).json()
    body = b"\n".join([
        json.dumps({"id": "ok1", "values": [1, 2, 3, 4]}).encode(),
        b"{not json",
        json.dumps({"id": "ok2", "values": [5, 6, 7, 8]}).encode(),
        json.dumps({"id": "baddim", "values": [1, 2, 3]}).encode(),
    ]) + b"\n"
    _upload(imp["upload"], body)
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "completed", st
    assert st["records_processed"] == 4
    assert st["records_accepted"] == 2
    assert st["records_rejected"] == 2
    assert st["rejected_records_url"] is not None


def test_max_bad_records_exceeded_fails(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson", "error_mode": "continue", "max_bad_records": 1},
    ).json()
    body = b"\n".join([
        json.dumps({"id": "ok1", "values": [1, 2, 3, 4]}).encode(),
        b"{not json",
        json.dumps({"id": "baddim", "values": [1, 2, 3]}).encode(),
    ]) + b"\n"
    _upload(imp["upload"], body)
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert st["error_message"]


# --- abort mode -----------------------------------------------------------


def test_abort_mode_fails_on_first_bad_record(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson", "error_mode": "abort"},
    ).json()
    body = b"\n".join([
        json.dumps({"id": "ok1", "values": [1, 2, 3, 4]}).encode(),
        json.dumps({"id": "baddim", "values": [1, 2, 3]}).encode(),
        json.dumps({"id": "ok2", "values": [5, 6, 7, 8]}).encode(),
    ]) + b"\n"
    _upload(imp["upload"], body)
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert st["error_message"]
    # Nothing indexed: dataset never folded the batch in.
    ds = client.get("/v1/datasets/bulk", headers=_auth(s["token"])).json()
    assert ds["row_count"] == 0, ds


# --- list -----------------------------------------------------------------


def test_list_imports_newest_first(client):
    s = _signup(client)
    _make_dataset(client, s["token"])
    ids = []
    for _ in range(3):
        imp = client.post(
            "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
            json={"format": "ndjson"},
        ).json()
        ids.append(imp["import_id"])
    r = client.get("/v1/datasets/bulk/imports", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    listed = [i["import_id"] for i in r.json()["imports"]]
    assert listed == list(reversed(ids))


# --- tenant isolation -----------------------------------------------------


def test_tenant_cannot_see_or_complete_others_import(client):
    a = _signup(client, email="a@example.com")
    b = _signup(client, email="b@example.com")
    _make_dataset(client, a["token"], name="bulk")
    _make_dataset(client, b["token"], name="bulk")
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(a["token"]),
        json={"format": "ndjson"},
    ).json()
    # B cannot GET A's import.
    r = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(b["token"])
    )
    assert r.status_code == 404, r.text
    # B cannot complete A's import.
    r = client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(b["token"]),
    )
    assert r.status_code == 404, r.text
    # B's list does not include A's import.
    listed = client.get("/v1/datasets/bulk/imports", headers=_auth(b["token"])).json()
    assert listed["imports"] == []


# --- quota: admission -----------------------------------------------------


def test_quota_admission_429_when_already_at_cap(client, monkeypatch):
    monkeypatch.setenv("RB_TEST_VECTOR_QUOTA", "5")
    s = _signup(client, email="capped@example.com")
    _make_dataset(client, s["token"])
    # Consume the whole quota directly so the tenant sits at the cap.
    import adapters.state.state as state_mod
    ok, _ = state_mod.try_consume_vectors(s["tenant"]["id"], 5)
    assert ok
    r = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    )
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "vector_quota_exceeded"


# --- quota: settlement ----------------------------------------------------


def test_parquet_import_indexes_exactly_once(client):
    """A Parquet import must not be double-indexed.

    The raw `upload.parquet` is staged OUTSIDE the dataset landing prefix; only
    the validator's produced landing part lives under it. The index builder's
    recursive landing scan must therefore see exactly ONE `.parquet` part for
    the import and add exactly `row_count` vectors — not twice that.
    """
    import services.index_builder.run as builder
    from services.validator_worker.run import process_import
    from adapters.queue.queue import consume
    from adapters.landing.parquet_reader import list_landing_parts

    s = _signup(client)
    _make_dataset(client, s["token"])
    tenant = s["tenant"]["id"]
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    rows = [{"id": f"p{i}", "values": [float(i), 0.2, 0.3, 0.4], "metadata": {"k": "v"}}
            for i in range(6)]
    _upload(imp["upload"], _parquet_bytes(rows, dim=4))
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )

    # Validate the import, then assert the dataset landing prefix the builder
    # scans contains exactly ONE parquet part (the produced landing part), not
    # the raw upload too.
    msg = consume("VALIDATE_DATASET", block=False)
    process_import(msg["import_id"])

    landing_prefix = f"memory://rosalinddb/landing/{tenant}/bulk"
    parts = list_landing_parts(landing_prefix)
    assert len(parts) == 1, f"expected exactly 1 landing part, got {parts}"
    assert "imports/" in parts[0] and parts[0].endswith("/landing/part-0001.parquet")
    # The raw staged upload must NOT live under the landing prefix.
    assert not any("upload.parquet" in p for p in parts)

    added = builder.run_once(msg["dataset"], msg["tenant"])
    assert added == 6, f"expected 6 vectors added, got {added} (double-indexed?)"
    assert builder._LAST_BUILD["parts_read"] == 1
    assert builder._LAST_BUILD["vectors_added"] == 6


def test_unhandled_exception_during_import_leaves_job_failed(client, monkeypatch):
    """An unhandled exception while processing an import is terminal.

    `process_import` itself only flips the job to `failed` for errors it
    catches internally. An UNHANDLED crash must still end the job `failed`
    (never stuck in `validating`/`indexing`) — that is the `main_loop`
    catch-all / `fail_import` guarantee.
    """
    import services.validator_worker.run as validator

    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    _upload(imp["upload"], _ndjson([{"id": "r0", "values": [1, 2, 3, 4]}]))
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )

    # Force an UNHANDLED crash deep inside processing.
    def _boom(*a, **k):
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(validator, "_dataset_dimension", _boom)

    # Mirror `main_loop`: process_import raises, the catch-all flips to failed.
    from adapters.queue.queue import consume
    msg = consume("VALIDATE_DATASET", block=False)
    try:
        validator.process_import(msg["import_id"])
    except Exception as exc:  # noqa: BLE001
        validator.fail_import(msg["import_id"], f"validator crashed: {exc}")

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert st["error_message"], st


def test_malformed_parquet_with_columns_present_is_rejected(client):
    """A Parquet whose columns exist but whose schema is wrong is rejected.

    A `values` column typed `list<string>` passes the old shallow check (the
    `id`/`values` columns are present) but is not a valid landing file. It
    must fail the import, not be passed through to the index builder.
    """
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()

    # Build a Parquet with id + values columns present, but `values` is a
    # list<string> rather than list<float>.
    bad = pa.table({
        "id": ["a", "b"],
        "values": pa.array([["x", "y", "z", "w"], ["1", "2", "3", "4"]],
                            type=pa.list_(pa.string())),
    })
    buf = io.BytesIO()
    pq.write_table(bad, buf)
    _upload(imp["upload"], buf.getvalue())
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


def test_parquet_with_unknown_extra_column_is_rejected(client):
    """A Parquet carrying an unexpected extra column is rejected."""
    s = _signup(client)
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "parquet"},
    ).json()
    rows = [{"id": f"p{i}", "values": [float(i), 0.2, 0.3, 0.4], "metadata": {"k": "v"}}
            for i in range(3)]
    base = _parquet_bytes(rows, dim=4)
    table = pq.read_table(io.BytesIO(base)).append_column(
        "surprise", pa.array(["x", "y", "z"])
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    _upload(imp["upload"], buf.getvalue())
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert "unexpected" in (st["error_message"] or "").lower()


def test_quota_settlement_aborts_over_quota_job(client, monkeypatch):
    monkeypatch.setenv("RB_TEST_VECTOR_QUOTA", "3")
    s = _signup(client, email="settle@example.com")
    _make_dataset(client, s["token"])
    imp = client.post(
        "/v1/datasets/bulk/imports", headers=_auth(s["token"]),
        json={"format": "ndjson"},
    ).json()
    # 5 valid records but the tenant's vector quota is only 3.
    rows = [{"id": f"r{i}", "values": [float(i), 0.2, 0.3, 0.4]} for i in range(5)]
    _upload(imp["upload"], _ndjson(rows))
    client.post(
        f"/v1/datasets/bulk/imports/{imp['import_id']}/complete",
        headers=_auth(s["token"]),
    )
    _drain()

    st = client.get(
        f"/v1/datasets/bulk/imports/{imp['import_id']}", headers=_auth(s["token"])
    ).json()
    assert st["status"] == "failed", st
    assert "quota" in (st["error_message"] or "").lower()
