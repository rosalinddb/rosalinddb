"""Unit tests for the RosalindDB REST client used by the mem0 adapter.

Covers the ephemeral query poll loop, the v1 error-envelope -> exception
mapping, and NDJSON/filter request shaping — all against a stubbed transport
(no network).

CI safety: gated by ``importorskip("mem0")`` like the rest of the mem0
integration suite, so the core CI (no ``mem0ai``) skips it and stays green.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

pytest.importorskip("mem0")  # keep the mem0-integration suite uniformly gated

_ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "integrations",
    "mem0",
)
if _ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _ADAPTER_DIR)

import rosalinddb_client as rc  # noqa: E402


class _StubResp:
    def __init__(self, status, payload):
        self.status_code = status
        self.content = (
            json.dumps(payload).encode("utf-8") if payload is not None else b""
        )


class _StubSession:
    """Records requests and replays a scripted queue of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, data=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers})
        return self._responses.pop(0)


def _client(responses):
    client = rc.RosalindDBClient(base_url="http://test:8080", token="rb_live_x")
    client._session = _StubSession(responses)
    return client


# -- error envelope mapping -------------------------------------------------


def test_dataset_not_found_maps_to_exception():
    client = _client([_StubResp(404, {"error": {"code": "dataset_not_found", "message": "no"}})])
    with pytest.raises(rc.DatasetNotFoundError) as exc:
        client.get_dataset("missing")
    assert exc.value.code == "dataset_not_found"
    assert exc.value.status == 404


def test_vector_not_found_maps_to_exception():
    client = _client([_StubResp(404, {"error": {"code": "not_found", "message": "gone"}})])
    with pytest.raises(rc.VectorNotFoundError):
        client.get("ds", "id1")


def test_generic_error_envelope_preserves_code_and_details():
    client = _client(
        [_StubResp(400, {"error": {"code": "invalid_dimension", "message": "bad", "details": {"got": 3}}})]
    )
    with pytest.raises(rc.RosalindDBError) as exc:
        client.create_dataset("d", -1)
    assert exc.value.code == "invalid_dimension"
    assert exc.value.details == {"got": 3}


# -- request shaping --------------------------------------------------------


def test_upsert_sends_ndjson():
    client = _client([_StubResp(200, {"accepted": 2, "rejected": 0, "errors": []})])
    body = client.upsert("ds", [
        {"id": "a", "values": [1, 0], "metadata": {}},
        {"id": "b", "values": [0, 1], "metadata": {"k": "v"}},
    ])
    assert body["accepted"] == 2
    call = client._session.calls[0]
    assert call["headers"]["Content-Type"] == "application/x-ndjson"
    lines = call["data"].decode().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "a"


def test_list_url_encodes_filter():
    client = _client([_StubResp(200, {"vectors": [], "next_cursor": None})])
    client.list("ds", filter={"user_id": "u1"}, limit=10)
    url = client._session.calls[0]["url"]
    assert "filter=" in url
    assert "limit=10" in url


def test_query_passes_filter_and_top_k():
    client = _client([_StubResp(200, {"matches": [], "mode": "hot"})])
    client.query("ds", [0.1, 0.2], top_k=7, filter={"user_id": "u1"})
    sent = json.loads(client._session.calls[0]["data"])
    assert sent == {
        "dataset": "ds",
        "vector": [0.1, 0.2],
        "top_k": 7,
        "filter": {"user_id": "u1"},
    }


# -- ephemeral poll loop ----------------------------------------------------


def test_query_polls_ephemeral_until_ready():
    client = _client([
        _StubResp(200, {"matches": [], "mode": "ephemeral", "job_id": "job_1", "latency_ms": 1}),
        _StubResp(200, {"ready": False}),
        _StubResp(200, {"ready": True, "matches": [{"id": "a", "score": 0.0, "metadata": {}}], "mode": "ephemeral"}),
    ])
    result = client.query("ds", [0.1, 0.2], top_k=5, poll_interval=0)
    assert result["ready"] is True
    assert result["matches"][0]["id"] == "a"
    # POST /v1/query + 2 status polls.
    assert len(client._session.calls) == 3
    assert client._session.calls[1]["url"].endswith("/v1/query/status/job_1")


def test_query_times_out_when_never_ready():
    client = _client([
        _StubResp(200, {"matches": [], "mode": "ephemeral", "job_id": "job_2"}),
        _StubResp(200, {"ready": False}),
    ])
    with pytest.raises(rc.QueryTimeoutError):
        client.query("ds", [0.1, 0.2], poll_timeout=0, poll_interval=0)


def test_query_non_ephemeral_returns_immediately():
    client = _client([_StubResp(200, {"matches": [{"id": "x", "score": 1.5, "metadata": {}}], "mode": "hot"})])
    result = client.query("ds", [0.1, 0.2])
    assert result["mode"] == "hot"
    assert len(client._session.calls) == 1


def test_delete_returns_none_on_204():
    client = _client([_StubResp(204, None)])
    assert client.delete("ds", "id1") is None


def test_get_default_omits_include_values():
    client = _client([_StubResp(200, {"id": "id1", "metadata": {}})])
    client.get("ds", "id1")
    url = client._session.calls[0]["url"]
    assert "include_values" not in url


def test_get_include_values_adds_query_param_and_returns_embedding():
    client = _client(
        [_StubResp(200, {"id": "id1", "metadata": {"k": "v"}, "embedding": [1.0, 2.0]})]
    )
    out = client.get("ds", "id1", include_values=True)
    url = client._session.calls[0]["url"]
    assert "include_values=true" in url
    assert out["embedding"] == [1.0, 2.0]


# -- transport-layer errors -------------------------------------------------


def test_transport_error_wraps_connection_failure():
    import requests as _requests

    class _BoomSession:
        calls = []

        def request(self, *a, **k):
            raise _requests.exceptions.ConnectionError("refused")

    client = rc.RosalindDBClient(base_url="http://test:8080")
    client._session = _BoomSession()
    with pytest.raises(rc.TransportError) as exc:
        client.get_dataset("d")
    assert exc.value.code == "transport_error"
    # The original exception is chained.
    assert isinstance(exc.value.__cause__, _requests.exceptions.ConnectionError)
