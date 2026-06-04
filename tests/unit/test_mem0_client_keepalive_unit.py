"""Unit tests for the urllib *fallback* transport of the RosalindDB REST client.

These exercise the path taken when the optional ``requests`` dependency is NOT
importable (``_HAVE_REQUESTS is False``). The optimisation under test (#23) is
HTTP keep-alive: the fallback must keep ONE persistent
``http.client.HTTPConnection`` / ``HTTPSConnection`` and reuse it across
sequential requests instead of opening a fresh TCP connection per call, while
reconnecting transparently if the connection is dropped.

CI safety: this suite imports ``rosalinddb_client`` *directly* (the REST client
has no ``mem0`` dependency) and forces the stdlib fallback, so it runs in the
core CI even though ``mem0ai`` is absent — it is intentionally NOT gated by
``importorskip("mem0")``.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

_ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "integrations",
    "mem0",
)
if _ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _ADAPTER_DIR)

import rosalinddb_client as rc  # noqa: E402


class _FakeResponse:
    """Stands in for ``http.client.HTTPResponse`` (status + readable body)."""

    def __init__(self, status, payload):
        self.status = status
        self._body = (
            json.dumps(payload).encode("utf-8") if payload is not None else b""
        )

    def read(self):
        return self._body


class _FakeConnection:
    """A scripted stand-in for ``http.client.HTTPConnection``.

    Records every ``request(...)`` and replays a *shared* class-level queue of
    responses (so the Nth request gets the Nth response regardless of which
    connection serves it — mirroring a real server). Tracks ``close`` calls so
    tests can assert the connection is reused (not re-created) across requests.
    The class-level ``instances`` list lets a test count how many connections
    were *constructed* over a run.
    """

    instances: list["_FakeConnection"] = []
    # Tests set this before constructing the client; it is a single shared queue
    # popped across all connections, so reconnects continue the script.
    next_responses: list = []

    def __init__(self, host, port=None, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests = []
        self.closed = False
        type(self).instances.append(self)

    def request(self, method, url, body=None, headers=None):
        self.requests.append(
            {"method": method, "url": url, "body": body, "headers": headers}
        )

    def getresponse(self):
        return type(self).next_responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture
def force_fallback(monkeypatch):
    """Force the urllib/http.client fallback path on for the test."""
    monkeypatch.setattr(rc, "_HAVE_REQUESTS", False)


def _install_fake_conn(monkeypatch, responses, scheme="http"):
    """Patch http.client.HTTPConnection/HTTPSConnection with the fake and reset."""
    import http.client

    _FakeConnection.instances = []
    _FakeConnection.next_responses = responses
    target = "HTTPSConnection" if scheme == "https" else "HTTPConnection"
    monkeypatch.setattr(http.client, target, _FakeConnection)
    return _FakeConnection


def _client(base_url="http://test:8080"):
    c = rc.RosalindDBClient(base_url=base_url, token="rb_live_x")
    c._session = None  # belt-and-suspenders: ensure the fallback is taken
    return c


# -- connection reuse (the core #23 fix) ------------------------------------


def test_fallback_reuses_single_connection_across_requests(force_fallback, monkeypatch):
    responses = [
        _FakeResponse(200, {"datasets": []}),
        _FakeResponse(200, {"datasets": []}),
        _FakeResponse(200, {"datasets": []}),
    ]
    fake = _install_fake_conn(monkeypatch, responses)
    client = _client()

    client.list_datasets()
    client.list_datasets()
    client.list_datasets()

    # Exactly ONE connection constructed across three sequential requests.
    assert len(fake.instances) == 1
    # ...and it serviced all three requests.
    assert len(fake.instances[0].requests) == 3


def test_fallback_https_uses_https_connection(force_fallback, monkeypatch):
    responses = [_FakeResponse(200, {"datasets": []})]
    fake = _install_fake_conn(monkeypatch, responses, scheme="https")
    client = _client(base_url="https://secure:8443")

    client.list_datasets()

    assert len(fake.instances) == 1
    assert fake.instances[0].host == "secure"
    assert fake.instances[0].port == 8443


def test_fallback_sends_keep_alive_header(force_fallback, monkeypatch):
    responses = [_FakeResponse(200, {"datasets": []})]
    fake = _install_fake_conn(monkeypatch, responses)
    client = _client()

    client.list_datasets()

    headers = fake.instances[0].requests[0]["headers"]
    # Header keys may be normalised; compare case-insensitively.
    lower = {k.lower(): v for k, v in headers.items()}
    assert lower.get("connection", "").lower() == "keep-alive"


# -- reconnect on a dropped connection --------------------------------------


def test_fallback_reconnects_after_dropped_connection(force_fallback, monkeypatch):
    """A dropped persistent connection is transparently re-established once."""
    import http.client

    _FakeConnection.instances = []

    class _DropOnceConnection(_FakeConnection):
        def request(self, method, url, body=None, headers=None):
            super().request(method, url, body=body, headers=headers)
            # The first connection's first reuse raises as if the server closed
            # the socket; subsequent (reconnected) connections behave normally.
            if len(type(self).instances) == 1 and len(self.requests) == 2:
                raise http.client.RemoteDisconnected("peer closed")

    _DropOnceConnection.next_responses = [
        _FakeResponse(200, {"datasets": ["a"]}),  # 1st request, conn #1
        # 2nd request on conn #1 raises before getresponse; conn #2 serves it:
        _FakeResponse(200, {"datasets": ["b"]}),
    ]
    monkeypatch.setattr(http.client, "HTTPConnection", _DropOnceConnection)
    client = _client()

    out1 = client.list_datasets()
    out2 = client.list_datasets()

    assert out1 == ["a"]
    assert out2 == ["b"]
    # Two connections total: the original plus one transparent reconnect.
    assert len(_DropOnceConnection.instances) == 2
    # The first (broken) connection was closed during recovery.
    assert _DropOnceConnection.instances[0].closed is True


# -- preserved semantics: responses & errors --------------------------------


def test_fallback_maps_error_envelope(force_fallback, monkeypatch):
    responses = [
        _FakeResponse(404, {"error": {"code": "dataset_not_found", "message": "no"}})
    ]
    _install_fake_conn(monkeypatch, responses)
    client = _client()
    with pytest.raises(rc.DatasetNotFoundError) as exc:
        client.get_dataset("missing")
    assert exc.value.code == "dataset_not_found"
    assert exc.value.status == 404


def test_fallback_request_shape_is_preserved(force_fallback, monkeypatch):
    responses = [_FakeResponse(200, {"accepted": 1, "rejected": 0, "errors": []})]
    fake = _install_fake_conn(monkeypatch, responses)
    client = _client()

    client.upsert("ds", [{"id": "a", "values": [1, 0], "metadata": {}}])

    req = fake.instances[0].requests[0]
    assert req["method"] == "POST"
    # The path (not the full origin) is what http.client sends on the line; the
    # absolute-URI form is also acceptable. Either must carry the route.
    assert req["url"].endswith("/v1/datasets/ds/vectors")
    headers = {k.lower(): v for k, v in req["headers"].items()}
    assert headers["content-type"] == "application/x-ndjson"
    assert headers["authorization"] == "Bearer rb_live_x"
    assert req["body"].decode().strip() == json.dumps(
        {"id": "a", "values": [1, 0], "metadata": {}}
    )


def test_fallback_transport_error_wraps_socket_failure(force_fallback, monkeypatch):
    import http.client

    class _BoomConnection(_FakeConnection):
        def request(self, method, url, body=None, headers=None):
            raise OSError("connection refused")

    _BoomConnection.instances = []
    _BoomConnection.next_responses = []
    monkeypatch.setattr(http.client, "HTTPConnection", _BoomConnection)
    client = _client()

    with pytest.raises(rc.TransportError) as exc:
        client.get_dataset("d")
    assert exc.value.code == "transport_error"


# -- thread-safety ----------------------------------------------------------


def test_fallback_is_thread_safe_and_still_reuses_connection(force_fallback, monkeypatch):
    """Concurrent callers share one locked connection without racing or error."""
    import threading

    # One response per request; threads each issue several requests.
    n_threads, per_thread = 4, 5
    responses = [
        _FakeResponse(200, {"datasets": []})
        for _ in range(n_threads * per_thread)
    ]
    fake = _install_fake_conn(monkeypatch, responses)
    client = _client()

    errors: list = []
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # maximise contention
        try:
            for _ in range(per_thread):
                client.list_datasets()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Still exactly one connection despite concurrent use (lock-serialised).
    assert len(fake.instances) == 1
    assert len(fake.instances[0].requests) == n_threads * per_thread
