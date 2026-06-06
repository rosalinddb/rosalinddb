"""Canonical metrics HTTP server for RosalindDB worker services.

This is the single home for the tiny `/metrics` + `/prometheus` + `/healthz`
HTTP handler that the long-running worker services (`validator_worker`,
`index_builder`, `ephemeral_runner`) each used to define as a byte-identical
copy. The three copies differed on EXACTLY two strings:

  * the ``service`` value in the ``/healthz`` JSON body
    (``validator_worker`` / ``index_builder`` / ``ephemeral_runner``); and
  * the Prometheus metric-name prefix
    (``validator_`` / ``builder_`` / ``ephemeral_`` — note the builder uses
    ``builder_``, NOT ``index_builder_``).

Both are parameterised here so every service's wire output (the ``/healthz``
bytes and every exported Prometheus metric name) stays byte-for-byte identical
to its previous inline copy. Everything else — the 404 fallback, the
``application/json`` content type, the ``prometheus-client not installed`` 503,
the timer ``_count`` / ``_avg_ms`` derivation, suppressed access logging — is
unchanged.

Lives in ``adapters/metrics`` because it depends only on
``adapters.metrics.metrics.snapshot`` plus the stdlib and the optional
``prometheus_client`` — pure adapter-layer concerns, no service imports — so it
respects the one-way ``adapters`` -> (never) -> ``services`` rule.
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from adapters.metrics.metrics import snapshot


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoints.

    Subclasses set ``service_name`` (the ``/healthz`` ``service`` field) and
    ``metric_prefix`` (the Prometheus metric-name prefix). Use
    :func:`make_metrics_handler` to build a configured subclass.
    """

    #: ``/healthz`` ``service`` field — overridden per service.
    service_name: str = "unknown"
    #: Prometheus metric-name prefix — overridden per service.
    metric_prefix: str = ""

    def do_GET(self):
        """Handle GET requests for metrics."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(snapshot()).encode())
        elif self.path == "/prometheus":
            self._serve_prometheus()
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                ('{"status": "ok", "service": "%s"}' % self.service_name).encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_prometheus(self):
        """Serve Prometheus format metrics."""
        try:
            from prometheus_client import CollectorRegistry, generate_latest, Gauge
        except ImportError:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"prometheus-client not installed")
            return

        prefix = self.metric_prefix
        # HELP-text descriptor word. The original copies used the bare service
        # word (`validator` / `builder` / `ephemeral`) — i.e. the prefix with
        # its trailing underscore stripped — so the exported metric DESCRIPTIONS
        # stay byte-identical, not just the metric names.
        label = prefix.rstrip("_")
        reg = CollectorRegistry()
        snap = snapshot()
        counters = snap.get("counters", {})
        gauges = snap.get("gauges", {})
        timers = snap.get("timers", {})

        # Export counters as gauges
        for name, value in counters.items():
            g = Gauge(f"{prefix}{name}", f"{label} counter {name}", registry=reg)
            g.set(float(value))

        # Export gauges
        for name, value in gauges.items():
            g = Gauge(f"{prefix}{name}", f"{label} gauge {name}", registry=reg)
            g.set(float(value))

        # Export timer stats
        for name, values in timers.items():
            if values:
                count = len(values)
                avg_ms = (sum(values) / count) * 1000.0
                Gauge(f"{prefix}{name}_count", f"{label} timer {name} count", registry=reg).set(float(count))
                Gauge(f"{prefix}{name}_avg_ms", f"{label} timer {name} avg ms", registry=reg).set(float(avg_ms))

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(generate_latest(reg))

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def make_metrics_handler(service_name: str, metric_prefix: str):
    """Return a `MetricsHandler` subclass bound to a service's two strings.

    `service_name` is the ``/healthz`` ``service`` field and `metric_prefix` is
    the Prometheus metric-name prefix. The returned class is what `HTTPServer`
    instantiates per request.
    """
    return type(
        "MetricsHandler",
        (MetricsHandler,),
        {"service_name": service_name, "metric_prefix": metric_prefix},
    )


def start_metrics_server(service_name: str, metric_prefix: str, port: int):
    """Start the metrics HTTP server in a background daemon thread.

    Byte-for-byte equivalent to the inline `start_metrics_server()` each worker
    used to define: it binds ``0.0.0.0:<port>``, serves forever on a daemon
    thread, and prints the same startup line.
    """
    handler_cls = make_metrics_handler(service_name, metric_prefix)

    def run_server():
        server = HTTPServer(("0.0.0.0", port), handler_cls)
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"Metrics server started on port {port}")
