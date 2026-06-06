"""OpenTelemetry observability adapter for RosalindDB.

This package is the new system of record for metrics, traces, and logs. It
configures the OpenTelemetry SDK to export OTLP/HTTP to a collector and
exposes:

  - `init_observability(service_name)` — one-call SDK bootstrap, invoked from
    each service entrypoint.
  - `metrics` — the custom business metric instruments (contract-pinned).
  - `tracing` — helpers for the manual pipeline spans.

The legacy `adapters.metrics` ad-hoc layer is left in place (harmless) but new
instrumentation goes through here.
"""

from adapters.observability.otel import init_observability, is_enabled

__all__ = ["init_observability", "is_enabled"]
