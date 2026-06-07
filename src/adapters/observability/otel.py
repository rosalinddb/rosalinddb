from __future__ import annotations

"""OpenTelemetry SDK bootstrap for the RosalindDB backend.

`init_observability(service_name)` is the single entrypoint each service calls
at startup. It wires a `TracerProvider`, `MeterProvider`, and `LoggerProvider`,
each with an OTLP/HTTP exporter, plus FastAPI / `requests` / `httpx` /
stdlib-logging auto-instrumentation.

Design goals:

  - **Vendor-neutral OTLP/HTTP** to a collector. Switching to Grafana Cloud
    later is just an `OTEL_EXPORTER_OTLP_ENDPOINT` change.
  - **Graceful failure.** Tests (and any dev run without a collector) must not
    hang or crash. The OTLP exporters get a short timeout and the SDK drops
    spans/metrics/logs silently when the endpoint is dead — the export happens
    on background threads, so a dead collector never blocks a request.
  - **Toggleable.** `OTEL_SDK_DISABLED=true` makes the whole thing a no-op:
    `init_observability` returns immediately and the metric/span helpers fall
    back to OpenTelemetry's built-in no-op API objects.
  - **Idempotent.** Calling `init_observability` twice (e.g. a single-process
    dev/test harness importing several service modules that each bootstrap)
    is safe — only the first call installs providers.

Env vars honoured:
  - `OTEL_SDK_DISABLED`            — `true` → no-op (default false).
  - `OTEL_EXPORTER_OTLP_ENDPOINT` — base OTLP/HTTP endpoint
                                     (default `http://localhost:4318`).
  - `OTEL_SERVICE_NAME`           — overrides the `service_name` argument.
  - `OTEL_EXPORTER_OTLP_TIMEOUT`  — per-export timeout in seconds (default 3).
"""

import logging
import threading

from adapters import config

# --- module state ---------------------------------------------------------

_LOCK = threading.Lock()
_INITIALIZED = False
_SERVICE_NAME = "rosalinddb"

# Short export timeout so a dead collector never stalls a flush. The exporters
# run on background threads (BatchSpanProcessor / PeriodicExportingMetricReader
# / BatchLogRecordProcessor), so this only bounds the worker thread, never a
# request path.
_DEFAULT_EXPORT_TIMEOUT_S = 3


def is_disabled() -> bool:
    """Return True when instrumentation should be a no-op.

    Honours the standard `OTEL_SDK_DISABLED` env var so tests can switch the
    whole SDK off without touching code.
    """
    return config.otel_sdk_disabled()


def is_enabled() -> bool:
    """Return True once the SDK providers have been installed."""
    return _INITIALIZED


def service_name() -> str:
    """Return the resolved service name for the current process."""
    return _SERVICE_NAME


def _otlp_endpoint() -> str:
    """Resolve the OTLP/HTTP base endpoint (no signal path suffix)."""
    return config.otel_exporter_otlp_endpoint()


def _export_timeout_s() -> int:
    return config.otel_exporter_otlp_timeout()


def init_observability(service_name: str) -> bool:
    """Configure the OpenTelemetry SDK for this process.

    Installs trace / metric / log providers with OTLP/HTTP exporters and
    enables `requests` + stdlib-logging auto-instrumentation. FastAPI apps are
    instrumented separately via `instrument_fastapi(app)`.

    Idempotent: a second call is a no-op. Safe to call from every service
    module's startup — in a single-process dev/test harness only the first wins.

    Returns True if the SDK is now active, False if disabled or if bootstrap
    failed (in which case the app keeps running uninstrumented).
    """
    global _INITIALIZED, _SERVICE_NAME

    resolved = config.otel_service_name() or service_name
    _SERVICE_NAME = resolved

    if is_disabled():
        # No-op mode: the OTel API ships no-op tracer/meter implementations,
        # so the rest of the codebase can call `get_tracer`/`get_meter`
        # unconditionally and simply produce nothing.
        logging.getLogger(__name__).debug("OTEL_SDK_DISABLED set — observability is a no-op")
        return False

    with _LOCK:
        if _INITIALIZED:
            return True
        try:
            _bootstrap(resolved)
        except Exception as exc:  # noqa: BLE001
            # Never let an observability bootstrap failure take down the app.
            logging.getLogger(__name__).warning("observability bootstrap failed: %s", exc)
            return False
        _INITIALIZED = True
        return True


def _bootstrap(resolved_service_name: str) -> None:
    """Build and register the three providers. Raises on hard failure."""
    from opentelemetry import metrics as _metrics_api
    from opentelemetry import trace as _trace_api
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    endpoint = _otlp_endpoint()
    timeout = _export_timeout_s()

    # Resource attributes — `service.name` + `service.namespace` on everything.
    resource = Resource.create(
        {
            "service.name": resolved_service_name,
            "service.namespace": "rosalinddb",
        }
    )

    # --- traces ------------------------------------------------------------
    span_exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        timeout=timeout,
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    _trace_api.set_tracer_provider(tracer_provider)

    # --- metrics -----------------------------------------------------------
    metric_exporter = OTLPMetricExporter(
        endpoint=f"{endpoint}/v1/metrics",
        timeout=timeout,
    )
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=config.otel_metric_export_interval(),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    _metrics_api.set_meter_provider(meter_provider)

    # --- logs --------------------------------------------------------------
    log_exporter = OTLPLogExporter(
        endpoint=f"{endpoint}/v1/logs",
        timeout=timeout,
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    # Bridge stdlib logging → OTLP logs. The JSON-stdout formatter (configured
    # by `logs.configure_json_logging`) handles human/Loki-readable output;
    # this handler additionally ships records as OTLP log signals so the
    # collector can route them to Loki with trace correlation.
    otlp_log_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(otlp_log_handler)

    # --- auto-instrumentation: requests + httpx + logging ------------------
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("requests instrumentation skipped: %s", exc)

    # `httpx` instrumentation. The CP→DP query proxy
    # (`services/query_api/query_proxy.py`) is an `httpx.AsyncClient`, NOT
    # `requests`, so `RequestsInstrumentor` above does not touch it. Without
    # this a proxied query would break the trace at the CP→DP hop: the DP span
    # would start a new, unparented trace. `HTTPXClientInstrumentor` injects
    # the W3C `traceparent` header on every outbound httpx request, so a query
    # stays one continuous CP→DP trace.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("httpx instrumentation skipped: %s", exc)

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        # Injects trace_id/span_id into stdlib LogRecords so any formatter
        # (including our JSON one) can emit them.
        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("logging instrumentation skipped: %s", exc)

    # Structured JSON logs to stdout (correlatable to traces by trace_id).
    from adapters.observability.logs import configure_json_logging

    configure_json_logging(resolved_service_name)

    # Quiet exporter failure noise. The contract requires the OTLP exporter to
    # "fail gracefully/quietly" when no collector is running.
    #
    # Two problems to solve here:
    #  1. The SDK logs the ConnectionError at ERROR on every failed batch.
    #  2. Worse: that ERROR record is itself a stdlib log record, so the OTLP
    #     `LoggingHandler` we attached above tries to *export it too* — a
    #     failure-feedback loop that amplifies the noise.
    # Fix: silence the whole `opentelemetry` logger subtree (set it to a level
    # above ERROR and stop it propagating to the root handlers). This drops
    # only OTel's *own internal* diagnostics; application logs — emitted on
    # other logger names — are unaffected and still reach stdout + OTLP.
    _otel_logger = logging.getLogger("opentelemetry")
    _otel_logger.setLevel(logging.CRITICAL)
    _otel_logger.propagate = False


def instrument_fastapi(app) -> None:
    """Attach FastAPI auto-instrumentation (HTTP server traces + metrics).

    No-op when the SDK is disabled or already-instrumented. Safe to call at
    import time from a service module — if `init_observability` has not run
    yet the instrumentation simply records against the no-op providers.
    """
    if is_disabled():
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # `instrument_app` is idempotent — it tags the app and skips re-wraps.
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("fastapi instrumentation skipped: %s", exc)
