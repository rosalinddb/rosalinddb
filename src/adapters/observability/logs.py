from __future__ import annotations

"""Structured JSON logging to stdout, correlatable to traces.

`configure_json_logging(service_name)` installs a JSON formatter on the root
logger's stdout handler. Each line is a single JSON object with the contract
fields: `timestamp`, `level`, `service`, `message`, plus `trace_id` / `span_id`
when the record was emitted inside an active span, plus any contextual
key/values passed via `logging`'s `extra=`.

Trace correlation: the OTel `LoggingInstrumentor` (enabled in `otel.py`)
injects `otelTraceID` / `otelSpanID` onto every `LogRecord`. We read those and
also fall back to the live span context so a log line can always be joined to
its trace in Loki/Tempo by `trace_id`.

This is intentionally additive — it only touches the root logger's stream
handler formatting; it does not remove the OTLP log handler that `otel.py`
attaches, so logs reach both stdout (for `docker logs` / Loki via the
collector's filelog receiver) and the OTLP logs pipeline.
"""

import json
import logging
import sys
import time

from opentelemetry import trace as _trace_api

# Standard LogRecord attributes — anything NOT in this set that appears on a
# record was passed by the caller via `extra=` and is treated as context.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
    # OTel LoggingInstrumentor injects these — surfaced explicitly below.
    "otelTraceID", "otelSpanID", "otelTraceSampled", "otelServiceName",
}


class _JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service = service_name

    def format(self, record: logging.LogRecord) -> str:
        trace_id, span_id = _trace_ids(record)
        payload = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self._service,
            "message": record.getMessage(),
        }
        if trace_id:
            payload["trace_id"] = trace_id
        if span_id:
            payload["span_id"] = span_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Any non-reserved record attributes were passed via `extra=` — emit
        # them as contextual key/values. JSON-unsafe values are stringified.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        return json.dumps(payload, default=str)


def _trace_ids(record: logging.LogRecord) -> tuple[str | None, str | None]:
    """Resolve (trace_id, span_id) hex strings for a record.

    Prefers the ids `LoggingInstrumentor` injected onto the record; falls back
    to the currently-active span context (covers records emitted before the
    instrumentor processes them, or when it is unavailable).
    """
    trace_id = getattr(record, "otelTraceID", None)
    span_id = getattr(record, "otelSpanID", None)
    if trace_id and trace_id != "0" and set(trace_id) != {"0"}:
        return trace_id, span_id

    ctx = _trace_api.get_current_span().get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    return None, None


def configure_json_logging(service_name: str) -> None:
    """Install the JSON formatter on the root logger's stdout handler.

    Idempotent: re-running swaps the formatter rather than stacking handlers.
    """
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    formatter = _JsonFormatter(service_name)

    # Reuse an existing stdout StreamHandler if one is present (avoid stacking
    # duplicate handlers when several service modules each configure logging).
    stream_handler = None
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(
            handler, "stream", None
        ) in (sys.stdout, sys.stderr):
            stream_handler = handler
            break
    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
        root.addHandler(stream_handler)

    stream_handler.setFormatter(formatter)
