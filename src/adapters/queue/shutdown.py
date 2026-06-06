from __future__ import annotations

"""Graceful-shutdown plumbing for the pipeline workers.

A deploy/restart sends `SIGTERM`. The reliable-queue processing-list pattern
makes a worker death *recoverable* — but recovery costs a reclaim-timeout of
latency. Graceful shutdown avoids that cost in the common case: on `SIGTERM` a
worker stops pulling *new* messages and finishes (or cleanly `nack`s) the one
it is currently holding before the process exits, so the message is acked or
re-queued immediately rather than waiting for the reaper.

`install_signal_handlers()` wires `SIGTERM`/`SIGINT` to set a process-wide
`threading.Event`. A worker's consume loop checks `should_stop()` at the top of
every iteration and breaks out cleanly. HTTP services (uvicorn) already drain
in-flight requests on `SIGTERM`; this module is for the queue workers, whose
loops are hand-rolled.
"""

import signal
import threading

# Process-wide shutdown signal. Set by the SIGTERM/SIGINT handler; polled by
# every worker's consume loop via `should_stop()`.
_STOP_EVENT = threading.Event()


def should_stop() -> bool:
    """True once a shutdown signal (SIGTERM/SIGINT) has been received."""
    return _STOP_EVENT.is_set()


def stop_event() -> threading.Event:
    """The shared shutdown `Event` — pass it to background loops (the reaper)."""
    return _STOP_EVENT


def request_stop() -> None:
    """Trip the shutdown signal manually (used by tests and explicit shutdown)."""
    _STOP_EVENT.set()


def reset() -> None:
    """Clear the shutdown signal — test hook so one test's stop does not leak."""
    _STOP_EVENT.clear()


def install_signal_handlers() -> None:
    """Wire SIGTERM and SIGINT to trip the shutdown event.

    Idempotent and safe to call once at worker startup. Registering a handler
    only works on the main thread; if a worker is hosted as a non-main thread
    (a single-process dev/test harness) the `ValueError` is swallowed — such a
    harness manages its own lifecycle and the workers there are daemon threads.
    """
    def _handler(signum, _frame):  # noqa: ANN001
        print(f"shutdown: signal {signum} received — draining and exiting")
        _STOP_EVENT.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not on the main thread (dev harness) — nothing to wire.
            pass
