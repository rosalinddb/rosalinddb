from __future__ import annotations

"""Lightweight in-process metrics adapter with AWS CloudWatch support.

Counters, gauges, and timers are aggregated under a thread lock and exposed via
`snapshot()` for surfacing in the Query API. In AWS mode, metrics are also sent
to CloudWatch.
"""

import threading
import time
from collections import defaultdict
from typing import Optional

from adapters import config

_counters = defaultdict(int)
_gauges = defaultdict(float)
_timers = defaultdict(list)
_lock = threading.Lock()

_CLOUD_PROVIDER = config.cloud_provider()
_cloudwatch = None
_namespace = config.cloudwatch_namespace()

if _CLOUD_PROVIDER == "aws":
    try:
        import boto3
        _cloudwatch = boto3.client('cloudwatch')
    except ImportError:
        _cloudwatch = None


def counter(name: str, inc: int = 1) -> None:
    """Increment a named counter by `inc` (default 1)."""
    with _lock:
        _counters[name] += inc
    
    # Send to CloudWatch if available
    _send_to_cloudwatch(name, inc, "Count")


def gauge(name: str, value: float) -> None:
    """Set a named gauge to a floating-point value."""
    with _lock:
        _gauges[name] = value
    
    # Send to CloudWatch if available
    _send_to_cloudwatch(name, value, "None")


def timer(name: str, duration_seconds: float) -> None:
    """Record a duration sample (in seconds) under the given timer name."""
    with _lock:
        _timers[name].append(duration_seconds)
    
    # Send to CloudWatch if available
    _send_to_cloudwatch(name, duration_seconds, "Seconds")


def _send_to_cloudwatch(metric_name: str, value: float, unit: str) -> None:
    """Send metric to CloudWatch if available."""
    if _cloudwatch is None:
        return
    
    try:
        service_name = config.service_role()
        _cloudwatch.put_metric_data(
            Namespace=_namespace,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Dimensions': [
                        {
                            'Name': 'Service',
                            'Value': service_name
                        }
                    ]
                }
            ]
        )
    except Exception:
        # Silently ignore CloudWatch failures to avoid breaking the application
        pass


def snapshot():
    """Return a threadsafe copy of current counters, gauges, and timers."""
    with _lock:
        return {
            "counters": dict(_counters),
            "gauges": dict(_gauges),
            "timers": {k: list(v) for k, v in _timers.items()},
        }

