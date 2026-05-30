"""Uvicorn import hook for ASGI servers.

Allows running this service via `uvicorn services.source_registry.uvicorn_entry:app`.
"""

from .main import app

