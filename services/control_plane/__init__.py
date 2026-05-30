"""Control Plane (CP) service package.

The CP is the public surface that hosts auth, the `/v1/datasets*` catalog, the
ingest path, and the proxied `/v1/query` surface (the latter reverse-proxied to
a private Query Data Plane). See `services/control_plane/cp_app.py`.
"""
