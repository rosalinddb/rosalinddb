# RosalindDB backend image.
#
# One image, five process roles (cp, query_dp, validator, builder, ephemeral)
# — the public Control Plane, the private Query Data Plane, and the three
# async workers. The run command per role is set in `docker-compose.yml`;
# this image only needs to contain the code and its dependencies.
# Python 3.11 is required: several pins in requirements.txt (faiss-cpu,
# pyarrow, psycopg2-binary) are gated on `python_version < '3.13'`.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Build toolchain is needed to compile psycopg2-binary / numpy wheels on slim;
# faiss-cpu ships a manylinux wheel so it installs without extra system libs.
# curl is kept for container-level debugging / health probing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer — cached unless requirements.txt changes.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Only the directories the process groups import are
# copied (see .dockerignore for what is excluded from the build context).
COPY adapters/ /app/adapters/
COPY services/ /app/services/
COPY scripts/ /app/scripts/

# Run as a non-root user. CACHE_DIR (the on-disk FAISS shard cache) lives under
# the writable home directory; set CACHE_DIR to a writable path in your
# deployment config (default below: /home/rosalind/cache).
RUN useradd --create-home --uid 10001 rosalind \
    && mkdir -p /home/rosalind/cache \
    && chown -R rosalind:rosalind /home/rosalind /app
USER rosalind

# The web process group listens here. The worker process groups ignore the
# exposed port.
EXPOSE 8080

# Default command is the Control Plane (the single public origin).
# `docker-compose.yml` overrides this per service for the Query-DP and the
# async workers, so this CMD only matters for a bare `docker run`.
CMD ["uvicorn", "services.control_plane.cp_app:app", "--host", "0.0.0.0", "--port", "8080"]
