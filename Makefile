VENV?=.venv
# Prefer Python 3.11 for binary wheels (faiss/pyarrow/psycopg2)
PYTHON_BIN?=$(shell (command -v python3.11 || command -v python3))
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
PYTEST=$(VENV)/bin/pytest

.PHONY: build test test-unit test-integration smoke run-local fmt lint

# Base URL the smoke check targets. Defaults to the local compose stack's
# Control Plane (port 8080 — the single public origin); override for a
# deployed instance:  make smoke BASE_URL=https://api.example.com
BASE_URL?=http://localhost:8080

# Build the one container image every service runs from. Per-service command
# overrides live in `docker-compose.yml`.
build:
	docker build -t rosalinddb-backend:latest .

venv:
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .

# test-unit:        fast, hermetic suite — memory:// storage, no Docker.
# test-integration: end-to-end suite — spins up an ephemeral MinIO container
#                   per session via testcontainers (Docker required).
# test:             both tiers.
test-unit: venv
	$(PYTEST) -m unit -q

test-integration: venv
	$(PYTEST) -m integration -q

test: test-unit test-integration

# smoke: lightweight API happy-path check against an ALREADY-RUNNING instance.
#        Exercises health -> signup -> dataset -> ingest -> index -> query and
#        exits non-zero on any failure. Does NOT manage a stack; bring one up
#        first (`docker compose up -d`) or point it at a deployed URL:
#            make smoke BASE_URL=https://api.example.com
#        Uses stdlib only, so no venv is required.
smoke:
	$(PYTHON_BIN) scripts/smoke.py --base-url $(BASE_URL)

# run-local: contributor compile-from-source flow. The default
# `docker compose up` PULLS the published image; this layers the build override
# so contributors build the image locally (tagged the same name the base file
# pulls) and run it. Behaviour is identical to the old compile-from-source path.
run-local:
	docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build

fmt:
	ruff format .

lint:
	ruff check .
