#!/usr/bin/env bash
# Drive the 6-cell bench matrix: 2 dims (128, 1536) x 3 VU levels (10, 20, 50).
#
# Each cell captures:
#   - k6 stdout log
#   - k6 JSON summary (latency histograms, error rates, per-tag)
#   - docker stats sampled at 1 Hz (one container snapshot per line)
#   - cell window start/end timestamps
#
# Corpora are seeded once per dimension; the three VU runs at that dim
# reuse the cache.
#
# Prerequisites — both stacks up first:
#   docker compose -p rosalinddb-observability \
#     -f observability/docker-compose.yml up -d
#   docker compose -p rosalinddb-bench \
#     -f docker-compose.yml -f bench/docker-compose.bench.yml up -d

set -euo pipefail

# --- config -----------------------------------------------------------------

# Where to run from: backend root (parent of bench/).
cd "$(dirname "$0")/.."

PROJECT="${BENCH_PROJECT:-rosalinddb-bench}"
COMPOSE=(docker compose -p "$PROJECT" -f docker-compose.yml -f bench/docker-compose.bench.yml)

DURATION="${DURATION:-5m}"
TENANTS="${TENANTS:-15}"
VECTORS_PER="${VECTORS_PER:-309}"
DIMS=(${DIMS:-128 1536})
VUS_LIST=(${VUS_LIST:-10 20 50})
SETTLE_S="${SETTLE_S:-5}"

TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RESULTS_ROOT="bench/results/$TS"
mkdir -p "$RESULTS_ROOT"

echo "=== bench run $TS ==="
echo "    project=$PROJECT  duration=$DURATION  tenants=$TENANTS  vectors/tenant=$VECTORS_PER"
echo "    dims=${DIMS[*]}  vus=${VUS_LIST[*]}"
echo "    results -> $RESULTS_ROOT"
echo

# --- prereq sanity check ---------------------------------------------------

echo "[check] backend stack health"
"${COMPOSE[@]}" ps  | tee "$RESULTS_ROOT/00_ps_before.txt"

echo
echo "[check] CP reachable at localhost:8080"
if ! curl -fsS http://localhost:8080/healthz > /dev/null; then
  echo "  CP /healthz failed — bring the stack up first." >&2
  exit 2
fi
echo "  ok"

# --- seed phase ------------------------------------------------------------

mkdir -p bench/cache
for DIM in "${DIMS[@]}"; do
  CACHE="bench/cache/dim-$DIM.json"
  if [[ -f "$CACHE" && "${REUSE_CORPUS:-0}" == "1" ]]; then
    n=$(python3 -c "import json,sys; print(len(json.load(open('$CACHE'))))")
    echo "[seed] reusing $CACHE ($n tenants)"
    continue
  fi
  rm -f "$CACHE"
  echo "[seed] dim=$DIM tenants=$TENANTS vectors/tenant=$VECTORS_PER"
  python3 bench/seed_corpus.py \
    --base-url http://localhost:8080 \
    --dim "$DIM" \
    --tenants "$TENANTS" \
    --vectors-per "$VECTORS_PER" \
    --out "$CACHE" \
    2>&1 | tee "$RESULTS_ROOT/seed_dim-$DIM.log"
done

# --- matrix loop -----------------------------------------------------------

for DIM in "${DIMS[@]}"; do
  for VUS in "${VUS_LIST[@]}"; do
    CELL_DIR="$RESULTS_ROOT/dim-$DIM/vus-$VUS"
    mkdir -p "$CELL_DIR"

    echo
    echo "=== cell DIM=$DIM VUS=$VUS  ($CELL_DIR) ==="
    CELL_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_START" > "$CELL_DIR/started_at.txt"

    # Container IDs to monitor (everything in the bench compose project).
    IDS=$("${COMPOSE[@]}" ps -q)

    # docker stats sampler in background; one JSON line per container per sample.
    (
      while true; do
        ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        # --no-stream returns one snapshot per container, then exits.
        docker stats --no-stream --format '{{json .}}' $IDS 2>/dev/null \
          | sed "s/^/{\"ts\":\"$ts\",/; s/^{\"ts\":\"$ts\",{/{\"ts\":\"$ts\",/"
        sleep 1
      done
    ) > "$CELL_DIR/docker_stats.jsonl" 2>&1 &
    STATS_PID=$!

    # Run k6 inside the bench's k6 container so RTT is loopback-only.
    set +e
    "${COMPOSE[@]}" exec -T \
      -e BASE_URL=http://cp:8080 \
      -e DIM="$DIM" \
      -e VUS="$VUS" \
      -e DURATION="$DURATION" \
      -e CORPUS="/bench/cache/dim-$DIM.json" \
      -e SUMMARY_PATH="/bench/results/$TS/dim-$DIM/vus-$VUS/k6_summary.json" \
      k6 k6 run /bench/load_test_queries.js \
      > "$CELL_DIR/k6_stdout.log" 2>&1
    K6_RC=$?
    set -e

    kill "$STATS_PID" 2>/dev/null || true
    wait "$STATS_PID" 2>/dev/null || true

    CELL_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_END" > "$CELL_DIR/ended_at.txt"
    echo "$K6_RC"    > "$CELL_DIR/k6_exit.txt"

    # Compact one-liner so the run log is readable at a glance.
    tail -n 12 "$CELL_DIR/k6_stdout.log" | grep -E 'rate|p\(95\)|p\(99\)|queries|error|FAIL|PASS' \
      || true
    echo "  -> k6 exit=$K6_RC  window=$CELL_START..$CELL_END"

    # Settle so the next cell starts from a quiet stat baseline.
    sleep "$SETTLE_S"
  done
done

echo
echo "=== done. raw results at $RESULTS_ROOT ==="
"${COMPOSE[@]}" ps  > "$RESULTS_ROOT/99_ps_after.txt"
ls -la "$RESULTS_ROOT"
