#!/usr/bin/env bash
# Multi-agent memory load benchmark — concurrent agents bombarding the recall
# (read-your-writes) path.
#
# Sweeps MODE in {per-agent, shared} x AGENTS in {10, 50, 100} (defaults;
# env-overridable). Each (mode x agents) cell runs load_test_agents.js against
# a RECALL-ENABLED stack and captures:
#   - k6 stdout log
#   - k6 JSON summary (write throughput, write/search latency, read-your-writes
#     hit-rate + lag, error rates)
#   - docker stats sampled at 1 Hz (one container snapshot per line)
#   - cell window start/end timestamps
#
# The stack is the self-contained recall-bench compose (separate project +
# ports, NOT :8080):
#   docker compose -p rosalinddb-recall-bench \
#     -f bench/docker-compose.recall-bench.yml up -d --build
#
# Usage:
#   bash bench/run_agents_bench.sh                 # full sweep (2m cells)
#   bash bench/run_agents_bench.sh --smoke         # 3 agents, 30s, both modes
#   bash bench/run_agents_bench.sh --keep-up       # don't tear the stack down
#   bash bench/run_agents_bench.sh --no-build      # reuse the existing image
#
# Knobs (env): AGENTS_LIST, MODES, DURATION, DIM, MEMORIES_PER, TOP_K,
#   SETTLE_S, BENCH_PROJECT, and the RB_BENCH_*_PORT host-port overrides
#   consumed by the recall-bench compose file.

set -euo pipefail

# --- args ------------------------------------------------------------------

SMOKE=0
KEEP_UP=0
DO_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --keep-up) KEEP_UP=1 ;;
    --no-build) DO_BUILD=0 ;;
    -h|--help)
      sed -n '2,33p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# Run from the backend root (parent of bench/).
cd "$(dirname "$0")/.."

PROJECT="${BENCH_PROJECT:-rosalinddb-recall-bench}"
COMPOSE=(docker compose -p "$PROJECT" -f bench/docker-compose.recall-bench.yml)

# Host port the CP is published on by the recall-bench compose — must match
# RB_BENCH_CP_PORT in docker-compose.recall-bench.yml. We probe health here.
CP_PORT="${RB_BENCH_CP_PORT:-18080}"

# --- cell parameters -------------------------------------------------------

DIM="${DIM:-768}"
MEMORIES_PER="${MEMORIES_PER:-5}"
TOP_K="${TOP_K:-10}"
SETTLE_S="${SETTLE_S:-5}"
MODES=(${MODES:-per-agent shared})

if [[ "$SMOKE" -eq 1 ]]; then
  AGENTS_LIST=(${AGENTS_LIST:-3})
  DURATION="${DURATION:-30s}"
else
  AGENTS_LIST=(${AGENTS_LIST:-10 50 100})
  DURATION="${DURATION:-2m}"
fi

# PID suffix avoids same-second collisions between retried runs.
TS="$(date -u +"%Y%m%dT%H%M%SZ")-$$"
RESULTS_ROOT="bench/results/agents-${TS}"
mkdir -p "$RESULTS_ROOT"

echo "=== agent-memory bench ${TS} ==="
echo "    project=$PROJECT  cp_port=$CP_PORT  smoke=$SMOKE"
echo "    modes=${MODES[*]}  agents=${AGENTS_LIST[*]}"
echo "    duration=$DURATION  dim=$DIM  memories/batch=$MEMORIES_PER  top_k=$TOP_K"
echo "    results -> $RESULTS_ROOT"
echo

# --- bring the recall-bench stack up --------------------------------------

UP_ARGS=(up -d)
[[ "$DO_BUILD" -eq 1 ]] && UP_ARGS+=(--build)

echo "[up] bringing the recall-bench stack up (separate project + ports)"
"${COMPOSE[@]}" "${UP_ARGS[@]}"

# tear down on exit unless --keep-up.
cleanup() {
  if [[ "$KEEP_UP" -eq 1 ]]; then
    echo "[keep-up] leaving the stack running ($PROJECT). Tear down with:"
    echo "  ${COMPOSE[*]} down -v"
    return
  fi
  echo "[down] tearing the recall-bench stack down"
  "${COMPOSE[@]}" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- wait for the CP + recall to be ready ---------------------------------

echo "[wait] CP health at localhost:${CP_PORT}"
ok=0
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:${CP_PORT}/healthz" >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 2
done
if [[ "$ok" != "1" ]]; then
  echo "  CP /healthz never came up on :${CP_PORT}" >&2
  "${COMPOSE[@]}" ps | tee "$RESULTS_ROOT/00_ps_unhealthy.txt"
  exit 2
fi
echo "  ok"

# Recall sanity: a synchronous write (200) followed by a read-your-writes probe
# that HITS. This proves the stack is recall-enabled before we burn cells on it.
echo "[check] recall sanity (sync 200 write + read-your-writes hit)"
SANITY_DS="recall_sanity_$$"
curl -fsS -X POST "http://localhost:${CP_PORT}/v1/datasets" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${SANITY_DS}\",\"dimension\":3}" >/dev/null 2>&1 || true
W_CODE=$(curl -s -o /dev/null -w '%{http_code}' \
  -X POST "http://localhost:${CP_PORT}/v1/datasets/${SANITY_DS}/vectors" \
  -H 'Content-Type: application/x-ndjson' \
  -d '{"id":"s1","values":[0.1,0.2,0.3],"metadata":{"k":"v"}}')
Q_BODY=$(curl -s -X POST "http://localhost:${CP_PORT}/v1/query" \
  -H 'Content-Type: application/json' \
  -d "{\"dataset\":\"${SANITY_DS}\",\"vector\":[0.1,0.2,0.3],\"top_k\":5}")
echo "  write http=$W_CODE"
echo "  query body=$(echo "$Q_BODY" | head -c 240)"
if [[ "$W_CODE" != "200" ]]; then
  echo "  EXPECTED 200 from the recall write — stack is NOT recall-enabled." >&2
  exit 3
fi
if ! echo "$Q_BODY" | grep -q '"s1"'; then
  echo "  read-your-writes sentinel 's1' NOT found in query — recall union broken." >&2
  exit 3
fi
echo "  ok — recall write returned 200 and the sentinel was read back"
echo "$W_CODE" > "$RESULTS_ROOT/recall_sanity_write_code.txt"
echo "$Q_BODY" > "$RESULTS_ROOT/recall_sanity_query.json"

"${COMPOSE[@]}" ps | tee "$RESULTS_ROOT/00_ps_before.txt" >/dev/null

# --- matrix loop -----------------------------------------------------------

for MODE in "${MODES[@]}"; do
  for AGENTS in "${AGENTS_LIST[@]}"; do
    CELL_DIR="$RESULTS_ROOT/${MODE}/agents-${AGENTS}"
    mkdir -p "$CELL_DIR"

    echo
    echo "=== cell MODE=$MODE AGENTS=$AGENTS  ($CELL_DIR) ==="
    CELL_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_START" > "$CELL_DIR/started_at.txt"

    # Container IDs to monitor (everything in the bench compose project).
    IDS=$("${COMPOSE[@]}" ps -q)

    # docker stats sampler in background; one JSON line per container per sample.
    (
      while true; do
        ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
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
      -e MODE="$MODE" \
      -e AGENTS="$AGENTS" \
      -e DIM="$DIM" \
      -e MEMORIES_PER="$MEMORIES_PER" \
      -e TOP_K="$TOP_K" \
      -e DURATION="$DURATION" \
      -e DATASET_PREFIX="am_${MODE//-/_}_${AGENTS}" \
      -e SUMMARY_PATH="/bench/results/agents-${TS}/${MODE}/agents-${AGENTS}/k6_summary.json" \
      k6 k6 run /bench/load_test_agents.js \
      > "$CELL_DIR/k6_stdout.log" 2>&1
    K6_RC=$?
    set -e

    kill "$STATS_PID" 2>/dev/null || true
    wait "$STATS_PID" 2>/dev/null || true

    CELL_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_END" > "$CELL_DIR/ended_at.txt"
    echo "$K6_RC"    > "$CELL_DIR/k6_exit.txt"

    # Compact one-liner so the run log is readable at a glance.
    grep -E 'throughput|hit-rate|p95|p\(95\)|writes|search|error|FAIL|PASS' \
      "$CELL_DIR/k6_stdout.log" | head -n 14 || true
    echo "  -> k6 exit=$K6_RC  window=$CELL_START..$CELL_END"

    sleep "$SETTLE_S"
  done
done

echo
echo "=== done. raw results at $RESULTS_ROOT ==="
"${COMPOSE[@]}" ps > "$RESULTS_ROOT/99_ps_after.txt"
ls -laR "$RESULTS_ROOT" | head -n 60

echo
echo "Aggregate with:"
echo "  python3 bench/analyze_agents.py $RESULTS_ROOT"
