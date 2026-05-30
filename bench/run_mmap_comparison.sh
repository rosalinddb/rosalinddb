#!/usr/bin/env bash
# Runs the bench matrix TWICE — once with RB_FAISS_MMAP=false, once =true —
# and writes both result trees side by side under bench/results/<ts>-mmap/.
#
# The corpus is seeded ONCE (with mmap off) and reused for the on run, so
# the comparison isolates the mmap toggle: same data, same load, two DP
# binaries pointed at the same shards.
#
# Usage:
#   bash bench/run_mmap_comparison.sh            # full run (~45 min)
#   bash bench/run_mmap_comparison.sh --smoke    # 30s cells, 10-vector seed
#                                                  for harness validation only
#
# Prerequisites — both stacks up first:
#   docker compose -p rosalinddb-observability \
#     -f observability/docker-compose.yml up -d
#   docker compose -p rosalinddb-bench \
#     -f docker-compose.yml -f bench/docker-compose.bench.yml up -d

set -euo pipefail

# --- args ------------------------------------------------------------------

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --skip-seed) SKIP_SEED=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# --- config ----------------------------------------------------------------

# Run from the backend root (parent of bench/).
cd "$(dirname "$0")/.."

PROJECT="${BENCH_PROJECT:-rosalinddb-bench}"
COMPOSE=(docker compose -p "$PROJECT" -f docker-compose.yml -f bench/docker-compose.bench.yml)

DIM="${DIM:-1536}"
VUS_LIST=(${VUS_LIST:-10 20 50})
SETTLE_S="${SETTLE_S:-5}"

if [[ "$SMOKE" -eq 1 ]]; then
  VECTORS_PER="${VECTORS_PER:-10}"
  DURATION="${DURATION:-30s}"
  INDEX_TIMEOUT="${INDEX_TIMEOUT:-180}"
else
  VECTORS_PER="${VECTORS_PER:-1000000}"
  DURATION="${DURATION:-1m}"
  INDEX_TIMEOUT="${INDEX_TIMEOUT:-1800}"
fi

# PID suffix avoids same-second collisions between retried/parallel runs.
TS="$(date -u +"%Y%m%dT%H%M%SZ")-$$"
RESULTS_ROOT="bench/results/${TS}-mmap"
CACHE_DIR="bench/cache"
CACHE="${CACHE_DIR}/dim-${DIM}-mmap.json"
mkdir -p "$RESULTS_ROOT" "$CACHE_DIR"

# Restore RB_FAISS_MMAP=false on every exit path (success, failure, Ctrl-C).
# Without this, a crash during phase 2 leaves the stack in mmap=true for the
# next ad-hoc user. The restart is idempotent — running it when DP is already
# off is a fast no-op.
_restore_mmap_off() {
  local rc=$?
  echo "[cleanup] restoring RB_FAISS_MMAP=false"
  RB_FAISS_MMAP=false "${COMPOSE[@]}" up -d --force-recreate query_dp ephemeral_runner \
    > /dev/null 2>&1 || true
  exit "$rc"
}
trap _restore_mmap_off EXIT INT TERM

echo "=== mmap-comparison bench $TS ==="
echo "    project=$PROJECT  dim=$DIM  vectors=$VECTORS_PER  duration=$DURATION"
echo "    vus=${VUS_LIST[*]}  smoke=$SMOKE"
echo "    results -> $RESULTS_ROOT"
echo

# --- prereq sanity check ---------------------------------------------------

echo "[check] backend stack reachable"
if ! "${COMPOSE[@]}" ps > /dev/null 2>&1; then
  echo "  docker compose project '$PROJECT' not found." >&2
  echo "  Stack not up; see bench/README.md for the bring-up command." >&2
  exit 2
fi
if ! curl -fsS http://localhost:8080/healthz > /dev/null; then
  echo "  CP /healthz failed — stack not up; see bench/README.md." >&2
  exit 2
fi
echo "  ok"

# --- helper: bring DP down + back up with a given RB_FAISS_MMAP value -----

restart_dp_with_mmap() {
  local flag="$1"
  echo "[dp] restarting query_dp and ephemeral_runner with RB_FAISS_MMAP=$flag"
  # `up -d` (without --no-deps) is intentional: the FAISS-loading services
  # carry a depends_on graph that references already-completed one-shot
  # services (migrator, createbuckets), and `--no-deps` errors when those
  # are in `exited(0)` state. Letting compose evaluate the graph is a no-op
  # for already-healthy services and is the only path that recreates only
  # the two we asked for.
  RB_FAISS_MMAP="$flag" "${COMPOSE[@]}" up -d --force-recreate \
    query_dp ephemeral_runner
  # Give DP a few seconds to load the FAISS catalog before driving load.
  sleep 5
}

# --- helper: run the N-cell matrix into a given subdir --------------------

run_cells() {
  local subdir="$1"
  mkdir -p "$subdir"
  for VUS in "${VUS_LIST[@]}"; do
    local CELL_DIR="$subdir/dim-$DIM/vus-$VUS"
    mkdir -p "$CELL_DIR"
    echo
    echo "=== cell DIM=$DIM VUS=$VUS  ($CELL_DIR) ==="
    local CELL_START
    CELL_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_START" > "$CELL_DIR/started_at.txt"

    local IDS
    IDS=$("${COMPOSE[@]}" ps -q)

    (
      while true; do
        local ts
        ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        docker stats --no-stream --format '{{json .}}' $IDS 2>/dev/null \
          | sed "s/^/{\"ts\":\"$ts\",/; s/^{\"ts\":\"$ts\",{/{\"ts\":\"$ts\",/"
        sleep 1
      done
    ) > "$CELL_DIR/docker_stats.jsonl" 2>&1 &
    local STATS_PID=$!

    # k6 result paths are container-relative to /bench (the volume mount).
    local SUMMARY_REL="results/${TS}-mmap/$(basename "$subdir")/dim-$DIM/vus-$VUS/k6_summary.json"

    set +e
    "${COMPOSE[@]}" exec -T \
      -e BASE_URL=http://cp:8080 \
      -e DIM="$DIM" \
      -e VUS="$VUS" \
      -e DURATION="$DURATION" \
      -e CORPUS="/bench/cache/dim-${DIM}-mmap.json" \
      -e SUMMARY_PATH="/bench/${SUMMARY_REL}" \
      k6 k6 run /bench/load_test_queries.js \
      > "$CELL_DIR/k6_stdout.log" 2>&1
    local K6_RC=$?
    set -e

    kill "$STATS_PID" 2>/dev/null || true
    wait "$STATS_PID" 2>/dev/null || true

    local CELL_END
    CELL_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "$CELL_END" > "$CELL_DIR/ended_at.txt"
    echo "$K6_RC"    > "$CELL_DIR/k6_exit.txt"

    tail -n 12 "$CELL_DIR/k6_stdout.log" | grep -E 'rate|p\(95\)|p\(99\)|queries|error|FAIL|PASS' \
      || true
    echo "  -> k6 exit=$K6_RC  window=$CELL_START..$CELL_END"
    sleep "$SETTLE_S"
  done
}

# --- phase 1: mmap OFF — seed + run ---------------------------------------

restart_dp_with_mmap "false"

echo
if [[ "${SKIP_SEED:-0}" == "1" ]]; then
  echo "[seed] --skip-seed: assuming \$CACHE already populated by an external"
  echo "       producer (e.g. bench/build_shard_directly.py for a multi-GB shard"
  echo "       that the CP ingest path can't deliver in tractable time)."
  SEED_RC=0
else
  echo "[seed] dim=$DIM vectors=$VECTORS_PER (single-tenant)"
  # REUSE_CORPUS=1 keeps the existing cache (and skips the ~minutes-long
  # 1M-vector seed) — useful for iterating on the cell-driving logic.
  if [[ "${REUSE_CORPUS:-0}" != "1" ]]; then
    rm -f "$CACHE"
  fi
  # `|| true` so a partial seed (the seeder returns 1 on zero successes) does
  # not trip `set -e` before the count check below can produce a clear error.
  set +e
  python3 bench/seed_corpus.py \
    --base-url http://localhost:8080 \
    --dim "$DIM" \
    --vectors-per "$VECTORS_PER" \
    --single-tenant \
    --index-timeout "$INDEX_TIMEOUT" \
    --out "$CACHE" \
    2>&1 | tee "$RESULTS_ROOT/seed.log"
  SEED_RC=${PIPESTATUS[0]}
  set -e
fi

# Refuse to drive k6 against an empty corpus: load_test_queries.js cannot
# pick a tenant and every request would 4xx, masking the real comparison.
if [[ ! -f "$CACHE" ]]; then
  echo "  seed wrote no cache file (rc=$SEED_RC) — aborting before cell runs." >&2
  echo "  inspect $RESULTS_ROOT/seed.log and the stack logs (docker compose -p $PROJECT logs cp)." >&2
  exit 3
fi
seed_count=$(python3 -c "import json; print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo 0)
if [[ "$seed_count" -eq 0 ]]; then
  echo "  seed produced 0 corpus entries (rc=$SEED_RC) — aborting before cell runs." >&2
  echo "  inspect $RESULTS_ROOT/seed.log and the stack logs (docker compose -p $PROJECT logs cp)." >&2
  exit 3
fi
echo "  seeded $seed_count corpus entr$([[ $seed_count -eq 1 ]] && echo y || echo ies)"

echo
echo "[run] mmap=off cells"
run_cells "$RESULTS_ROOT/off"

# --- phase 2: mmap ON — restart DP, reuse corpus --------------------------

restart_dp_with_mmap "true"

echo
echo "[run] mmap=on cells (reusing seeded corpus)"
run_cells "$RESULTS_ROOT/on"

# --- summary --------------------------------------------------------------

echo
echo "=== analyzing results ==="
python3 bench/analyze.py "$RESULTS_ROOT/off" || echo "  (analyze off subdir failed)"
echo
python3 bench/analyze.py "$RESULTS_ROOT/on"  || echo "  (analyze on subdir failed)"

echo
echo "=== done. raw results at $RESULTS_ROOT ==="
echo "    off: $RESULTS_ROOT/off/RESULTS.md"
echo "    on:  $RESULTS_ROOT/on/RESULTS.md"
