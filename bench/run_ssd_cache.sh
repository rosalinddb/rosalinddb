#!/usr/bin/env bash
# Drive the 3-cell SSD-cache bench: tier-off, tier-on-cold, tier-on-warm.
#
# The cells:
#
#   1. tier-off (baseline)
#      RB_SHARD_TIER_BYTES unset on query_dp + ephemeral_runner.
#      Cold reset. 1 min at 20 VUs.
#      This is "today's behaviour" — the architecture preserves it.
#
#   2. tier-on-cold
#      RB_SHARD_TIER_BYTES=2147483648 (2 GB SSD budget).
#      Cold reset between cells 1 and 2.
#      1 min at 20 VUs. The first queries pay the GET from object storage;
#      later queries within the cell hit the SSD tier.
#
#   3. tier-on-warm
#      Same env as cell 2. NO reset between cells 2 and 3 — the residency
#      table and the local SSD files survive. 1 min at 20 VUs.
#      Every query should be a warm-tier hit; the GET is skipped.
#
# What the deltas are supposed to show:
#   (a) cell 1 vs cell 2 within margin of error (tier-off and tier-on-cold
#       both pay the cold GET — the tier helps subsequent queries, not the
#       first).
#   (b) cell 3 measurably faster than cell 2 (warm-tier skips the GET).
#   (c) recall@10 >= 0.95 across all cells (anything below is a wiring
#       bug — see analyze_ssd_cache.py for the rationale).
#
# Usage:
#   bash bench/run_ssd_cache.sh                 # full run (~5 min total)
#   bash bench/run_ssd_cache.sh --smoke         # 10s cells, 10-vector seed
#   bash bench/run_ssd_cache.sh --dry-run       # print what would run, no docker/k6
#   bash bench/run_ssd_cache.sh --smoke --dry-run    # both
#
# --dry-run is the contract the unit test in tests/unit/test_bench_ssd_cache_smoke.py
# pins: argument parsing + cell sequencing without a live stack.
#
# Prereqs (for a real run, not --dry-run):
#   docker compose -p rosalinddb-bench \
#     -f docker-compose.yml -f bench/docker-compose.bench.yml up -d

set -euo pipefail

# --- args ------------------------------------------------------------------

SMOKE=0
DRY_RUN=0
DROP_CACHES=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --drop-caches) DROP_CACHES=1 ;;
    -h|--help)
      sed -n '2,45p' "$0"
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

PROJECT="${BENCH_PROJECT:-rosalinddb-bench}"
COMPOSE=(docker compose -p "$PROJECT" -f docker-compose.yml -f bench/docker-compose.bench.yml)

# --- cell parameters ------------------------------------------------------

DIM="${DIM:-128}"
VUS="${VUS:-20}"
SETTLE_S="${SETTLE_S:-3}"
NUM_QUERIES="${NUM_QUERIES:-200}"
# 2 GB SSD-tier budget — large enough to fit the seeded shard in the smoke
# and full runs (both well under a GB at the configured dims/sizes), small
# enough that an operator can see the cap-vs-budget shape in docker stats.
TIER_BYTES="${RB_SHARD_TIER_BYTES:-2147483648}"

if [[ "$SMOKE" -eq 1 ]]; then
  VECTORS_PER="${VECTORS_PER:-10}"
  DURATION="${DURATION:-10s}"
  INDEX_TIMEOUT="${INDEX_TIMEOUT:-180}"
  NUM_QUERIES="${NUM_QUERIES:-20}"
else
  VECTORS_PER="${VECTORS_PER:-50000}"
  DURATION="${DURATION:-1m}"
  INDEX_TIMEOUT="${INDEX_TIMEOUT:-600}"
fi

# PID suffix avoids same-second collisions between retried runs.
TS="$(date -u +"%Y%m%dT%H%M%SZ")-$$"
RESULTS_ROOT="bench/results/${TS}-ssd-cache"
CACHE_DIR="bench/cache"
CORPUS_CACHE="${CACHE_DIR}/dim-${DIM}-ssd-cache.json"
GT_CACHE="${CACHE_DIR}/ground-truth-dim-${DIM}-n${VECTORS_PER}.json"

# --- dry-run printer ------------------------------------------------------

# `say` is the unified dispatch point: in dry-run it just prints what would
# happen (with a `[dry-run]` prefix); in live mode it prints the line then
# runs it. This is what the smoke test inspects — every state-changing call
# the script makes is funnelled through here so the test can assert the
# sequence without a live docker daemon.
say() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    eval "$@"
  fi
}

# `say_exec_capture` is the same idea for `docker compose exec k6 ...` calls,
# which need redirection of stdout to a file. Kept separate so the dry-run
# log doesn't include a confusing redirect.
say_exec_capture() {
  local out_path="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $* > $out_path"
  else
    echo "[run] $* > $out_path"
    eval "$@" > "$out_path" 2>&1
  fi
}

echo "=== ssd-cache bench $TS ==="
echo "    project=$PROJECT  dim=$DIM  vectors=$VECTORS_PER  duration=$DURATION  vus=$VUS"
echo "    tier_bytes=$TIER_BYTES  num_queries=$NUM_QUERIES"
echo "    results -> $RESULTS_ROOT"
echo "    flags: smoke=$SMOKE  dry_run=$DRY_RUN  drop_caches=$DROP_CACHES"
echo

# Pick a Python that has faiss / numpy / requests / boto3 available.
# Order: `$PYTHON_BIN` if exported, then the project venv, then a bare
# `python3`. The bare fallback usually fails the ground-truth step
# because system Python rarely has faiss; the script then exits with a
# clear error so an operator can set PYTHON_BIN explicitly.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
echo "    python=$PYTHON_BIN"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  # --- prereq sanity check (live runs only) ---
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
  mkdir -p "$RESULTS_ROOT" "$CACHE_DIR"
fi

# --- helper: restart query_dp + ephemeral_runner with a given tier setting -

# When `flag` is "off" we explicitly unset the env on the two containers
# (compose's `RB_SHARD_TIER_BYTES: "${RB_SHARD_TIER_BYTES:-}"` would
# otherwise inherit whatever's in the shell). When "on" we pass the byte
# budget.
restart_dp_with_tier() {
  local flag="$1"  # "off" or "on"
  echo
  echo "[dp] restarting query_dp + ephemeral_runner with tier=$flag"
  if [[ "$flag" == "off" ]]; then
    # Empty string -> the boolean check in v1_query.py treats it as "tier off".
    say "RB_SHARD_TIER_BYTES='' ${COMPOSE[*]} up -d --force-recreate query_dp ephemeral_runner"
  else
    say "RB_SHARD_TIER_BYTES='$TIER_BYTES' ${COMPOSE[*]} up -d --force-recreate query_dp ephemeral_runner"
  fi
  # Sleep so the DP can finish FAISS catalog load before the cell starts.
  if [[ "$DRY_RUN" -eq 0 ]]; then
    sleep "$SETTLE_S"
  fi
}

# --- helper: run one cell -------------------------------------------------

# A cell:
#   - creates its results subdir
#   - kicks off a 1 Hz docker stats sampler in the background
#   - drives k6 against load_test_queries_ssd_cache.js with the named CELL
#   - shuts the sampler down and stamps start/end timestamps + k6 rc
#
# `cell_name` is one of {tier-off, tier-on-cold, tier-on-warm}; matches
# CELL_ORDER in analyze_ssd_cache.py.
run_cell() {
  local cell_name="$1"
  local cell_dir="$RESULTS_ROOT/$cell_name"
  echo
  echo "=== cell $cell_name  ($cell_dir) ==="

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] mkdir -p $cell_dir"
    echo "[dry-run] start docker stats sampler -> $cell_dir/docker_stats.jsonl"
    echo "[dry-run] k6 run /bench/load_test_queries_ssd_cache.js (cell=$cell_name)"
    echo "[dry-run] stop docker stats sampler"
    echo "[dry-run] stamp started_at.txt / ended_at.txt / k6_exit.txt"
    return 0
  fi

  mkdir -p "$cell_dir"
  local cell_start
  cell_start="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "$cell_start" > "$cell_dir/started_at.txt"

  local ids
  ids=$("${COMPOSE[@]}" ps -q)

  (
    while true; do
      local ts
      ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      docker stats --no-stream --format '{{json .}}' $ids 2>/dev/null \
        | sed "s/^/{\"ts\":\"$ts\",/; s/^{\"ts\":\"$ts\",{/{\"ts\":\"$ts\",/"
      sleep 1
    done
  ) > "$cell_dir/docker_stats.jsonl" 2>&1 &
  local stats_pid=$!

  local summary_rel="results/${TS}-ssd-cache/${cell_name}/k6_summary.json"
  set +e
  "${COMPOSE[@]}" exec -T \
    -e BASE_URL=http://cp:8080 \
    -e VUS="$VUS" \
    -e DURATION="$DURATION" \
    -e CORPUS="/bench/cache/dim-${DIM}-ssd-cache.json" \
    -e GROUND_TRUTH="/bench/cache/ground-truth-dim-${DIM}-n${VECTORS_PER}.json" \
    -e SUMMARY_PATH="/bench/${summary_rel}" \
    -e RB_CELL_NAME="$cell_name" \
    k6 k6 run /bench/load_test_queries_ssd_cache.js \
    > "$cell_dir/k6_stdout.log" 2>&1
  local k6_rc=$?
  set -e

  kill "$stats_pid" 2>/dev/null || true
  wait "$stats_pid" 2>/dev/null || true

  local cell_end
  cell_end="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "$cell_end" > "$cell_dir/ended_at.txt"
  echo "$k6_rc"    > "$cell_dir/k6_exit.txt"

  tail -n 20 "$cell_dir/k6_stdout.log" \
    | grep -E 'rate|p\(95\)|p\(99\)|queries|error|FAIL|PASS' \
    || true
  echo "  -> k6 exit=$k6_rc  window=$cell_start..$cell_end"

  # Capture per-query response IDs against a QUIESCENT backend (the k6
  # load has stopped). analyze_ssd_cache.py uses this for the cell-
  # agreement check — every cell must return the same IDs as the
  # baseline (tier-off) cell for the same queries.
  echo
  echo "[capture] per-query results -> $cell_dir/query_results.json"
  set +e
  "$PYTHON_BIN" bench/capture_query_results.py \
    --base-url http://localhost:8080 \
    --corpus "$CORPUS_CACHE" \
    --query-set "$GT_CACHE" \
    --num-queries "$NUM_QUERIES" \
    --out "$cell_dir/query_results.json" \
    2> "$cell_dir/capture_query_results.log"
  local capture_rc=$?
  set -e
  if [[ "$capture_rc" -ne 0 ]]; then
    echo "  [warn] capture_query_results.py rc=$capture_rc — see $cell_dir/capture_query_results.log" >&2
  fi
}

# --- helper: invoke cold_reset.sh between cells ---------------------------

cold_reset() {
  local extra=""
  [[ "$DROP_CACHES" -eq 1 ]] && extra="--drop-caches"
  echo
  say "BENCH_PROJECT='$PROJECT' bash bench/cold_reset.sh $extra"
}

# --- phase 0: seed corpus (only when not --dry-run) ----------------------

seed_corpus() {
  echo
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] seed corpus dim=$DIM vectors=$VECTORS_PER -> $CORPUS_CACHE"
    echo "[dry-run] build ground truth dim=$DIM vectors=$VECTORS_PER num_queries=$NUM_QUERIES -> $GT_CACHE"
    return 0
  fi

  echo "[seed] corpus dim=$DIM vectors=$VECTORS_PER (single tenant)"
  rm -f "$CORPUS_CACHE"
  set +e
  "$PYTHON_BIN" bench/seed_corpus.py \
    --base-url http://localhost:8080 \
    --dim "$DIM" \
    --vectors-per "$VECTORS_PER" \
    --single-tenant \
    --index-timeout "$INDEX_TIMEOUT" \
    --out "$CORPUS_CACHE" \
    2>&1 | tee "$RESULTS_ROOT/seed.log"
  local seed_rc=${PIPESTATUS[0]}
  set -e
  if [[ ! -f "$CORPUS_CACHE" ]]; then
    echo "  seed wrote no cache file (rc=$seed_rc) — aborting before cell runs." >&2
    exit 3
  fi
  local seed_count
  seed_count=$("$PYTHON_BIN" -c "import json; print(len(json.load(open('$CORPUS_CACHE'))))" 2>/dev/null || echo 0)
  if [[ "$seed_count" -eq 0 ]]; then
    echo "  seed produced 0 corpus entries (rc=$seed_rc) — aborting." >&2
    exit 3
  fi
  echo "  seeded $seed_count corpus entr$([[ $seed_count -eq 1 ]] && echo y || echo ies)"

  # The seed cache lists the dataset name; pull it for the ground-truth builder.
  local dataset
  dataset=$("$PYTHON_BIN" -c "import json; print(json.load(open('$CORPUS_CACHE'))[0]['dataset'])")
  echo "[ground-truth] dataset=$dataset dim=$DIM vectors=$VECTORS_PER num_queries=$NUM_QUERIES"
  "$PYTHON_BIN" bench/build_ground_truth.py \
    --dataset "$dataset" \
    --dim "$DIM" \
    --vectors-per "$VECTORS_PER" \
    --num-queries "$NUM_QUERIES" \
    --out "$GT_CACHE" \
    --reuse-if-exists \
    2>&1 | tee "$RESULTS_ROOT/ground_truth.log"
}

seed_corpus

# --- phase 1: cell 1 — tier off (baseline) -------------------------------

restart_dp_with_tier "off"
cold_reset
run_cell "tier-off"

# --- phase 2: cell 2 — tier on, cold ------------------------------------

restart_dp_with_tier "on"
cold_reset
run_cell "tier-on-cold"

# --- phase 3: cell 3 — tier on, warm ------------------------------------

# Critical: NO cold reset here. Cell 2 left the residency table populated
# and the SSD-tier files on disk; this cell measures the steady-state warm
# path that benefits from both.
echo
echo "[note] skipping cold_reset between tier-on-cold and tier-on-warm"
echo "       (warm-tier benefit lives in the SSD files cell 2 just wrote)"
run_cell "tier-on-warm"

# --- summary --------------------------------------------------------------

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] $PYTHON_BIN bench/analyze_ssd_cache.py $RESULTS_ROOT"
  echo "[dry-run] done"
  exit 0
fi

echo "=== analyzing results ==="
set +e
"$PYTHON_BIN" bench/analyze_ssd_cache.py "$RESULTS_ROOT"
ANALYZE_RC=$?
set -e

echo
echo "=== done. raw results at $RESULTS_ROOT ==="
echo "    writeup: $RESULTS_ROOT/RESULTS.md"
if [[ "$ANALYZE_RC" -ne 0 ]]; then
  echo "    ANALYZE EXIT $ANALYZE_RC — likely a recall floor violation; see RESULTS.md" >&2
  exit "$ANALYZE_RC"
fi
