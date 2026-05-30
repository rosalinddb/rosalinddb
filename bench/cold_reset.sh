#!/usr/bin/env bash
# Cold reset for the bench harness — forces the next query to pay the cold path.
#
# What "cold" means for RosalindDB:
#   1. The query_dp + ephemeral_runner in-process state — the shard cache map
#      (`_SHARD_CACHE` in services/query_api/v1_query.py), the FAISS-mmap'd
#      file descriptors, and any pending coalesced downloads — all live in
#      the process's address space. Restarting the two containers is the only
#      way to drop them.
#   2. The OS page cache backing the local SSD tier files (when
#      RB_SHARD_TIER_BYTES is set, downloaded shards live as files under the
#      container's data dir). On Linux a `drop_caches` evicts them from the
#      VM's page cache; on macOS the host's drop_caches does NOT reach the
#      Docker VM's page cache, so the container restart above is the
#      load-bearing reset for any portable harness.
#
# This script restarts the two containers unconditionally and attempts
# drop_caches only when `--drop-caches` is passed AND the kernel exposes
# /proc/sys/vm/drop_caches (Linux). On macOS the drop_caches call is a
# silent no-op — documented, not pretended away.
#
# Usage:
#   bash bench/cold_reset.sh                         # container restart only
#   bash bench/cold_reset.sh --drop-caches           # + Linux page cache drop
#   COMPOSE_PROJECT=rosalinddb-bench \
#       bash bench/cold_reset.sh --drop-caches       # custom compose project
#
# The script writes a one-line log of what it actually did to stdout so the
# bench cell output captures the reset. Exit code is non-zero only when the
# container restart itself fails; a no-op drop_caches on macOS does not fail.

set -euo pipefail

# --- args ------------------------------------------------------------------

DROP_CACHES=0
for arg in "$@"; do
  case "$arg" in
    --drop-caches) DROP_CACHES=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
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

PROJECT="${COMPOSE_PROJECT:-${BENCH_PROJECT:-rosalinddb-bench}}"
COMPOSE=(docker compose -p "$PROJECT" -f docker-compose.yml -f bench/docker-compose.bench.yml)

TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[cold_reset $TS] project=$PROJECT drop_caches=$DROP_CACHES"

# --- step 1: drop_caches (Linux only) -------------------------------------

DROP_RESULT="skipped"
if [[ "$DROP_CACHES" -eq 1 ]]; then
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    # Already root (CI runner). No sudo needed.
    sync
    echo 3 > /proc/sys/vm/drop_caches
    DROP_RESULT="dropped (direct write)"
  elif [[ -e /proc/sys/vm/drop_caches ]]; then
    # Linux but not root; try sudo. Silent on failure — the harness's
    # cold guarantee is the container restart, not drop_caches.
    if sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
      DROP_RESULT="dropped (sudo)"
    else
      DROP_RESULT="sudo unavailable — host page cache not flushed"
    fi
  else
    # macOS / Windows: drop_caches doesn't exist. The Docker VM has its own
    # page cache that the host can't reach; the container restart below is
    # what actually gives a cold tier file. Be explicit about it.
    DROP_RESULT="no /proc/sys/vm/drop_caches (macOS/Windows host)"
  fi
fi
echo "[cold_reset] drop_caches: $DROP_RESULT"

# --- step 2: restart the FAISS-loading containers ------------------------

# Both query_dp and ephemeral_runner import the shard-cache module from
# services/query_api/v1_query.py at process start; the cache state lives in
# module-level dicts. `up -d --force-recreate` cycles the container PID,
# which is the only deterministic way to clear that state.
#
# `up -d` (without --no-deps) is intentional — see run_mmap_comparison.sh
# for the rationale: the FAISS-loading services have depends_on edges to
# already-completed one-shot services (migrator, createbuckets) and
# --no-deps trips on those.
echo "[cold_reset] restarting query_dp + ephemeral_runner"
RESTART_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
"${COMPOSE[@]}" up -d --force-recreate query_dp ephemeral_runner

# Give the DP a few seconds to load the FAISS catalog before we hand back
# to the caller. 5s matches run_mmap_comparison.sh's settle.
sleep 5
RESTART_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[cold_reset] restart window: $RESTART_START..$RESTART_END"
echo "[cold_reset] done"
