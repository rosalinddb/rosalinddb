// k6 multi-agent memory load test — concurrent agents bombarding the recall
// (read-your-writes) path.
//
// N VUs == N concurrent agents. Each VU loops:
//   (a) WRITE  a batch of MEMORIES_PER random DIM-dim memory vectors to the
//       recall tier (POST /v1/datasets/{ds}/vectors, NDJSON). With RB_RECALL on
//       this is synchronous (HTTP 200) and immediately queryable.
//   (b) SEARCH POST /v1/query against the agent's own memories.
//   (c) READ-YOUR-WRITES PROBE: write a sentinel vector with a known id, then
//       immediately query with that exact vector and record whether the
//       sentinel id comes back (hit) and the round-trip lag.
//
// MODE selects the comparison:
//   per-agent (default, recommended) — each VU owns its OWN dataset. Models
//       agents as isolated memory stores; the recall scan per query is scoped
//       to that small partition.
//   shared    — ALL VUs write to ONE shared dataset, tagging each record's
//       metadata with {agent_id: <vu>}; searches filter {agent_id: <vu>}
//       (server-side, exhaustive). Exposes the scaling cliff as the shared
//       partition grows and every query brute-force-scans the whole thing.
//
// Custom metrics:
//   rb_write_latency   Trend (ms)   write-batch latency (SUCCESSFUL 2xx only)
//   rb_writes          Counter      accepted BATCH memory-writes -> throughput
//                                   ops/s. This is the recall-WRITE LOAD only;
//                                   it EXCLUDES the per-iteration sentinel probe
//                                   write (a correctness check, ~1/(MEMORIES_PER+1)
//                                   ~17% of write ops at the default 5). So
//                                   rb_writes is batch-memory throughput, NOT
//                                   total recall write ops/s.
//   rb_search_latency  Trend (ms)   query latency (SUCCESSFUL 2xx only)
//   rb_ryw_hit         Rate         read-your-writes: sentinel id returned?
//                                   (failed sentinel write OR failed/empty probe
//                                   query counts as a MISS)
//   rb_ryw_lag_ms      Trend (ms)   read-your-writes round-trip lag (genuine
//                                   HITS only)
//   rb_errors          Rate         non-2xx across all request kinds
//
// Env:
//   BASE_URL       default http://cp:8080
//   MODE           per-agent | shared           (default per-agent)
//   AGENTS         number of VUs (agents)        (default 10)
//   DIM            vector dimension              (default 768)
//   MEMORIES_PER   memory vectors per write batch(default 5)
//   TOP_K          query top_k                   (default 10)
//   DURATION       k6 run duration               (default 2m)
//   DATASET_PREFIX dataset name prefix           (default agentmem)
//   SUMMARY_PATH   where handleSummary writes json (default /bench/k6_summary.json)

import http from 'k6/http';
import { Trend, Rate, Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://cp:8080';
const MODE = (__ENV.MODE || 'per-agent').toLowerCase();
const AGENTS = parseInt(__ENV.AGENTS || '10', 10);
const DIM = parseInt(__ENV.DIM || '768', 10);
const MEMORIES_PER = parseInt(__ENV.MEMORIES_PER || '5', 10);
const TOP_K = parseInt(__ENV.TOP_K || '10', 10);
const DURATION = __ENV.DURATION || '2m';
const DATASET_PREFIX = __ENV.DATASET_PREFIX || 'agentmem';
const SHARED_DATASET = `${DATASET_PREFIX}_shared`;

if (MODE !== 'per-agent' && MODE !== 'shared') {
  throw new Error(`MODE must be 'per-agent' or 'shared', got '${MODE}'`);
}

const writeLatency = new Trend('rb_write_latency', true);
const writes = new Counter('rb_writes');
const searchLatency = new Trend('rb_search_latency', true);
const rywHit = new Rate('rb_ryw_hit');
const rywLag = new Trend('rb_ryw_lag_ms', true);
const errors = new Rate('rb_errors');

export const options = {
  scenarios: {
    agents: {
      executor: 'constant-vus',
      vus: AGENTS,
      duration: DURATION,
    },
  },
  thresholds: {
    // Synchronous recall write + immediate query is the read-your-writes
    // promise: a write should be visible to its own query essentially always.
    rb_ryw_hit: ['rate>0.99'],
    rb_write_latency: ['p(95)<1500'],
    rb_search_latency: ['p(95)<1500'],
    rb_errors: ['rate<0.02'],
    http_req_failed: ['rate<0.05'],
  },
  noConnectionReuse: false,
  // We must read query response bodies to score the read-your-writes probe, so
  // do NOT discard them globally.
  discardResponseBodies: false,
};

// Uniform [-1, 1] floats — matches seed_corpus.py / load_test_queries.js.
function randVec(d) {
  const v = new Array(d);
  for (let i = 0; i < d; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

function authHeaders(extra) {
  // RB_REQUIRE_AUTH=false on the recall-bench stack — every request resolves to
  // the single built-in `default` tenant, so no Authorization header is needed.
  return Object.assign({ 'Content-Type': 'application/json' }, extra || {});
}

// The dataset this VU operates on. In per-agent mode each VU owns its own; in
// shared mode all VUs share one.
function datasetFor(vu) {
  return MODE === 'shared' ? SHARED_DATASET : `${DATASET_PREFIX}_a${vu}`;
}

// Create a dataset; tolerate 409 (already exists) so first-touch is idempotent
// across VUs (shared mode) and across the warmup re-check. A 409 is an
// EXPECTED status here (the dataset already exists — fine on a --keep-up re-run
// against a populated stack), so we mark 200/201/409 as expected via a
// per-request responseCallback. Otherwise k6 would count the setup() 409 toward
// the built-in http_req_failed rate even though it is not a real failure.
const datasetExistsOk = http.expectedStatuses(200, 201, 409);
function ensureDataset(name) {
  const res = http.post(
    `${BASE_URL}/v1/datasets`,
    JSON.stringify({ name, dimension: DIM }),
    { headers: authHeaders(), responseCallback: datasetExistsOk }
  );
  // 200/201 created, 409 already exists — both are fine.
  return res.status === 200 || res.status === 201 || res.status === 409;
}

// setup() runs ONCE before the VUs spin up. Pre-create the datasets so the
// per-VU loop never races a create against its first write.
export function setup() {
  if (MODE === 'shared') {
    if (!ensureDataset(SHARED_DATASET)) {
      throw new Error(`setup: failed to create shared dataset ${SHARED_DATASET}`);
    }
  } else {
    for (let vu = 1; vu <= AGENTS; vu++) {
      if (!ensureDataset(`${DATASET_PREFIX}_a${vu}`)) {
        throw new Error(`setup: failed to create dataset for agent ${vu}`);
      }
    }
  }
  return { mode: MODE, dim: DIM, agents: AGENTS };
}

// Build an NDJSON body of `n` memory vectors. In shared mode every record is
// tagged with the agent_id so searches can filter it back out server-side.
function memoryBatch(vu, n, idPrefix) {
  const lines = new Array(n);
  for (let i = 0; i < n; i++) {
    const rec = {
      id: `${idPrefix}-${i}`,
      values: randVec(DIM),
      metadata: { agent_id: vu, kind: 'memory' },
    };
    lines[i] = JSON.stringify(rec);
  }
  return lines.join('\n');
}

export default function (data) {
  const vu = __VU;
  const ds = datasetFor(vu);

  // --- (a) WRITE a batch of memory vectors -------------------------------
  const batchId = `m-${vu}-${__ITER}`;
  const body = memoryBatch(vu, MEMORIES_PER, batchId);
  const wres = http.post(`${BASE_URL}/v1/datasets/${ds}/vectors`, body, {
    headers: authHeaders({ 'Content-Type': 'application/x-ndjson' }),
    tags: { op: 'write' },
  });
  const wok = wres.status === 200; // recall ON -> synchronous 200
  errors.add(!wok, { op: 'write' });
  if (wok) {
    // Record latency ONLY on success so the published p50/p95/p99 are over
    // successful requests (the standard convention). Error-response latency
    // (a fast fast-fail or a slow timeout at the scaling cliff) would otherwise
    // contaminate the percentiles the per-agent-vs-shared comparison hinges on.
    // Failures still count toward rb_errors / http_req_failed above.
    writeLatency.add(wres.timings.duration);
    // Count accepted records for a true write-throughput number.
    let accepted = MEMORIES_PER;
    try {
      const j = wres.json();
      if (j && typeof j.accepted === 'number') accepted = j.accepted;
    } catch (_e) { /* keep the optimistic count */ }
    writes.add(accepted);
  }

  // --- (b) SEARCH the agent's own memories -------------------------------
  const sbody = {
    dataset: ds,
    vector: randVec(DIM),
    top_k: TOP_K,
  };
  if (MODE === 'shared') {
    // Exhaustive server-side filter back to this agent's own rows.
    sbody.filter = { agent_id: vu };
  }
  const sres = http.post(`${BASE_URL}/v1/query`, JSON.stringify(sbody), {
    headers: authHeaders(),
    tags: { op: 'search' },
  });
  const sok = sres.status >= 200 && sres.status < 300;
  errors.add(!sok, { op: 'search' });
  // Same as the write path: record search latency only on a 2xx so percentiles
  // are over successful queries. Failures count toward rb_errors / http_req_failed.
  if (sok) searchLatency.add(sres.timings.duration);

  // --- (c) READ-YOUR-WRITES probe ---------------------------------------
  // Write a sentinel with a known id + known vector, then query with that
  // exact vector and confirm the sentinel id is in the matches.
  const sentinelId = `sentinel-${vu}-${__ITER}`;
  const sentinelVec = randVec(DIM);
  const sentinelRec = JSON.stringify({
    id: sentinelId,
    values: sentinelVec,
    metadata: { agent_id: vu, kind: 'sentinel' },
  });
  const t0 = Date.now();
  const pwres = http.post(`${BASE_URL}/v1/datasets/${ds}/vectors`, sentinelRec, {
    headers: authHeaders({ 'Content-Type': 'application/x-ndjson' }),
    tags: { op: 'ryw_write' },
  });
  const pwok = pwres.status === 200;
  errors.add(!pwok, { op: 'ryw_write' });

  if (!pwok) {
    // A failed sentinel write means the write was never visible — that is a
    // read-your-writes MISS, not an absence of evidence. Recording nothing here
    // would inflate the hit-rate by exclusion. No genuine hit -> no lag sample.
    rywHit.add(false);
  } else {
    const pbody = {
      dataset: ds,
      vector: sentinelVec,
      top_k: TOP_K,
    };
    if (MODE === 'shared') pbody.filter = { agent_id: vu };
    const pqres = http.post(`${BASE_URL}/v1/query`, JSON.stringify(pbody), {
      headers: authHeaders(),
      tags: { op: 'ryw_query' },
    });
    const pqok = pqres.status >= 200 && pqres.status < 300;
    errors.add(!pqok, { op: 'ryw_query' });

    let hit = false;
    if (pqok) {
      try {
        const j = pqres.json();
        const matches = (j && j.matches) || [];
        for (let i = 0; i < matches.length; i++) {
          if (matches[i] && matches[i].id === sentinelId) { hit = true; break; }
        }
      } catch (_e) { hit = false; }
    }
    // A failed/empty probe query is a MISS (the write was not read back), so it
    // counts toward the hit-rate as false. Record the lag Trend ONLY on a
    // genuine hit (sentinel id actually returned) — a 503 or empty round-trip
    // must not land in the lag distribution.
    rywHit.add(hit);
    if (hit) rywLag.add(Date.now() - t0);
  }
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data),
    [__ENV.SUMMARY_PATH || '/bench/k6_summary.json']: JSON.stringify(data, null, 2),
  };
}

// Inline text-summary — avoids fetching k6-utils at runtime.
function textSummary(data) {
  const m = data.metrics;
  const t = (name) => (m[name] ? m[name].values : {});
  const wl = t('rb_write_latency');
  const sl = t('rb_search_latency');
  const w = t('rb_writes');
  const ryw = t('rb_ryw_hit');
  const lag = t('rb_ryw_lag_ms');
  const errs = t('rb_errors');
  const httpf = t('http_req_failed');
  const fmt = (v) =>
    v === undefined ? '-' : typeof v === 'number' ? v.toFixed(2) : v;
  return [
    `--- agent-memory bench (MODE=${MODE} AGENTS=${AGENTS} DIM=${DIM} MEMORIES_PER=${MEMORIES_PER} DURATION=${DURATION}) ---`,
    `writes (batch)    : ${fmt(w.count)}  (throughput ${fmt(w.rate)} ops/s, excl. sentinel probe)`,
    `write p50/p95/p99 : ${fmt(wl.med)} / ${fmt(wl['p(95)'])} / ${fmt(wl['p(99)'])} ms`,
    `search p50/p95/p99: ${fmt(sl.med)} / ${fmt(sl['p(95)'])} / ${fmt(sl['p(99)'])} ms`,
    `read-your-writes  : hit-rate ${fmt((ryw.rate || 0) * 100)}%  lag p95 ${fmt(lag['p(95)'])} ms`,
    `error rate        : ${fmt((errs.rate || 0) * 100)}%`,
    `http_req_failed   : ${fmt((httpf.rate || 0) * 100)}%`,
    '',
  ].join('\n');
}
