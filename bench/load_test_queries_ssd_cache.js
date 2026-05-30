// k6 query loader for the SSD-cache cold-vs-warm bench.
//
// Differs from load_test_queries.js in one way: it picks queries from a
// fixed ground-truth file rather than generating random vectors per-VU.
// Determinism matters because we want cells to receive the *same* query
// distribution — the cell-agreement check in analyze_ssd_cache.py
// compares per-cell responses to the baseline cell's responses for the
// same queries.
//
// This driver does NOT compute recall against brute-force ground truth.
// That comparison was removed when we discovered the bench's
// synthetic uniform-random corpus is exactly the workload IVFFlat
// performs worst on (no cluster structure to partition) — recall vs
// brute force was near-zero for reasons unrelated to the SSD-cache
// layer. The correctness check happens AFTER the cell finishes, via
// `bench/capture_query_results.py` (a separate Python step that records
// per-query response IDs without competing with the load test).
//
// Env:
//   BASE_URL         default http://cp:8080
//   VUS              default 20
//   DURATION         default 1m
//   CORPUS           required — path to seed_corpus.py output (single-tenant)
//   GROUND_TRUTH     required — path to build_ground_truth.py output
//                                (we only use queries[].vector here; the
//                                 top_k_ids field is unused in the driver
//                                 but consumed by capture_query_results.py)
//   SUMMARY_PATH     default /bench/k6_summary.json
//   RB_CELL_NAME     optional label, surfaced in stdout summary only

import http from 'k6/http';
import { Trend, Rate, Counter } from 'k6/metrics';

// open() is an init-context API; both files MUST exist when k6 starts.
const CORPUS = JSON.parse(open(__ENV.CORPUS || './cache/dim-128.json'));
const GROUND_TRUTH = JSON.parse(open(__ENV.GROUND_TRUTH || './cache/ground-truth-dim-128.json'));

const BASE_URL = __ENV.BASE_URL || 'http://cp:8080';
const VUS = parseInt(__ENV.VUS || '20', 10);
const DURATION = __ENV.DURATION || '1m';
const CELL_NAME = __ENV.RB_CELL_NAME || 'unnamed';

const TOP_K = GROUND_TRUTH.k || 10;

const queryLatency = new Trend('rb_query_latency', true);
const queryErrors = new Rate('rb_query_errors');
const queriesRun = new Counter('rb_queries_run');

export const options = {
  scenarios: {
    query_stress: {
      executor: 'constant-vus',
      vus: VUS,
      duration: DURATION,
    },
  },
  thresholds: {
    // Loose latency thresholds — the SSD-cache bench is about cold vs warm
    // deltas, not absolute SLOs. Cell agreement (the load-bearing
    // correctness check) is enforced by analyze_ssd_cache.py against the
    // post-cell capture_query_results.py output, NOT inside the k6 driver.
    rb_query_errors: ['rate<0.05'],
    http_req_failed: ['rate<0.05'],
  },
  // Body parsing is unnecessary here (the driver doesn't inspect matches).
  // capture_query_results.py runs a separate, quiescent query set AFTER
  // this cell finishes; the agreement check works on that capture.
  discardResponseBodies: true,
  noConnectionReuse: false,
};

function pickCorpus() {
  return CORPUS[Math.floor(Math.random() * CORPUS.length)];
}

function pickQuery() {
  return GROUND_TRUTH.queries[Math.floor(Math.random() * GROUND_TRUTH.queries.length)];
}

export default function () {
  if (!CORPUS || CORPUS.length === 0) {
    throw new Error('empty corpus — did seed_corpus.py run?');
  }
  if (!GROUND_TRUTH.queries || GROUND_TRUTH.queries.length === 0) {
    throw new Error('empty ground-truth queries — did build_ground_truth.py run?');
  }

  const t = pickCorpus();
  const q = pickQuery();
  const body = {
    dataset: t.dataset,
    vector: q.vector,
    top_k: TOP_K,
  };

  const res = http.post(`${BASE_URL}/v1/query`, JSON.stringify(body), {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${t.api_key}`,
    },
    tags: { cell: CELL_NAME },
  });

  queriesRun.add(1);
  const ok = res.status >= 200 && res.status < 300;
  queryErrors.add(!ok);
  queryLatency.add(res.timings.duration);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data),
    [__ENV.SUMMARY_PATH || '/bench/k6_summary.json']: JSON.stringify(data, null, 2),
  };
}

// Inline text summary — avoids fetching k6-utils at runtime.
function textSummary(data) {
  const m = data.metrics;
  const t = (name) => (m[name] ? m[name].values : {});
  const ql = t('rb_query_latency');
  const errs = t('rb_query_errors');
  const httpf = t('http_req_failed');
  const cnt = t('rb_queries_run');
  const fmt = (v) =>
    v === undefined ? '-' : typeof v === 'number' ? v.toFixed(3) : v;
  return [
    `--- ssd-cache bench cell=${CELL_NAME} (VUS=${VUS} DURATION=${DURATION}) ---`,
    `queries           : ${fmt(cnt.count)}  (rate ${fmt(cnt.rate)} QPS)`,
    `query p50/p95/p99 : ${fmt(ql.med)} / ${fmt(ql['p(95)'])} / ${fmt(ql['p(99)'])} ms`,
    `error rate        : ${fmt((errs.rate || 0) * 100)}%`,
    `http_req_failed   : ${fmt((httpf.rate || 0) * 100)}%`,
    '',
  ].join('\n');
}
