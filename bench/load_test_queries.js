// k6 query-only load test.
//
// Reads a pre-built corpus (api_key + dataset list) at init via open(),
// then runs POST /v1/query for DURATION at VUS concurrent users. About
// 30% of queries carry a metadata filter.
//
// Custom metrics:
//   rb_query_latency           unfiltered queries (ms)
//   rb_filtered_query_latency  filtered queries (ms)
//   rb_query_errors            non-2xx rate
//   rb_queries_run             total counter
//
// Env:
//   BASE_URL   default http://cp:8080
//   DIM        default 128             (must match the corpus file)
//   VUS        default 10
//   DURATION   default 5m
//   CORPUS     default /bench/cache/dim-128.json

import http from 'k6/http';
import { Trend, Rate, Counter } from 'k6/metrics';

// open() is an init-context API; the file MUST exist when k6 starts.
// seed_corpus.py creates it before run_matrix.sh invokes k6.
const CORPUS = JSON.parse(open(__ENV.CORPUS || './cache/dim-128.json'));

const BASE_URL = __ENV.BASE_URL || 'http://cp:8080';
const DIM = parseInt(__ENV.DIM || '128', 10);
const VUS = parseInt(__ENV.VUS || '10', 10);
const DURATION = __ENV.DURATION || '5m';

const CATEGORIES = ['books', 'movies', 'music'];

const queryLatency = new Trend('rb_query_latency', true);
const filteredQueryLatency = new Trend('rb_filtered_query_latency', true);
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
    rb_query_latency: ['p(95)<750', 'p(99)<1500'],
    rb_filtered_query_latency: ['p(95)<1200'],
    rb_query_errors: ['rate<0.02'],
    http_req_failed: ['rate<0.05'],
  },
  noConnectionReuse: false,
  discardResponseBodies: true,
};

function randVec(d) {
  // Uniform [-1, 1] floats — matches seed_corpus.py.
  const v = new Array(d);
  for (let i = 0; i < d; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

function pickCorpus() {
  return CORPUS[Math.floor(Math.random() * CORPUS.length)];
}

export default function () {
  if (!CORPUS || CORPUS.length === 0) {
    throw new Error('empty corpus — did seed_corpus.py run?');
  }

  const t = pickCorpus();
  const useFilter = Math.random() < 0.3;
  const body = {
    dataset: t.dataset,
    vector: randVec(DIM),
    top_k: 10,
  };
  if (useFilter) {
    body.filter = {
      category: CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)],
    };
  }

  const res = http.post(`${BASE_URL}/v1/query`, JSON.stringify(body), {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${t.api_key}`,
    },
    tags: { filtered: useFilter ? 'yes' : 'no' },
  });

  queriesRun.add(1);
  const ok = res.status >= 200 && res.status < 300;
  queryErrors.add(!ok);
  if (useFilter) filteredQueryLatency.add(res.timings.duration);
  else queryLatency.add(res.timings.duration);
}

export function handleSummary(data) {
  // Write a compact one-pager next to k6's default text summary.
  return {
    stdout: textSummary(data),
    [__ENV.SUMMARY_PATH || '/bench/k6_summary.json']: JSON.stringify(data, null, 2),
  };
}

// Inline text-summary — avoids fetching k6-utils at runtime.
function textSummary(data) {
  const m = data.metrics;
  const t = (name) => m[name] ? m[name].values : {};
  const ql = t('rb_query_latency');
  const fql = t('rb_filtered_query_latency');
  const errs = t('rb_query_errors');
  const httpf = t('http_req_failed');
  const cnt = t('rb_queries_run');
  const fmt = (v) => v === undefined ? '-' : (typeof v === 'number' ? v.toFixed(2) : v);
  return [
    `--- bench summary (DIM=${DIM} VUS=${VUS} DURATION=${DURATION}) ---`,
    `queries           : ${fmt(cnt.count)}  (rate ${fmt(cnt.rate)} QPS)`,
    `query p50/p95/p99 : ${fmt(ql.med)} / ${fmt(ql['p(95)'])} / ${fmt(ql['p(99)'])} ms`,
    `filtered p95      : ${fmt(fql['p(95)'])} ms`,
    `error rate        : ${fmt((errs.rate || 0) * 100)}%`,
    `http_req_failed   : ${fmt((httpf.rate || 0) * 100)}%`,
    '',
  ].join('\n');
}
