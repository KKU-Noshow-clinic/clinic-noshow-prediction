# Load test report (Serving)

- Date: 2026-10-03 22:53
- Configuration: 4 workers + batching, inside Docker network
- Tool: Locust, 50 concurrent users, duration 2m
- Traffic mix: /predict 10 : /predict_batch (20 rows) 1 : /health 1
- Raw results: `loadtest/results/net50_w4b32_stats.csv`

## SLO check: 4 workers + batching, inside Docker network (targets from `configs/slo.yaml`)

| SLO | Measured | Target | Result |
|---|---:|---:|:---:|
| p50 latency /predict | 46 ms | <= 50 ms | PASS |
| p95 latency /predict | 110 ms | <= 200 ms | PASS |
| error rate (all requests) | 0.00% | <= 1.00% | PASS |
| availability | 100.00% | >= 99.00% | PASS |

**Overall: all SLOs met**

## Per-endpoint latency: 4 workers + batching, inside Docker network

| Endpoint | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| GET /health | 1393 | 11.7 | 9 | 31 | 71 | 0.00% |
| POST /predict | 13937 | 116.6 | 46 | 110 | 170 | 0.00% |
| POST /predict_batch | 1427 | 11.9 | 46 | 120 | 180 | 0.00% |
| Aggregated | 16757 | 140.2 | 43 | 110 | 170 | 0.00% |

<!-- notes: everything below is kept when this report is regenerated -->

## Analysis

(write the findings here)
