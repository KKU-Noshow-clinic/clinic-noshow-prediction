# Load test report (Serving)

- Date: 2026-10-03 22:09
- Configuration: 50 users, inside Docker network, after warm-up
- Tool: Locust, 50 concurrent users, duration 2m
- Traffic mix: /predict 10 : /predict_batch (20 rows) 1 : /health 1
- Raw results: `loadtest/results/net50b_stats.csv`

## SLO check: 50 users, inside Docker network, after warm-up (targets from `configs/slo.yaml`)

| SLO | Measured | Target | Result |
|---|---:|---:|:---:|
| p50 latency /predict | 45 ms | <= 50 ms | PASS |
| p95 latency /predict | 80 ms | <= 200 ms | PASS |
| error rate (all requests) | 0.00% | <= 1.00% | PASS |
| availability | 100.00% | >= 99.00% | PASS |

**Overall: all SLOs met**

## Per-endpoint latency: 50 users, inside Docker network, after warm-up

| Endpoint | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| GET /health | 1421 | 11.9 | 11 | 24 | 35 | 0.00% |
| POST /predict | 14224 | 119.1 | 45 | 80 | 120 | 0.00% |
| POST /predict_batch | 1369 | 11.5 | 47 | 78 | 110 | 0.00% |
| Aggregated | 17014 | 142.5 | 44 | 78 | 110 | 0.00% |

<!-- notes: everything below is kept when this report is regenerated -->

## Analysis

(write the findings here)
