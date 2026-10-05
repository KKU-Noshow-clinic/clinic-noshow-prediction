## Sequential HTTP timing from the host (through Docker port forwarding)

- Date: 2026-10-03 21:58
- Client: Windows-10-10.0.26200-SP0, Python 3.11.9, 16 CPUs; target http://localhost:8000
- Method: one keep-alive connection, Python http.client, one request at a time, 20 warm-up rounds discarded, 200 timed rounds of each request

| Request | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| GET /health | 1.5 | 1.9 | 1.6 |
| POST /predict | 58.7 | 68.4 | 59.5 |

