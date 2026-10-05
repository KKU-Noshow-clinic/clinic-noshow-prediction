## Sequential HTTP timing from the inside the api container

- Date: 2026-10-03 14:58
- Client: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.41, Python 3.11.16, 16 CPUs; target http://localhost:8000
- Method: one keep-alive connection, Python http.client, one request at a time, 20 warm-up rounds discarded, 200 timed rounds of each request

| Request | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| GET /health | 1.1 | 1.3 | 1.1 |
| POST /predict | 12.9 | 15.3 | 13.1 |

