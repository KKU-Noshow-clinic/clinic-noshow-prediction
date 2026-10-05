## Under load: 50 users (prof50)

- Date: 2026-10-03 22:30
- Host: Windows-10-10.0.26200-SP0, AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD, 16 logical CPUs
- Docker engine: 16 CPUs, 7933870080 bytes RAM, Docker Desktop, kernel 5.15.167.4-microsoft-standard-WSL2
- API: 2 uvicorn worker(s), model v1, micro-batching as configured (API_BATCH), SERVING_PROFILE=1
- Method: Locust 50 users, spawn 10/s, 2m, traffic mix /predict 10 : /predict_batch (20 rows) 1 : /health 1. Server stages from Prometheus histogram deltas before/after the run (percentiles interpolated inside buckets, so approximate); client numbers from Locust.
- Requests: 13126 /predict, 0 failed; 7716 pipeline calls, 5.2 rows per call on average

| Where | Count | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|---:|
| **client sees** /predict (Locust) | 13126 | 66 | 120 | 70.2 |
| inside app, whole /predict request | 13167 | 54.2 | 113.9 | 59.6 |
| waiting for micro-batch (per request) | 14526 | 10.2 | 28.8 | 10.7 |
| build DataFrame (per call) | 7716 | 1.8 | 4.0 | 1.9 |
| feature builder (per call) | 7716 | 10.2 | 17.9 | 10.7 |
| ColumnTransformer (per call) | 7716 | 9.7 | 16.4 | 10.3 |
| LightGBM predict_proba (per call) | 7716 | 1.9 | 6.6 | 2.6 |
| inside app, whole /predict_batch request | 1359 | 53.1 | 118.8 | 59.9 |

HTTP + network + client overhead (mean client - mean in-app) ~ **10.6 ms**. Per-call stages are shared by all rows in a micro-batch, so they do not add up to per-request latency under load.
