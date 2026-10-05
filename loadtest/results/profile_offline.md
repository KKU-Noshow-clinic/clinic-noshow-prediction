## Offline stage breakdown (inside the api container, no HTTP, no load)

- Date: 2026-10-02 17:03
- Model: `models:/clinic-noshow@champion` v1
- Container: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.41, Python 3.11.16, 16 CPUs visible, cgroup quota unknown
- OMP_NUM_THREADS=1, WEB_CONCURRENCY=2 (workers keep running idle while this measures)
- pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0
- Method: single process, sequential calls, 30 warm-up calls discarded, then 300 timed calls per batch size; time.perf_counter per stage. `total` = the API's normal path (one predict_proba) timed separately.

| Rows/call | Stage | p50 (ms) | p95 (ms) | mean (ms) | share of stages |
|---:|---|---:|---:|---:|---:|
| 1 | schema | 0.1 | 0.1 | 0.1 |  |
| 1 | frame | 0.9 | 1.3 | 1.0 | 6% |
| 1 | preprocess.features | 7.0 | 9.3 | 7.3 | 43% |
| 1 | preprocess.columns | 7.4 | 9.9 | 7.6 | 44% |
| 1 | model | 1.2 | 1.6 | 1.3 | 7% |
| 1 | total | 17.0 | 21.6 | 17.5 |  |
| 8 | schema | 0.2 | 0.3 | 0.2 |  |
| 8 | frame | 1.0 | 1.4 | 1.0 | 6% |
| 8 | preprocess.features | 7.2 | 10.0 | 7.5 | 42% |
| 8 | preprocess.columns | 7.6 | 11.0 | 8.0 | 45% |
| 8 | model | 1.2 | 1.7 | 1.3 | 7% |
| 8 | total | 17.7 | 23.4 | 18.4 |  |
| 32 | schema | 0.5 | 0.9 | 0.6 |  |
| 32 | frame | 1.1 | 1.6 | 1.1 | 6% |
| 32 | preprocess.features | 7.6 | 10.4 | 7.9 | 42% |
| 32 | preprocess.columns | 7.8 | 11.2 | 8.3 | 44% |
| 32 | model | 1.3 | 1.8 | 1.3 | 7% |
| 32 | total | 18.7 | 26.2 | 19.9 |  |

Step-by-step scores were identical to predict_proba on every call.

