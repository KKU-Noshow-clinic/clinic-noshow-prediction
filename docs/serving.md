# Serving & Infra (Issue #3)

ผู้รับผิดชอบ: ภูรินทร์ ศรีฐาน

## ภาพรวม

API โหลดโมเดล **champion** จาก MLflow Registry (`models:/clinic-noshow@champion`) ซึ่งเป็น
sklearn Pipeline ทั้งก้อน (feature builder → preprocessing → LightGBM Exp 2) จาก workstream Model/Registry
ดังนั้น preprocessing ตอน serve คือโค้ดชุดเดียวกับตอน train (`src/noshow/features/`) ไม่มี training-serving skew

threshold อ่านจาก tag `threshold` ของ model version เดียวกัน (ตั้งโดย `registry/manage.py`)
ถ้าไม่มี tag จะใช้ `model.selected_threshold` ใน `configs/params.yaml` และ override ได้ด้วย env `MODEL_THRESHOLD`

| ไฟล์ | หน้าที่ |
|---|---|
| `src/noshow/serving/app.py` | FastAPI endpoints + Prometheus metrics |
| `src/noshow/serving/schemas.py` | Pydantic schema ของ input/output และกฎ validation |
| `src/noshow/serving/model.py` | โหลดโมเดล/threshold จาก registry, ทำนาย, reload, ตรวจ champion ทุก 30 วิ |
| `src/noshow/serving/batching.py` | micro-batching: รวม request ที่มาพร้อมกันเป็นการเรียก pipeline ครั้งเดียว |
| `src/noshow/serving/profiling.py` | จับเวลาแต่ละขั้น (เปิดด้วย `SERVING_PROFILE=1`) |
| `tests/test_serving_model.py`, `tests/test_serving_api.py` | unit + API tests (ใช้โมเดลจำลอง ไม่ต้องมี MLflow) |
| `loadtest/locustfile.py` | Locust load test |
| `scripts/check_slo.py` | เทียบผล load test กับ `configs/slo.yaml` → `docs/loadtest_report.md` |

## Endpoints

| Method | Path | ใช้ทำอะไร |
|---|---|---|
| GET | `/health` | liveness + version/threshold ของโมเดลที่โหลดอยู่ |
| POST | `/predict` | ทำนาย 1 นัดหมาย |
| POST | `/predict_batch` | ทำนาย 1–1000 นัดหมาย |
| POST | `/reload` | โหลด champion ใหม่ทันที**ใน worker ที่รับ request** (worker อื่นตามมาเองภายใน `MODEL_POLL_SECONDS`) |
| GET | `/metrics` | Prometheus metrics |

ตัวอย่าง:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "PatientId": 29872499824296, "Gender": "F",
  "ScheduledDay": "2016-05-04T08:00:00Z", "AppointmentDay": "2016-05-10",
  "Age": 45, "Neighbourhood": "JARDIM DA PENHA",
  "Scholarship": 0, "Hipertension": 1, "Diabetes": 0, "Alcoholism": 0,
  "Handcap": 0, "SMS_received": 1}'
# {"PatientId": 29872499824296.0, "noshow_score": 0.61, "alert": true, "threshold": 0.5, "model_version": "3"}
```

`noshow_score` มาจากโมเดลที่ใช้ class weight จึง**ไม่ใช่ความน่าจะเป็นที่ calibrate แล้ว**
ใช้จัดลำดับ/ตัดสินใจส่งเตือน (`alert`) ไม่ควรแสดงเป็น "โอกาสไม่มา xx%"

### กฎ validation (ตอบ 422 เมื่อผิด)

- ใช้ชื่อคอลัมน์เดียวกับข้อมูลดิบ ห้ามส่ง `No-show` และห้ามมี field เกิน
- `Gender` ∈ {F, M}, `Age` 0–120, flag ต่างๆ เป็น 0/1, `Handcap` 0–4
- `ScheduledDay` ต้องไม่หลัง `AppointmentDay` (กฎเดียวกับ `data/clean.py`)
- เวลาที่ไม่มี timezone ถือเป็น UTC เหมือนข้อมูล train; `Handcap` 2–4 ถูกยุบเป็น 1 เหมือนตอน clean
- `Neighbourhood` ถูกแปลงเป็นตัวพิมพ์ใหญ่; ค่าที่ไม่เคยเห็นยังทำนายได้ (`handle_unknown="ignore"`)

### พฤติกรรมเมื่อยังไม่มีโมเดล

API เปิดได้แม้ MLflow ยังไม่มี champion: thread เบื้องหลังจะลองโหลดใหม่ทุก 15 วินาที
ระหว่างนั้น `/predict` ตอบ **503** และ `/health` แสดง `model_loaded: false` พร้อม `last_error`
ถ้า `/reload` ล้มเหลว API จะใช้โมเดลเดิมต่อ (ไม่ล่ม)

แต่ละ worker เช็ก registry ทุก `MODEL_POLL_SECONDS` (30 วิ) ถ้า alias `champion` ย้ายไป version อื่น
(promote หรือ rollback) จะโหลดใหม่เอง จำเป็นเพราะมีหลาย worker แต่ `POST /reload` ไปถึง worker เดียว

## การตั้งค่า (env)

| ตัวแปร | ค่าเริ่มต้น | หมายเหตุ |
|---|---|---|
| `MLFLOW_TRACKING_URI` | – | ใน compose = `http://mlflow:5000` |
| `MODEL_URI` | `models:/clinic-noshow@champion` | ระบุ version ตรงได้ เช่น `models:/clinic-noshow/2` |
| `MODEL_PATH` | – | path โฟลเดอร์ MLflow model ในเครื่อง (dev แบบไม่มี server) |
| `MODEL_THRESHOLD` | tag ของ version | override threshold |
| `MODEL_LOAD_ON_STARTUP` | `1` | ตั้ง `0` เพื่อไม่โหลดตอนเริ่ม |
| `MODEL_POLL_SECONDS` | `30` | ความถี่ที่แต่ละ worker เช็กว่า champion เปลี่ยนไหม |
| `WEB_CONCURRENCY` | `2` (Docker) | จำนวน uvicorn worker process; ใน compose ตั้งจาก shell ด้วย `API_WORKERS` |
| `MICROBATCH_MAX_SIZE` | `32` | จำนวนแถวสูงสุดต่อการเรียก pipeline; `1` = ปิด batching; ใน compose ใช้ `API_BATCH` |
| `MICROBATCH_MAX_WAIT_MS` | `0` | เวลารอ request เพิ่มก่อนทำนาย (0 = ไม่รอ, ไม่เพิ่ม latency ตอนโหลดน้อย) |
| `OMP_NUM_THREADS` | `1` (Docker) | LightGBM ใช้ 1 thread ต่อการเรียก กันหลาย worker แย่ง core |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus` (Docker) | ให้ `/metrics` รวมตัวเลขจากทุก worker |
| `SERVING_PROFILE` | `0` | `1` = จับเวลาแต่ละขั้นของ pipeline (ใน compose ใช้ `API_PROFILE`) |

## วิธีรัน

```bash
make up                       # API + MLflow + Prefect + Prometheus + Grafana
make pipeline                 # train -> register -> gate -> promote champion
curl -X POST localhost:8000/reload   # หรือรอ background loader
curl localhost:8000/health
```

รันแบบไม่ใช้ Docker: `MLFLOW_TRACKING_URI=http://localhost:5001 make api`

Demo rollback: รัน rollback ของ registry แล้วรอ ≤30 วิ (หรือ `POST /reload`) → `/health` จะแสดง version ก่อนหน้า
(`/health` มี `pid` บอกว่า worker ไหนตอบ ยิงหลายครั้งจะเห็นทุก worker)

## Load test และ SLO

```bash
make loadtest                          # 50 users, ramp 10/s, 2 นาที
make loadtest LT_USERS=100 LT_TIME=5m  # ปรับได้
```

ผลดิบอยู่ที่ `loadtest/results/run_stats.csv` และสรุป PASS/FAIL อยู่ที่ `docs/loadtest_report.md`
traffic mix = `/predict` 10 : `/predict_batch` (20 แถว) 1 : `/health` 1

| SLO (`configs/slo.yaml`) | เป้าหมาย | วัดจาก |
|---|---|---|
| Latency p50 | ≤ 50 ms | `/predict` ใน Locust |
| Latency p95 | ≤ 200 ms | `/predict` ใน Locust |
| Error rate | ≤ 1% | ทุก request ใน Locust |
| Availability | ≥ 99% | ทุก request ใน Locust (production: จาก Prometheus) |

## Performance (optimize ก่อน merge PR #15)

### สิ่งที่พบ (profile)

เวลาเกือบทั้งหมดของการทำนายเป็น**ค่าใช้จ่ายคงที่ต่อการเรียก pipeline 1 ครั้ง** (pandas/sklearn:
แปลงวันที่, ColumnTransformer, validation) ไม่ขึ้นกับจำนวนแถว: 1 แถวกับ 32 แถวใช้เวลาใกล้กัน
ส่วน LightGBM เองเร็วมาก ทดลอง `OMP_NUM_THREADS=1` แบบ worker เดียวแล้ว latency ไม่เปลี่ยน
(ตัดสมมติฐานว่า LightGBM ช้าออก)

### สิ่งที่ทำ (ไม่แตะโค้ด preprocessing / ผลโมเดลเหมือนเดิม)

1. **Uvicorn workers** (`WEB_CONCURRENCY`) ใช้หลาย core แก้ปัญหาคิวเมื่อมีผู้ใช้พร้อมกันหลายคน
   ต้องแก้ 2 เรื่องตามมา: metrics รวมทุก worker (Prometheus multiprocess) และให้ทุก worker ตาม champion เอง
2. **Micro-batching** (`serving/batching.py`) request ที่เข้ามาระหว่างที่ worker กำลังทำนาย
   จะถูกรวมเป็นการเรียก pipeline ครั้งเดียว (สูงสุด 32 แถว) ลดจำนวนครั้งที่ต้องจ่ายค่าคงที่
   ตอนมี request เดียวจะทำนายทันทีไม่รอ จึงไม่ทำให้ช้าลง
3. ไม่แก้: แปลงวันที่ล่วงหน้าในฝั่ง serving (เร็วขึ้นแค่ ~13% แต่ทำให้ input ต่างจากตอน train)
   และ `assume_finite` (วัดแล้วไม่ช่วย)

ผลโมเดลยืนยันด้วย `tests/test_serving_performance.py` (batch = ทีละแถว) และ
`scripts/check_serving_parity.py` (API จริง vs โมเดล champion ในเครื่อง)

### วิธีทดลองและเทียบผล (Windows cmd)

```bat
:: เก็บผลรอบก่อน optimize ไว้เทียบ (1 worker, ไม่มี batching)
ren loadtest\results\run_stats.csv baseline_stats.csv

:: แต่ละรอบ: ตั้งค่า → สร้าง api ใหม่ → รอ model_loaded → locust 2 นาที
set API_WORKERS=2
set API_BATCH=1
docker compose up -d --build api
curl localhost:8000/health
uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless --users 50 --spawn-rate 10 --run-time 2m --csv loadtest/results/w2
```

ทำซ้ำโดยเปลี่ยนค่าและชื่อไฟล์ (`w4` = 4 workers ไม่มี batching, `b32` = 1 worker + batching,
`w2b32` = 2 workers + batching) แล้วสร้างรายงานเทียบ:

```bat
uv run python scripts/check_slo.py loadtest/results/w2b32 --label "2 workers + batching 32" ^
  --compare "1 worker (baseline)=loadtest/results/baseline" ^
  --compare "2 workers=loadtest/results/w2" ^
  --compare "4 workers=loadtest/results/w4" ^
  --compare "1 worker + batching 32=loadtest/results/b32"
uv run python scripts/check_serving_parity.py
```

ส่วนวิเคราะห์และข้อเสนอ SLO ให้เขียนใต้บรรทัด `<!-- notes ... -->` ใน `docs/loadtest_report.md`
(สคริปต์จะเก็บส่วนนั้นไว้ทุกครั้งที่สร้างรายงานใหม่)

## Latency breakdown: HTTP / feature builder / ColumnTransformer / LightGBM

ผลการวัดและข้อสรุปอยู่ใน `docs/latency_breakdown.md`

ใช้หาว่าเวลา p50 ของ `/predict` หมดไปกับขั้นไหน ก่อนตัดสินใจ optimize (ยังไม่ได้แก้ `features/`)

- `SERVING_PROFILE=1` (ใน compose: `API_PROFILE=1`) ทำให้ API เรียก pipeline ทีละขั้นแทน `predict_proba` ครั้งเดียว
  แล้วบันทึกเวลาแต่ละขั้นลง histogram `noshow_stage_seconds` คะแนนเหมือนเดิมทุกบิต
  (`tests/test_serving_profiling.py` เทียบแบบ bit-for-bit) ค่าเริ่มต้นปิดไว้
- `scripts/profile_serving.py offline` รันใน container ไม่ผ่าน HTTP วัดทีละขั้นที่ 1 / 8 / 32 แถวต่อครั้ง
- `scripts/profile_serving.py http` ยิงทีละ request รันได้ทั้งใน container และบน host ส่วนต่างคือต้นทุนของ port forwarding
- `scripts/profile_serving.py load` รันบนเครื่อง host ยิง Locust แล้วเทียบเวลาที่ client เห็นกับเวลาใน app
  ส่วนต่างของค่าเฉลี่ย = HTTP + network + client

```bat
:: 1) เปิด profiling (2 workers + batching 32 เหมือนค่าที่ใช้จริง)
set API_WORKERS=2
set API_BATCH=32
set API_PROFILE=1
docker compose up -d --build api
curl localhost:8000/health

:: 2) ไม่มีโหลด: แยกเวลาทีละขั้นใน container
docker compose exec -T api python - offline < scripts\profile_serving.py > loadtest\results\profile_offline.md

:: 2b) HTTP ทีละ request: จากในตัว container (ไม่ผ่าน port forwarding) เทียบกับจาก Windows
docker compose exec -T api python - http --rounds 200 < scripts\profile_serving.py > loadtest\results\profile_http_inside.md
uv run python scripts/profile_serving.py http --rounds 200 > loadtest\results\profile_http_host.md

:: 3) โหลดเบา (1 user) และ 50 users
uv run python scripts/profile_serving.py load --users 1 --spawn-rate 1 --run-time 1m --name prof1
uv run python scripts/profile_serving.py load --users 50 --run-time 2m --name prof50

:: 3b) 50 users จาก client ในเครือข่าย Docker (ไม่ผ่าน Windows port forwarding) รอบแรก 30 วิคือ warm-up
docker run --rm --network clinic-noshow-prediction_default -v "%cd%":/mnt/locust locustio/locust -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 30s --only-summary
docker run --rm --network clinic-noshow-prediction_default -v "%cd%":/mnt/locust locustio/locust -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 2m --csv /mnt/locust/loadtest/results/net50b --only-summary

:: 4) ผลโมเดลต้องเหมือนเดิมทั้งตอนเปิดและปิด profiling
uv run python scripts/check_serving_parity.py
set API_PROFILE=0
docker compose up -d api
uv run python scripts/check_serving_parity.py
```

## Metrics สำหรับ Monitoring

| Metric | ใช้ทำอะไร |
|---|---|
| `noshow_request_latency_seconds` (histogram, label `path`) | p50/p95 ใน Grafana: `histogram_quantile(0.95, sum by (le) (rate(noshow_request_latency_seconds_bucket{path="/predict"}[5m])))` |
| `noshow_requests_total` (label `path`, `method`, `status`) | error rate / availability |
| `noshow_predictions_total` (label `alert`) | ปริมาณการแจ้งเตือน |
| `noshow_prediction_score` (histogram) | การกระจายของ score (prediction drift) |
| `noshow_model_info` (label `version`, `threshold`) | version ที่ serve อยู่ (=1 ถ้ามี worker ใดใช้ version นั้น) |
