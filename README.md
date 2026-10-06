# Clinic Appointment No-Show Prediction
ระบบทำนายผู้ป่วยที่ไม่มาตามนัดของคลินิก เพื่อให้คลินิกส่ง SMS/โทรเตือน หรือจัด overbooking ได้ทันเวลา
(ชุดข้อมูล [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments), 110,527 นัดหมาย)

โครงงานรายวิชา **CP413008 Machine Learning Engineering for Production** ภาคเรียนที่ 1/2569 มหาวิทยาลัยขอนแก่น

## สมาชิกและความรับผิดชอบ

| # | รหัสนักศึกษา | ชื่อ-นามสกุล | Sec | Workstream |
|---|---|---|---|---|
| 1 | 673380532-1 | นายศรัณยู เจริญผล | 1 | Lead / Framing / Integration |
| 2 | 673380526-6 | นายพีรพงษ์ ทองฤทธิ์ | 2 | Data & Validation |
| 3 | 673380078-7 | นายธันว์ สว่างศรี | 2 | Features & Modeling |
| 4 | 673380306-0 | นายกิตตินันท์ ไขไพรวัน | 1 | Tracking, Registry & Pipeline |
| 5 | 673380528-2 | นายภูรินทร์ ศรีฐาน | 1 | Serving & Infra |
| 6 | 673380075-3 | นายฑีฌานนท์ อัศวะภูมิ | 2 | Monitoring |
| 7 | 673380313-3 | นายณันทพงศ์ พยัคมะเริง | 2 | CI/CD & Testing |

## Quickstart (จากเครื่องเปล่า)

ต้องมี [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/) และ Docker Desktop (เปิดไว้ก่อนเริ่ม)

### macOS / Linux

คำสั่งเดียว ตั้งแต่ติดตั้งจนได้ API ที่โหลดโมเดลแล้ว:

```bash
git clone https://github.com/KKU-Noshow-clinic/clinic-noshow-prediction.git
cd clinic-noshow-prediction
make all          # setup → data → test → up → รอ MLflow/API → pipeline (~10 นาทีครั้งแรก)
```

หรือรันทีละขั้น:

```bash
make setup        # ติดตั้ง Python 3.11 + dependencies ตาม uv.lock
make data         # ดาวน์โหลดข้อมูลและตรวจ SHA-256
make test         # ruff + pytest
make up           # เปิด API, MLflow, Prefect, Prometheus, Grafana (ครั้งแรก ~5 นาที)
make wait         # รอจน MLflow และ API พร้อม
make pipeline     # train → gate → promote champion (~5 นาที)
curl localhost:8000/health    # ต้องเห็น "model_loaded": true
```

ปิดระบบด้วย `docker compose stop` (ข้อมูลใน MLflow/Prefect ยังอยู่)

### Windows (PowerShell)

Windows ไม่มี `make` ให้ใช้คำสั่งด้านล่างแทน

<details>
<summary>กดเพื่อดูคำสั่ง</summary>

**1. ดาวน์โหลดโค้ดและเตรียมข้อมูล**

```powershell
git clone https://github.com/KKU-Noshow-clinic/clinic-noshow-prediction.git
cd clinic-noshow-prediction
uv python install 3.11
uv sync --locked
uv run python scripts/download_data.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

**2. เปิดระบบและรัน pipeline**

```powershell
$env:API_WORKERS = "2"
$env:API_BATCH = "32"
$env:API_PROFILE = "1"
docker compose up -d --build
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:NOSHOW_API_URL = "http://127.0.0.1:8000"
uv run python -m noshow.pipeline.flow
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

รอให้ `model_loaded: true` ก่อนทดสอบ ปิดระบบด้วย `docker compose down`

</details>

### Load test (ไม่บังคับ)

วัด p50/p95 เทียบ SLO: warm-up 30 วินาที แล้ววัด 2 นาที ผู้ใช้จำลอง 50 คน (ต้องเปิดระบบก่อน)

<details>
<summary>macOS / Linux</summary>

```bash
make loadtest
```

</details>

<details>
<summary>Windows (PowerShell, รันใน Docker network)</summary>

```powershell
New-Item -ItemType Directory -Force -Path loadtest/results | Out-Null
$projectPath = (Get-Location).Path
$apiContainer = docker compose ps -q api
$apiNetwork = (docker inspect $apiContainer | ConvertFrom-Json)[0].NetworkSettings.Networks.PSObject.Properties.Name | Select-Object -First 1
docker run --rm --network "$apiNetwork" --mount "type=bind,source=$projectPath,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 30s --only-summary
docker run --rm --network "$apiNetwork" --mount "type=bind,source=$projectPath,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 2m --csv /mnt/locust/loadtest/results/docker_run --only-summary
uv run python scripts/check_slo.py loadtest/results/docker_run --label "Docker; 2 workers; batch 32; profile on; warm-up 30s" --users 50 --duration 2m --out docs/loadtest_report_docker.md
```

ดูผลที่ `docs/loadtest_report_docker.md`

</details>

เป้า SLO: p50 ≤ 50 ms, p95 ≤ 200 ms, error ≤ 1%

> **หมายเหตุ:** ถ้า `localhost:8000` ตอบ `Not Found` แปลว่ามีโปรแกรมอื่นใช้ port 8000 อยู่ ให้ปิดโปรแกรมนั้นก่อน
> ถ้า Kaggle ขอ login ให้ตั้ง [credentials](https://github.com/Kaggle/kagglehub#authentication) แล้วรัน `make data` ใหม่

| Service | URL |
|---|---|
| API (FastAPI docs) | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Prefect | http://localhost:4200 |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

## ข้อตกลงหลักของทีม

- **Target:** `No-show == "Yes"` → 1 (ผู้ป่วยไม่มา) ประมาณ 20% ของข้อมูล
- **Split:** ตามเวลา `AppointmentDay` (ดู `configs/params.yaml`) ห้ามสุ่ม
- **Optimizing metric สำหรับงานส่งเตือน:** no-show recall ที่ threshold 0.5; ทีมเลือก Exp 2
  โดยใช้ PR-AUC เป็นเกณฑ์ขั้นต่ำและรายงาน precision/จำนวนคนที่ต้องเตือนด้วย (ดู `configs/slo.yaml`)
- **Preprocessing:** โค้ดชุดเดียวใน `src/noshow/features/` ใช้ทั้งตอน train และ serve
- **Registry model name:** `clinic-noshow`

## Git workflow

- ห้าม push ตรงเข้า `main` ทุกงานทำผ่าน branch แล้วเปิด Pull Request
- ตั้งชื่อ branch: `feat/<ชื่อ>-<เรื่อง>` เช่น `feat/kittinan-pandera-schema`, แก้บั๊กใช้ `fix/...`
- PR ต้องผ่าน CI และมีคน approve อย่างน้อย 1 คนก่อน merge
- ก่อนเปิด PR ให้รัน `make format && make test`
- คะแนนรายบุคคลดูจาก commit history — ทุกคน commit งานของตัวเองด้วย account ตัวเอง

## โครงสร้าง

```
src/noshow/
  data/        ingest, time-based split, Pandera validation
  features/    preprocessing ชุดเดียว ใช้ทั้ง train และ serve (กัน training-serving skew)
  models/      train, evaluate, gate
  registry/    MLflow registry: promote, rollback
  serving/     FastAPI app (/predict, /predict_batch, /health, /reload, /metrics) - docs/serving.md
  monitoring/  Evidently drift + retrain trigger
  pipeline/    Prefect flows (DAG)
configs/       params.yaml, slo.yaml
docker/        Dockerfiles
monitoring/    Prometheus / Grafana config
scripts/       download_data.py
tests/         unit + data tests
docs/          canvas, architecture, report
```

## การใช้เครื่องมือ AI

ระบุส่วนที่ใช้ AI ช่วยเขียนโค้ดไว้ในรายงาน (ตามข้อกำหนดรายวิชา) และทุกคนต้องอธิบายโค้ดที่ตนเอง commit ได้
