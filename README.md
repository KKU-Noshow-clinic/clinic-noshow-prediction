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

ขั้นตอนนี้ติดตั้งระบบครั้งแรก จากนั้นรัน pipeline ด้วยคำสั่งเดียวเพื่อทำงานตั้งแต่ข้อมูลดิบจนถึงการโหลดโมเดลเข้า API
การเปิด container อย่างเดียวไม่ได้สร้างโมเดล: เครื่องใหม่จะยังมี `model_loaded: false` จนกว่า pipeline จะสำเร็จ

### 1. เตรียมเครื่องและดาวน์โหลดโค้ด

ติดตั้ง [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/)
และ [Docker Desktop](https://docs.docker.com/desktop/) (หรือ Docker Engine พร้อม Compose บน Linux)
บน Windows ให้เปิด Docker Desktop โดยใช้ Linux containers และตั้งค่า WSL2 ตามขั้นตอนติดตั้งของ Docker
เปิด terminal ใหม่หลังติดตั้ง และเปิด Docker ให้พร้อมใช้งานก่อนเริ่ม
ไม่จำเป็นต้องติดตั้ง Python แยก เพราะ uv ติดตั้ง Python 3.11 ให้ได้

ตรวจเครื่องมือ:

```text
git --version
uv --version
docker compose version
docker info
```

ใช้ terminal ปกติ ไม่จำเป็นต้องเป็น Administrator เลือกโฟลเดอร์ที่จะเก็บงาน แล้วรัน:

```text
git clone https://github.com/KKU-Noshow-clinic/clinic-noshow-prediction.git
cd clinic-noshow-prediction
```

ถ้ามี checkout อยู่แล้ว ให้เข้าโฟลเดอร์เดิม ไม่ต้อง clone ซ้ำ ทุกคำสั่งต่อไปนี้ให้รันจากโฟลเดอร์โครงงาน

### 2. ติดตั้ง dependencies เตรียมข้อมูล และตรวจโค้ด

คำสั่งต่อไปนี้ใช้ได้ทั้ง PowerShell และ Bash โดยไม่ต้องมี `make`:

```text
uv python install 3.11
uv sync --locked
uv run python scripts/download_data.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

- `uv sync --locked` ติดตั้ง dependencies รวมเครื่องมือทดสอบตาม `uv.lock` โดยไม่เปลี่ยน lockfile
- ดาวน์โหลดข้อมูลเป็น `data/raw/noshow.csv` และตรวจ SHA-256 ให้ตรงกับ `configs/params.yaml`
  ถ้ามีไฟล์อยู่แล้วจะตรวจไฟล์เดิม หาก SHA-256 ไม่ตรง ให้ตรวจที่มาของข้อมูลก่อนทำต่อ
- ถ้า Kaggle ขอ authentication ให้ตั้งค่าบัญชี/API credentials ตาม
  [KaggleHub](https://github.com/Kaggle/kagglehub#authentication) ห้าม commit token หรือ credentials
- ถ้าการติดตั้ง ดาวน์โหลด หรือทดสอบล้มเหลว ให้แก้ error ก่อนทำขั้นต่อไป

### 3. เปิด API และบริการประกอบ

```text
docker compose up -d --build
docker compose ps
```

การ build ครั้งแรกอาจใช้เวลาหลายนาที รอให้บริการพร้อม แล้วตรวจ API บน Windows PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

บน Linux/macOS ใช้ `curl http://localhost:8000/health`
ถ้าเป็นเครื่องใหม่ การได้ `status: ok` แต่ `model_loaded: false` พร้อมข้อความว่าไม่พบ `clinic-noshow`
เป็นสถานะที่คาดไว้: API เปิดแล้ว แต่ MLflow ยังไม่มี Champion ให้โหลด
ถ้าเชื่อมต่อไม่ได้หรือ container หยุด ให้ดู error ด้วย:

```text
docker compose logs --tail=100 api mlflow prefect
```

### 4. รัน pipeline ด้วยคำสั่งเดียว

ต้องมีไฟล์ข้อมูลจากขั้นที่ 2 และเปิดบริการจากขั้นที่ 3 ก่อน
รัน pipeline บนเครื่อง host ใน terminal เดิม ไม่ใช่ภายใน container
เพราะ flow ปัจจุบันเชื่อม MLflow ผ่าน `http://localhost:5001`

Windows PowerShell:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:NOSHOW_API_URL = "http://127.0.0.1:8000"
uv run python -m noshow.pipeline.flow
```

Linux/macOS Bash:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export NOSHOW_API_URL=http://127.0.0.1:8000
uv run python -m noshow.pipeline.flow
```

หลังตั้ง environment แล้ว คำสั่ง `uv run python -m noshow.pipeline.flow` จะทำทั้งกระบวนการ:

```text
ingest → validate → clean/split → preprocess → train → evaluate
→ register challenger → gate → promote → deploy (เรียก POST /reload)
```

- รันทั้ง 3 การทดลอง และเลือก Exp 2 ที่ threshold 0.5 ตาม config
- เครื่องใหม่: ถ้าโมเดลผ่าน Gate จะตั้ง Champion และสั่ง API โหลดโมเดลนั้น
- เครื่องที่มี Champion แล้ว: โมเดลใหม่ต้องผ่านเกณฑ์และมี Recall สูงกว่า Champion
  หากผลเท่าเดิมหรือไม่ผ่าน จะได้ `gate_passed: false`, `promotion_status: not_promoted`
  และ `deployment: None` โดย API ใช้ Champion เดิมต่อ นี่ไม่ใช่ deploy สำเร็จของโมเดลใหม่
- ข้อมูลไม่ผ่าน validation: flow หยุดก่อน train พร้อม failure log/Prefect Artifact
  การแจ้งเตือน Discord ต้องตั้ง webhook เพิ่มตาม [Pipeline handoff](docs/pipeline_handoff.md)

ดูขั้นตอนและสถานะการรันที่ [Prefect](http://localhost:4200) และโมเดลที่
[MLflow](http://localhost:5001) คำสั่งเดียวข้างต้นใช้ได้โดยไม่ต้องสร้างปุ่มในหน้าเว็บ
หากต้องการกดรันผ่าน Prefect UI ต้องตั้ง Deployment เพิ่ม

### 5. ตรวจว่า API พร้อมทำนาย

Windows PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

ต้องเห็น `model_loaded: true`, `model_version` มีค่า, `threshold: 0.5` และ `last_error: null`
API มีหลาย workers จึงอาจต้องรอสักครู่ให้ workers อื่นตรวจพบ Champion ใหม่
ถ้า pipeline สำเร็จแต่ยังโหลดไม่ได้ ให้ตรวจ log ของ API และ MLflow ก่อนทำ load test

เปิด [API docs](http://localhost:8000/docs) แล้วใช้ `POST /predict` → **Try it out**
ส่งข้อมูลตามตัวอย่าง schema (ไม่ส่ง `No-show` ซึ่งเป็นคำตอบที่ต้องการทำนาย)
ควรได้ HTTP 200 พร้อม `noshow_score`, `alert`, `threshold` และ `model_version`

ตรวจผล API เทียบกับ Champion ที่เรียกในเครื่อง:

Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
uv run python scripts/check_serving_parity.py
```

Linux/macOS Bash:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
uv run python scripts/check_serving_parity.py
```

### 6. วัด latency และ throughput

รันเมื่อ API โหลดโมเดลแล้วเท่านั้น หากต้องการทดสอบตามรอบ Docker ที่ทีม Serving รายงาน
ให้ใช้ Locust ใน **container อีกตัว** เรียก API ผ่าน HTTP บนเครือข่าย Docker:

```text
Locust container → HTTP ผ่านเครือข่าย Docker → API container → โมเดล → ส่งคำตอบกลับ
```

นี่คือการทดสอบ API จริงจากภายนอก container ของ API ไม่ใช่การจับเวลาเฉพาะฟังก์ชันทำนาย
เงื่อนไข: 2 workers, micro-batching สูงสุด 32 แถว, warm-up API 30 วินาที
แล้วเริ่มรอบวัดใหม่ 2 นาที โดยเพิ่มผู้ใช้ 10 คน/วินาทีจนถึง 50 คน
ช่วง warm-up ไม่รวมในไฟล์ผลรอบวัด แต่ช่วงเพิ่มผู้ใช้ของรอบวัดยังรวมอยู่
traffic mix คือ `/predict` 10 : `/predict_batch` (20 แถว) 1 : `/health` 1
ใช้ Locust image เวอร์ชันตรงกับ `uv.lock` และเปิด profiling เพื่อให้ตรงกับรอบที่ทีมบันทึกใน
[Latency breakdown](docs/latency_breakdown.md)

**Windows PowerShell:** รันจากโฟลเดอร์โครงงานตามลำดับ

```powershell
$env:API_WORKERS = "2"
$env:API_BATCH = "32"
$env:API_PROFILE = "1"
docker compose up -d --build api
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

รอให้ `model_loaded: true` และไม่มี load error ก่อนทำต่อ จากนั้นหาเครือข่ายที่ API ใช้
อัตโนมัติ เพื่อรองรับชื่อโฟลเดอร์/Compose project ที่ต่างกัน:

```powershell
New-Item -ItemType Directory -Force -Path loadtest/results | Out-Null
$projectPath = (Get-Location).Path
$apiContainer = docker compose ps -q api
$apiNetwork = (docker inspect $apiContainer | ConvertFrom-Json)[0].NetworkSettings.Networks.PSObject.Properties.Name | Select-Object -First 1
docker run --rm --network "$apiNetwork" --mount "type=bind,source=$projectPath,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 30s --only-summary
docker run --rm --network "$apiNetwork" --mount "type=bind,source=$projectPath,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 2m --csv /mnt/locust/loadtest/results/docker_run --only-summary
uv run python scripts/check_slo.py loadtest/results/docker_run --label "Docker network; 2 workers; batch 32; profile on; API warm-up 30s" --users 50 --duration 2m --out docs/loadtest_report_docker.md
```

**Linux/macOS Bash:** ใช้เงื่อนไขเดียวกัน

```bash
export API_WORKERS=2 API_BATCH=32 API_PROFILE=1
docker compose up -d --build api
curl http://localhost:8000/health
# รอให้ model_loaded เป็น true ก่อนรันคำสั่งด้านล่าง
mkdir -p loadtest/results
api_container=$(docker compose ps -q api)
api_network=$(docker inspect --format '{{range $name, $config := .NetworkSettings.Networks}}{{$name}}{{end}}' "$api_container")
docker run --rm --network "$api_network" --mount "type=bind,source=$PWD,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 30s --only-summary
docker run --rm --network "$api_network" --mount "type=bind,source=$PWD,target=/mnt/locust" locustio/locust:2.46.6 -f /mnt/locust/loadtest/locustfile.py --host http://api:8000 --headless --users 50 --spawn-rate 10 --run-time 2m --csv /mnt/locust/loadtest/results/docker_run --only-summary
uv run python scripts/check_slo.py loadtest/results/docker_run --label "Docker network; 2 workers; batch 32; profile on; API warm-up 30s" --users 50 --duration 2m --out docs/loadtest_report_docker.md
```

ผลรอบ Docker อยู่ใน `loadtest/results/docker_run_*.csv` และ `docs/loadtest_report_docker.md`
ไม่เขียนทับรายงานรอบ host เดิม เป้าหมาย `/predict`: p50 ≤ 50 ms, p95 ≤ 200 ms
และ error rate รวม ≤ 1% ตาม `configs/slo.yaml` พร้อมรายงาน throughput เป็น requests/second
ค่า availability ในรายงานเป็นสัดส่วนคำขอที่สำเร็จในรอบทดสอบ ไม่ใช่ uptime ระยะยาว
ผลเดิมที่ทีมรายงาน p50 45 ms / p95 80 ms เป็นหลักฐานของเครื่องและรอบนั้นเท่านั้น
ผู้รันต้องบันทึกผลใหม่ พร้อมสเปกเครื่อง เวอร์ชันโมเดล และการตั้งค่า ไม่รับรองว่าจะผ่านทุกเครื่อง
API และ Locust ใช้ทรัพยากร Docker host ร่วมกัน ซึ่งอาจมีผลต่อ latency

เมื่อทดสอบเสร็จ ปิด profiling กลับเป็นค่าปกติ (API จะเริ่มใหม่และต้องรอโหลดโมเดล):

```powershell
# Windows PowerShell
$env:API_PROFILE = "0"
docker compose up -d api
```

บน Bash ใช้ `export API_PROFILE=0` แล้ว `docker compose up -d api`

### 7. ปิดระบบเมื่อเลิกใช้งาน

```text
docker compose down
```

คำสั่งนี้เก็บ named volumes ของ MLflow/Prefect/Grafana ไว้สำหรับครั้งถัดไป
หากต้องการเก็บโมเดลและประวัติการทดลอง อย่าเพิ่ม `--volumes` หรือ `-v`

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
