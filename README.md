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

ต้องมี [uv](https://docs.astral.sh/uv/) และ Docker เท่านั้น
`make data` ดาวน์โหลดจาก Kaggle ได้โดยไม่ต้อง login (ถ้าเจอ rate limit ให้ตั้ง env `KAGGLE_USERNAME` / `KAGGLE_KEY`)

```bash
git clone https://github.com/KKU-Noshow-clinic/clinic-noshow-prediction.git && cd clinic-noshow-prediction
make setup      # ติดตั้ง Python 3.11 + dependencies ตาม uv.lock (ล็อกเวอร์ชันทุกตัว)
make data       # ดาวน์โหลดข้อมูล -> data/raw/noshow.csv และพิมพ์ SHA-256 (= data version)
make test       # ruff + pytest
make up         # เปิด API, MLflow, Prefect, Prometheus, Grafana (ครั้งแรก build image ~5 นาที)
make pipeline   # train 3 experiments → gate → promote champion → API โหลดโมเดล (~5 นาที)
make loadtest   # Locust p50/p95 เทียบ SLO -> docs/loadtest_report.md (ต้องเปิด API ก่อน)
curl localhost:8000/health   # ต้องเห็น "model_loaded": true
```

ถ้า `/health` ขึ้น `model_loaded: false` แปลว่ายังไม่ได้รัน `make pipeline` (ยังไม่มี champion ใน registry)

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
