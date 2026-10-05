# Latency breakdown (Serving, PR #15)

คำถาม: p50 ของ `/predict` หมดเวลาไปกับขั้นไหน (HTTP / feature builder / ColumnTransformer / LightGBM)
และจะให้ต่ำกว่า SLO 50 ms ได้อย่างไร ระหว่างการวัด**ไม่ได้แก้** `src/noshow/features/build.py`

## เครื่องและการตั้งค่าที่ใช้วัด

- Host: Windows 10 (build 26200), AMD Family 25 Model 117, 16 logical CPUs
- Docker Desktop (WSL2 kernel 5.15.167.4), 16 CPUs, ~7.9 GB RAM; container Python 3.11.16
- API: 2 uvicorn workers + micro-batching 32 (ค่าเริ่มต้น), model `clinic-noshow` v1 (Exp 2 LightGBM),
  `OMP_NUM_THREADS=1`; ตอนวัดแยกขั้นเปิด `SERVING_PROFILE=1` (คะแนนเหมือนเดิมทุกบิต)
- Load: Locust, traffic mix `/predict` 10 : `/predict_batch` (20 แถว) 1 : `/health` 1
- เครื่องมือ: `scripts/profile_serving.py` (offline / http / load), `scripts/check_slo.py`,
  `scripts/check_serving_parity.py` (วิธีรันอยู่ใน `docs/serving.md`)

## 1. ไม่มีโหลด: แยกเวลาทีละขั้นใน container (ไม่ผ่าน HTTP)

`profile_serving.py offline`, 1 process, warm-up 30 ครั้ง + วัด 300 ครั้งต่อขนาด batch (ไฟล์: `loadtest/results/profile_offline.md`)

| แถวต่อครั้ง | สร้าง DataFrame | feature builder | ColumnTransformer | LightGBM | รวม (p50) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9 ms | 7.0 ms (43%) | 7.4 ms (44%) | 1.2 ms (7%) | 17.0 ms |
| 8 | 1.0 ms | 7.2 ms | 7.6 ms | 1.2 ms | 17.7 ms |
| 32 | 1.1 ms | 7.6 ms | 7.8 ms | 1.3 ms | 18.7 ms |

- preprocessing (feature builder + ColumnTransformer) = ~87% ของเวลาคำนวณ, LightGBM แค่ ~7%
- 32 แถวใช้เวลาแทบเท่า 1 แถว จึงคุ้มที่จะรวม request เป็น batch
- ทุกครั้งคะแนนจากการเรียกทีละขั้นเท่ากับ `predict_proba` ทุกบิต

## 2. ไม่มีโหลด: HTTP ทีละ request (warm-up 20 + 200 รอบ, keep-alive)

| ยิงจาก | `GET /health` p50 | `POST /predict` p50 | `POST /predict` p95 |
|---|---:|---:|---:|
| ใน container ของ API เอง (loopback) | 1.1 ms | 12.9 ms | 15.3 ms |
| container อื่นบนเครือข่าย Docker | 1.1 ms | 13.1 ms | 23.4 ms |
| Windows ผ่าน Docker Desktop port forwarding | 1.5 ms | 58.7 ms | 68.4 ms |

- API ทั้ง request (รับ JSON, validate, ทำนาย, ตอบ) ใช้ ~13 ms; เครือข่าย Docker เพิ่ม ~0.2 ms
- ทางผ่าน Windows → Docker Desktop เพิ่ม ~46 ms **เฉพาะ POST** (GET ผ่านทางเดียวกันแต่เพิ่มแค่ 0.4 ms)
  ลักษณะตรงกับปัญหา TCP Nagle + delayed ACK (~40 ms) ใน port forwarding — เป็นสมมติฐาน ยังไม่ได้พิสูจน์ตรงๆ
- ค่า ~61 ms ที่วัดได้ก่อนหน้า (1 user จาก Windows) จึงไม่ใช่เวลาของตัว service

## 3. มีโหลด 50 users: แยกเวลาใน app (client บน Windows)

`profile_serving.py load --users 50`, 2 นาที, 13,126 `/predict`, error 0; 7,716 pipeline calls เฉลี่ย 5.2 แถว/ครั้ง

| ส่วน | p50 | mean |
|---|---:|---:|
| client เห็น `/predict` | 66 ms | 70.2 ms |
| ใน app ทั้ง request | 54.2 ms | 59.6 ms |
| ↳ รอคิว micro-batch (ต่อ request) | 10.2 ms | 10.7 ms |
| ↳ pipeline 1 ครั้ง: สร้าง DataFrame | 1.8 ms | 1.9 ms |
| ↳ pipeline 1 ครั้ง: feature builder | 10.2 ms | 10.7 ms |
| ↳ pipeline 1 ครั้ง: ColumnTransformer | 9.7 ms | 10.3 ms |
| ↳ pipeline 1 ครั้ง: LightGBM | 1.9 ms | 2.6 ms |
| HTTP + network + client (mean client − mean in-app) | | ~10.6 ms |

- ตอนมีโหลด ทางผ่าน Windows เพิ่มแค่ ~11 ms (ต่างจาก ~46 ms ตอนยิงทีละ request)
- ทุกขั้นของ preprocessing ช้าลง ~40% จากตอนไม่มีโหลด (แย่ง CPU)
- ใน app 54 ms = รอคิว ~10 + pipeline ~24 + **ส่วนที่ไม่ได้วัดตรงๆ ~20 ms** (validation/serialization ของ request พร้อมกันหลายตัว
  และการรอ CPU) เพิ่ม worker เป็น 4 แล้วไม่ลด (ตารางข้อ 4) จึงน่าจะเป็นการแย่ง CPU ทั้งเครื่องมากกว่า GIL ใน process — สมมติฐาน

## 4. ผล SLO ที่ 50 users (`/predict`)

| Client | Config | p50 | p95 | p99 | req/s | errors |
|---|---|---:|---:|---:|---:|---:|
| Windows | 1 worker (baseline ก่อน optimize) | 1,100 | 1,500 | 1,800 | 35.7 | 0% |
| Windows | 2 workers + batching 32 | 74 | 99 | 120 | 133.1 | 0% |
| เครือข่าย Docker, ไม่ warm-up | 2 workers + batching 32 | 39 | 210 | 550 | 132.5 | 0% |
| **เครือข่าย Docker, หลัง warm-up 30 วิ** | **2 workers + batching 32** | **45** | **80** | **120** | **142.5** | **0%** |
| เครือข่าย Docker, หลัง warm-up 30 วิ | 4 workers + batching 32 | 46 | 110 | 170 | 140.2 | 0% |

(ms) รอบเครือข่าย Docker: Locust รันใน container (`locustio/locust`) บนเครือข่าย `clinic-noshow-prediction_default`
ใช้ CPU เครื่องเดียวกับ API และเปิด `SERVING_PROFILE=1` อยู่ ตัวเลขจึงไม่ได้ดีเกินจริง

- รอบไม่ warm-up มี request ช้าถึง 2.6 วินาทีช่วงเริ่มต้น จึงใช้รอบหลัง warm-up เป็นผลหลัก
- 4 workers ไม่ช่วย p50 และทำให้ p95/p99 แย่ลง → คงค่าเริ่มต้นที่ 2 workers + batching 32

## 5. ผลโมเดลไม่เปลี่ยน

- `check_serving_parity.py` ตอนเปิด `SERVING_PROFILE=1`: PARITY OK (300 แถวผ่าน `/predict_batch` + 20 แถวผ่าน `/predict`,
  ต่างสูงสุด 5.0e-7 จากการปัดเศษ 6 ตำแหน่ง, alert ตรงทุกแถว)
- `tests/test_serving_profiling.py`: การเรียกทีละขั้นให้ผล `predict_proba` เท่ากันทุกบิต และคะแนนจาก API เหมือนกันทั้งเปิด/ปิด profiling

## สรุปและข้อเสนอ

1. **เป้า p50 ≤ 50 ms ผ่านแล้ว (45 ms) โดยไม่แก้ `features/build.py`** เมื่อวัดจาก client ในเครือข่ายเดียวกับ service
   ส่วน ~46 ms ที่เห็นจาก Windows เป็นต้นทุนของ Docker Desktop บนเครื่อง dev ไม่ใช่ของ service
2. **ข้อเสนอ SLO (ยังไม่ได้แก้ `configs/slo.yaml`, รอทีมตัดสิน):** คง p50 ≤ 50 ms และ p95 ≤ 200 ms แต่เขียนเงื่อนไขการวัดให้ชัด
   คือ client อยู่ในเครือข่ายเดียวกับ API (หรือใช้ metric ฝั่ง server `noshow_request_latency_seconds`), 50 users,
   warm-up 30 วินาที, 2 workers + batching 32; เพิ่ม p99 ≤ 250 ms (วัดได้ 120 ms)
3. **margin ของ p50 เหลือ ~5 ms** ถ้าต้องการเพิ่ม: feature builder ใช้ ~10 ms ต่อ pipeline call ตอนมีโหลด (ราว 43% ของ pipeline)
   ทำให้เร็วขึ้นได้ แต่คาดว่าจะลด p50 ได้ราว 5–10 ms เท่านั้น ต้องคุยกับทีม Model ก่อน และต้องมีเทสต์ยืนยันว่า output เท่าเดิม
4. ข้อจำกัด: วัดบนเครื่อง dev เครื่องเดียว และ load generator ใช้ CPU ร่วมกับ API สาเหตุเรื่อง TCP และส่วน ~20 ms ใน app
   ยังเป็นสมมติฐาน ควรวัดซ้ำบนเครื่อง Linux ที่ใช้ deploy จริงก่อนสรุปเป็นตัวเลขทางการ
