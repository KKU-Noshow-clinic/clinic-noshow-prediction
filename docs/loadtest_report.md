# Load test report (Serving)

- Date: 2026-10-03 23:15
- Configuration: 2 workers + batching 32, client inside Docker network
- Tool: Locust, 50 concurrent users, duration 2m
- Traffic mix: /predict 10 : /predict_batch (20 rows) 1 : /health 1
- Raw results: `loadtest/results/net50b_stats.csv`

## SLO check: 2 workers + batching 32, client inside Docker network (targets from `configs/slo.yaml`)

| SLO | Measured | Target | Result |
|---|---:|---:|:---:|
| p50 latency /predict | 45 ms | <= 50 ms | PASS |
| p95 latency /predict | 80 ms | <= 200 ms | PASS |
| error rate (all requests) | 0.00% | <= 1.00% | PASS |
| availability | 100.00% | >= 99.00% | PASS |

**Overall: all SLOs met**

## Before / after optimization

| Configuration | /predict p50 | /predict p95 | /predict p99 | Total req/s | Errors | p50 <= 50 | p95 <= 200 |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| 1 worker baseline (Windows client) | 1100 ms | 1500 ms | 1800 ms | 35.7 | 0.00% | FAIL | FAIL |
| 2 workers (Windows client) | 160 ms | 300 ms | 420 ms | 107.0 | 0.02% | FAIL | FAIL |
| 4 workers (Windows client) | 75 ms | 170 ms | 280 ms | 130.0 | 0.00% | FAIL | PASS |
| 1 worker + batching 32 (Windows client) | 68 ms | 110 ms | 170 ms | 133.6 | 0.00% | FAIL | PASS |
| 2 workers + batching 32 (Windows client) | 74 ms | 99 ms | 120 ms | 133.1 | 0.00% | FAIL | PASS |
| 4 workers + batching 32 (Docker network) | 46 ms | 110 ms | 170 ms | 140.2 | 0.00% | PASS | PASS |
| 2 workers + batching 32, client inside Docker network | 45 ms | 80 ms | 120 ms | 142.5 | 0.00% | PASS | PASS |

## Per-endpoint latency: 2 workers + batching 32, client inside Docker network

| Endpoint | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| GET /health | 1421 | 11.9 | 11 | 24 | 35 | 0.00% |
| POST /predict | 14224 | 119.1 | 45 | 80 | 120 | 0.00% |
| POST /predict_batch | 1369 | 11.5 | 47 | 78 | 110 | 0.00% |
| Aggregated | 17014 | 142.5 | 44 | 78 | 110 | 0.00% |

<!-- notes: everything below is kept when this report is regenerated -->

## Analysis

ตัวเลขทั้งหมดของ `/predict` ที่ 50 users, Locust 2 นาที รายละเอียดการแยกเวลาทีละขั้นอยู่ใน `docs/latency_breakdown.md`

### ก่อน optimize
- 1 process รับได้ ~36 req/s ที่เหลือต่อคิว ทำให้ p50/p95 เป็น 1.1/1.5 วินาที
- profile แล้วพบว่า preprocessing (feature builder + ColumnTransformer) ใช้ ~87% ของเวลาคำนวณ
  และ 32 แถวใช้เวลาแทบเท่า 1 แถว ส่วน LightGBM ใช้แค่ ~7%

### สิ่งที่ทำ (ไม่แก้ `features/`, ผลโมเดลเหมือนเดิม)
- uvicorn 2 workers + micro-batching 32 (ค่าเริ่มต้นใน Docker) → client บน Windows: p50 74 / p95 99 / p99 120 ms, 0% error
- 4 workers หรือ 1 worker + batching ไม่ได้ดีกว่า (ตารางด้านบน)

### ทำไม p50 จาก Windows ยังเกิน 50 ms
- ยิงทีละ request: API ใช้ ~13 ms (วัดจาก container ใน/นอก API) แต่จาก Windows ได้ ~59 ms
  ส่วนต่าง ~46 ms อยู่ที่ Docker Desktop port forwarding และเกิดเฉพาะ POST (สมมติฐาน: TCP Nagle + delayed ACK)
- **วัดจาก client ในเครือข่าย Docker หลัง warm-up 30 วิ: p50 45 / p95 80 / p99 120 ms, 0% error, 142.5 req/s ผ่านทุก SLO**
  (Locust ใช้ CPU เครื่องเดียวกับ API และเปิด SERVING_PROFILE อยู่)

### ผลโมเดลไม่เปลี่ยน
- pytest ผ่านทั้งหมด รวมเทสต์ batching และ profiling (คะแนนเท่ากันทุกบิต)
- `check_serving_parity.py`: PARITY OK ทั้งก่อนและหลังเปิด profiling (320 แถวจริง ต่างสูงสุด 5e-7 จากการปัดเศษ alert ตรงทุกแถว)

### ข้อเสนอ SLO (ยังไม่ได้แก้ `configs/slo.yaml`)
- คง p50 ≤ 50 ms และ p95 ≤ 200 ms แต่ระบุจุดวัด: client ในเครือข่ายเดียวกับ API (หรือ metric ฝั่ง server),
  50 users, warm-up 30 วิ, 2 workers + batching 32; เพิ่ม p99 ≤ 250 ms
- margin p50 เหลือ ~5 ms ถ้าต้องการเพิ่ม ทางถัดไปคือลดเวลา feature builder (~10 ms ต่อ pipeline call ตอนมีโหลด)
  ร่วมกับทีม Model คาดว่าได้ 5–10 ms
- ควรวัดซ้ำบนเครื่อง Linux ที่ใช้ deploy จริงก่อนสรุปเป็นตัวเลขทางการ
