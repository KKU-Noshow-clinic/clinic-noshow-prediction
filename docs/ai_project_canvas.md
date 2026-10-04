# AI Project Canvas — Clinic Appointment No-Show Prediction

ตัวเลขทั้งหมดอ้างอิงจาก `docs/eda.txt`, `docs/data_decision.txt`, `docs/modeling.md`,
`docs/model_results.json` และ `docs/latency_breakdown.md`

## 1. Problem / Value Proposition

- ผู้ป่วยประมาณ **20.2%** ไม่มาตามนัด (22,319 จาก 110,527 นัด) ทำให้คลินิกเสียสล็อตแพทย์และเวลาพยาบาล
  ขณะที่ผู้ป่วยคนอื่นต้องรอคิวนานขึ้น
- **คุณค่า:** ทำนายล่วงหน้าว่านัดไหนเสี่ยงจะไม่มา เพื่อให้คลินิกโทรหรือส่ง SMS เตือนเฉพาะกลุ่มเสี่ยง
  หรือจัด overbooking ให้เหมาะสม แทนการเตือนทุกคนแบบเหวี่ยงแห

## 2. Users / Stakeholders

- **ผู้ใช้หลัก:** เจ้าหน้าที่เวชระเบียนหรือฝ่ายนัดหมาย ซึ่งเห็นรายชื่อนัดเสี่ยงสูงของวันพรุ่งนี้
- **ผู้ได้รับผลกระทบ:** ผู้ป่วย (ได้รับการเตือน) และผู้บริหารคลินิก (ใช้ทรัพยากรได้คุ้มขึ้น)

## 3. Prediction Task

- Binary classification: `No-show = Yes` (1) หรือมาตามนัด (0) ต่อนัดหมาย 1 รายการ
- **ทำนายเมื่อ:** ตอนจองนัด (real-time) และคืนก่อนวันนัด (batch)
- **Output:** ความน่าจะเป็นที่จะไม่มา และ flag ว่าเสี่ยงหรือไม่ โดยใช้ threshold 0.5

## 4. Decisions

- ถ้า flag = เสี่ยง ให้ส่ง SMS ซ้ำหรือโทรยืนยันในคืนก่อนวันนัด
- ถ้าวันนั้นมีนัดเสี่ยงหลายรายการ ให้พิจารณา overbook เพิ่ม 1–2 สล็อต
- ต้นทุนของการทำนายผิดไม่เท่ากัน:
  - **False Negative** (ไม่เตือนคนที่ไม่มา): เสียสล็อตแพทย์ ซึ่งแพงกว่า
  - **False Positive** (เตือนคนที่มาอยู่แล้ว): เสียค่า SMS หรือเวลาโทร ซึ่งถูกกว่า
- ด้วยเหตุนี้ทีมจึง**เน้น recall มากกว่า precision**

## 5. Data Sources

- Kaggle *Medical Appointment No Shows* (บราซิล, 29 เม.ย. – 8 มิ.ย. 2016) 110,527 แถว 14 คอลัมน์
- Data version v1: SHA-256 `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`
- หลัง cleaning เหลือ **110,521 แถว** (ลบ Age < 0 และแถวที่ ScheduledDay อยู่หลัง AppointmentDay;
  Handcap 2–4 ปรับเป็น 1)
- **Split ตามเวลา:** train ≤ 2016-05-20 / val ≤ 2016-05-31 / test ถึง 2016-06-08 ตรวจด้วย Pandera schema

## 6. Features

- `lead_time_days`, วันในสัปดาห์ของวันนัด, กลุ่มอายุ
- จำนวนนัดก่อนหน้าและอัตรา no-show ในอดีต (นับเฉพาะนัดที่ทราบผลแล้วก่อนวันจอง เพื่อกัน leakage)
- `SMS_received`, โรคประจำตัว, `Scholarship`, `Neighbourhood`
- ใช้ preprocessing ชุดเดียวกันทั้งตอน train และ serve (sklearn `Pipeline`) จึงไม่เกิด training-serving skew

## 7. Building Models

| Exp | โมเดล | Val PR-AUC | Val Recall | Test PR-AUC | Test Recall |
|---|---|---|---|---|---|
| 1 | Logistic Regression (baseline) | 0.299 | 0.714 | 0.294 | 0.637 |
| **2** | **LightGBM balanced (เลือกใช้)** | 0.340 | **0.802** | 0.321 | **0.755** |
| 3 | LightGBM tuned (thr 0.598) | **0.345** | 0.603 | 0.329 | 0.540 |

- **เหตุผลที่เลือก Exp 2:** recall บน test สูงที่สุดและผ่าน gate ส่วน Exp 3 มี PR-AUC ดีกว่าเล็กน้อย
  แต่ recall บน test ต่ำกว่า 0.60
- Track ด้วย MLflow (git SHA, data hash, params, metrics, artifacts, environment) และอธิบายผลด้วย SHAP

## 8. Making Predictions (Serving)

- FastAPI: `/predict`, `/predict_batch` (1–1000 แถว), `/health`, `/metrics` โหลดโมเดล alias `champion` จาก registry
- **รูปแบบหลักคือ batch คืนก่อนวันนัด** เพราะการเตือนไม่จำเป็นต้องเร็วระดับวินาที
  ส่วน real-time ใช้ตอนเจ้าหน้าที่จองนัด
- **Latency ที่วัดได้:** p50 ≈ 45 ms ภายใน Docker network (micro-batching + uvicorn workers)
  เทียบกับ SLO p50 ≤ 50 ms, p95 ≤ 200 ms

## 9. Metrics

| ประเภท | ตัวชี้วัด | เกณฑ์ |
|---|---|---|
| Optimizing | PR-AUC คลาส no-show | ยิ่งสูงยิ่งดี |
| Gating (โมเดล) | recall ≥ 0.60, PR-AUC ≥ baseline 0.299, ขนาด ≤ 100 MB, ไม่แย่กว่าโมเดล Production | ไม่ผ่าน = ไม่ promote |
| Gating (ระบบ) | p50 ≤ 50 ms, p95 ≤ 200 ms, error ≤ 1%, uptime ≥ 99% | ตาม `configs/slo.yaml` |
| Business KPI | % ของนัด no-show ที่ถูกเตือนทัน, อัตราการใช้สล็อต, ค่า SMS ต่อ no-show 1 รายที่ป้องกันได้ | — |

**แปลงเป็นภาษาธุรกิจ (test set 26,451 นัด, no-show 18.5%):** โมเดลจับผู้ที่จะไม่มาได้ **75%**
และในทุก ๆ 100 คนที่ถูกเตือน มี **28 คน**ที่เสี่ยงจริง เทียบกับการเตือนแบบสุ่มที่จะเจอประมาณ **18 คน**
โมเดลจึงแม่นกว่าการสุ่มประมาณ **1.5 เท่า** (lift ≈ 1.54)

## 10. Pipeline & Ops

- Prefect DAG: ingest → validate → split → train → evaluate → gate → register → deploy สั่งรันด้วย `make pipeline`
- ถ้า validation หรือ gate ไม่ผ่าน flow จะหยุดและแจ้งเตือนใน **Discord**
- Registry ใช้ alias champion/challenger และ rollback ได้
- CI 3 jobs: code-quality, data-validation, model-gate

## 11. Monitoring & Retraining ⏳

> รอยืนยันกับงาน Monitoring (Issue #6)

- Data drift: Evidently ตรวจการกระจายของอายุ, Neighbourhood, lead time
- Concept drift: เช่น ผลของ SMS ต่อการมาตามนัดเปลี่ยนไป ตรวจจาก recall บน label ที่ได้กลับมาหลังวันนัด
- Grafana แสดง latency, error และ prediction distribution
- เมื่อเกินเกณฑ์ → retrain → ผ่าน gate → promote โมเดลใหม่

## 12. Risks & Ethics

- **ความเป็นธรรม:** โมเดลอาจให้คะแนนเสี่ยงสูงกับบางย่านหรือกลุ่มรายได้ (`Neighbourhood`, `Scholarship`)
  ห้ามนำผลไปใช้**ตัดสิทธิ์หรือลดลำดับคิว** ให้ใช้เพื่อ**เพิ่มการเตือน**เท่านั้น
- **ความเป็นส่วนตัว:** ข้อมูลสุขภาพต้องใช้ PatientId แบบ anonymized และไม่ log ข้อมูลส่วนบุคคลใน API
- **ข้อจำกัด:** ข้อมูลมาจากบราซิลเพียง 6 สัปดาห์ และวันที่ 21–23 กับ 26–29 พ.ค. หายไป
  จึงไม่ควรนำไปใช้กับคลินิกไทยโดยตรงก่อน retrain ด้วยข้อมูลท้องถิ่น
- **Precision ต่ำ (0.28):** ผู้ที่ถูกเตือนส่วนใหญ่มาตามนัดอยู่แล้ว แต่ยอมรับได้เพราะต้นทุน SMS
  ต่ำกว่าการเสียสล็อตแพทย์
