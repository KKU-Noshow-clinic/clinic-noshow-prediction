# ส่งต่องาน Features & Modeling

เอกสารนี้สำหรับเพื่อนที่ทำ Tracking/Registry/Pipeline, Serving/Infra, Monitoring และ CI/CD ต่อจาก Issue #2
สถานะในเอกสารอ้างอิงไฟล์ใน branch `feat/thun-features-modeling` ณ วันที่ 30 กันยายน 2569

## สถานะสำคัญก่อนเริ่มต่อ

- มีโค้ดสร้าง features, preprocessing, train 3 experiments, ประเมินผล, บันทึก MLflow และสร้าง SHAP แล้ว
- รันกับข้อมูลจริงแล้ว แต่ไฟล์งานใน branch นี้ยังอยู่ใน working tree; ยังไม่ได้ commit, push หรือเปิด PR
- ทีมเลือก **Exp 2: LightGBM balanced** สำหรับการส่งเตือนที่เน้นจับผู้ป่วยเสี่ยงให้ได้มาก
  `configs/params.yaml`, `train.py`, `docs/model_results.json` และ `docs/modeling.md` ระบุ Exp 2 ตรงกันแล้ว
  threshold ที่ใช้คือ **0.5**; run ID และ model URI ของรอบล่าสุดอยู่ใน `docs/model_results.json`
- `src/noshow/serving/app.py` ปัจจุบันมีเพียง `/health` และ `/metrics`; ยังไม่มี `/predict`
- `src/noshow/registry/` และ `src/noshow/pipeline/` ยังเป็นโครงเริ่มต้น
- MLflow runs และ dataset อยู่ในเครื่องผู้ทำโมเดลและถูก Git ignore ไว้ จึงไม่ติดไปกับ PR

## สิ่งที่งานโมเดลส่งมอบ

| สิ่งที่ส่งมอบ | ตำแหน่ง | ใช้ทำอะไร |
|---|---|---|
| สร้าง features และประวัติผู้ป่วย | `src/noshow/features/build.py` | แปลงแถวนัดหมายเป็นข้อมูลให้โมเดล |
| Shared preprocessing | `src/noshow/features/pipeline.py` | เติมค่าว่าง, one-hot encode, scale ตัวเลข |
| ฝึกและบันทึก 3 experiments | `src/noshow/models/train.py` | สร้าง sklearn Pipeline ที่รวม preprocessing กับโมเดล |
| คำนวณ metrics/threshold | `src/noshow/models/evaluate.py` | PR-AUC, recall, precision และ threshold |
| พารามิเตอร์และเกณฑ์ | `configs/params.yaml`, `configs/slo.yaml` | split, target, model names และ gate |
| คำอธิบายและผล | `docs/modeling.md`, `docs/model_results.json` | หลักฐานสำหรับรายงานและ PR |
| Tests | `tests/test_features.py`, `tests/test_models.py` | ตรวจ feature history, unseen categories และ threshold |

ตัวโมเดลที่ log เข้า MLflow เป็น **sklearn Pipeline ทั้งก้อน** ไม่ใช่ LightGBM อย่างเดียว
ขั้น `preprocess` ใน pipeline ต้องทำงานตอนทำนายด้วยเพื่อป้องกันข้อมูลที่ใช้ train กับ serve ต่างกัน
โมเดลใช้ `cloudpickle` สำหรับ serialization; ให้โหลดเฉพาะ artifact จากแหล่งที่ทีมเชื่อถือ

## ข้อมูลนำเข้าที่ทีม Serving ต้องเตรียม

pipeline รับตารางที่มีอย่างน้อยคอลัมน์ต่อไปนี้:

| กลุ่ม | คอลัมน์ |
|---|---|
| ระบุตัวผู้ป่วยและเวลา | `PatientId`, `ScheduledDay`, `AppointmentDay` |
| ข้อมูลหมวดหมู่ | `Gender`, `Neighbourhood` |
| ข้อมูลตัวเลข | `Age`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received` |

`ScheduledDay` และ `AppointmentDay` ต้องแปลงเป็นวันเวลาได้ โดย pipeline รองรับ timestamp string
เช่น `2016-05-04T08:00:00Z` และวันที่เช่น `2016-05-05` คอลัมน์ `No-show` เป็นคำตอบจริง
สำหรับ train/evaluation เท่านั้น **ไม่ต้องส่งเข้า API ตอนทำนาย**

`previous_appointments` และ `previous_noshow_rate` ถูกคำนวณภายใน pipeline
โดยนับเฉพาะผลนัดที่เกิดก่อน **วันที่จองนัดปัจจุบัน** โมเดลที่ฝึกแล้วมีประวัติจาก train set เท่านั้น
หากต้องการให้ประวัติเพิ่มตามข้อมูลจริงหลัง deploy ทีมต้องออกแบบวิธีอัปเดต/ฝึกโมเดลใหม่
ชุดข้อมูลไม่มีเวลาบันทึกผล no-show จึงใช้สมมติฐานว่ารู้ผลได้ตั้งแต่วันถัดจากวันนัด

## ผลทดลองที่ใช้ประกอบการตัดสินใจ

ข้อมูลถูกแบ่งตาม `AppointmentDay`: train 67,360, validation 16,711 และ test 26,450 นัด
`No-show == "Yes"` คือคลาส 1 ข้อมูล test มี no-show ประมาณ 18.46%

| Experiment | Validation PR-AUC | Validation recall | Test PR-AUC | Test recall | Test precision | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Exp 1: Logistic Regression | 0.2989 | 0.7141 | 0.2936 | 0.6374 | 0.2763 | 0.5 |
| Exp 2: LightGBM balanced | 0.3399 | 0.8025 | 0.3215 | **0.7548** | 0.2841 | **0.5** |
| Exp 3: LightGBM tuned | **0.3452** | 0.6025 | **0.3294** | 0.5404 | **0.3279** | 0.5978 |

Exp 2 ที่ threshold 0.5 จับผู้ป่วย no-show ใน test ได้ประมาณ 75 จาก 100 คนที่ไม่มาจริง
แต่ใน 100 คนที่ถูกเลือกให้เตือน จะไม่มาจริงประมาณ 28 คน Exp 3 ส่งเตือนน้อยกว่าและ precision สูงกว่า
ผลต่าง recall ส่วนหนึ่งอาจเกิดจาก threshold ที่ต่างกัน จึงควรเทียบสองโมเดลที่ **งบส่งเตือนเท่ากัน**
หรือ **recall เป้าหมายเท่ากัน** ก่อนยืนยันโมเดลสำหรับการใช้งานจริง

**มติทีม:** ใช้ Exp 2 สำหรับการส่ง SMS/โทรเตือนที่เน้น recall โดยผ่าน validation recall
และ PR-AUC ขั้นต่ำใน `configs/slo.yaml` แล้ว `selected_model` ในผลรันล่าสุดเป็น Exp 2
README อธิบายเกณฑ์นี้ตรงกัน ส่วนการเทียบที่งบส่งเตือนเท่ากันยังเป็นงานก่อนใช้งานจริง

## งานต่อสำหรับ Tracking, Registry & Pipeline

1. จัด MLflow tracking และ artifact store ที่ทุกคนเข้าถึงได้ แล้วรันสาม experiments ใหม่กับ code commit
   ที่จะส่ง PR; run เดิมอยู่ในเครื่องผู้ทำโมเดล และ Git SHA ใน `docs/model_results.json` เป็น SHA ก่อน commit
2. ใช้ `selected_run_id`/`selected_model_uri` ใน `docs/model_results.json` ระบุ artifact ของ Exp 2
   และจัดเก็บ threshold `0.5` ควบคู่กับ model version (threshold อยู่ใน metric/JSON และ config
   ยังไม่ได้ฝังใน serialized pipeline)
3. กำหนด gate ก่อน promote ตาม `configs/slo.yaml` รวมถึง PR-AUC, recall, ขนาดโมเดล
   และการเทียบกับ Production; เขียนและสาธิต rollback ตามขอบเขตงาน Registry
4. เชื่อมขั้น validate → clean/split → train → evaluate → gate → registry เป็น pipeline ที่รันซ้ำได้

## งานต่อสำหรับ Serving & Infra

1. โหลด model artifact ของ Exp 2 จาก registry/tracking ที่ทีมตกลงกัน และโหลด threshold ของ version เดียวกัน
2. สร้าง `/predict` และ `/predict_batch`; ส่งข้อมูลตามตาราง input ข้างต้นเข้า pipeline ทั้งก้อน
   โดยใช้ `predict_proba(X)[:, 1]` เป็นคะแนน no-show และใช้ `score >= 0.5` สำหรับ Exp 2
3. ตรวจการโหลดโมเดลจริงใน `/health` และทดสอบกับข้อมูลที่ถูกต้อง/ผิด schema
4. ยืนยันว่าผลจาก API ตรงกับการทำนายจาก pipeline ในเครื่องสำหรับ input เดียวกัน
5. ระวังว่าคะแนนจากโมเดลที่ใช้ class weight ยังไม่ผ่านการ calibration
   จึงไม่ควรแสดงเป็น "โอกาสไม่มาเป็นเปอร์เซ็นต์ที่แม่นยำ" โดยไม่มีการตรวจเพิ่มเติม

## งานต่อสำหรับ Monitoring และ CI/CD

- Monitoring: ติดตามสัดส่วน no-show จริง, PR-AUC/recall/precision เมื่อมีผลจริง,
  ปริมาณการแจ้งเตือน, data drift และ concept drift พร้อมเกณฑ์แจ้งเตือนและนโยบาย retrain
- CI/CD: รัน Ruff, pytest, data validation และ model quality gate; เพิ่ม test การโหลด artifact
  และ prediction ผ่าน API เมื่อ Serving เชื่อมแล้ว

## วิธีรันซ้ำและข้อควรตรวจ

จาก root ของ repository:

```powershell
uv sync
uv run python scripts/download_data.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -m noshow.models.train
```

ข้อมูล version นี้มี SHA-256:
`9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`

**ตรวจงานก่อน merge:** รัน checks และตรวจ diff ของทุกไฟล์ก่อน commit/PR
MLflow run และ `selected_model_uri` ใน JSON ปัจจุบันเป็นของเครื่องผู้ทำโมเดล
เมื่อย้ายไป shared tracking server ต้องรันใหม่และอัปเดต URI
