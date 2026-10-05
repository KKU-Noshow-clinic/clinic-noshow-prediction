# Features & Modeling — งานตาม Issue #2

เอกสารนี้อธิบายสิ่งที่ทำใน branch `feat/thun-features-modeling` ตาม checklist ของ Issue #2
โดยใช้โค้ด Data & Validation ที่ merge เข้า `main` แล้วเป็นจุดเริ่มต้น ข้อมูลดิบมาจากชุด
Medical Appointment No Shows และใช้ `No-show == "Yes"` เป็นคลาส 1 (ผู้ป่วยไม่มาตามนัด)
ทีมเลือกโมเดลสำหรับส่งเตือนโดยเน้น **no-show recall** ที่ threshold ที่กำหนด
และใช้ PR-AUC ของคลาสนี้เป็นเกณฑ์ขั้นต่ำและตัวชี้วัดประกอบ

## 1. Shared preprocessing ใน `src/noshow/features/`

สร้าง `AppointmentFeatureBuilder` ใน `src/noshow/features/build.py` และ
`build_preprocessor()` ใน `src/noshow/features/pipeline.py` แล้วประกอบเข้ากับโมเดลเป็น
scikit-learn `Pipeline` เดียวกัน:

```python
Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
```

pipeline สร้าง features ก่อน แล้วให้ `ColumnTransformer` จัดการข้อมูลสองกลุ่ม:

- ตัวเลข: เติมค่าว่างด้วย median และปรับสเกลด้วย `StandardScaler`
- หมวดหมู่: เติมค่าว่างด้วยค่าที่พบบ่อยที่สุดและแปลงด้วย `OneHotEncoder`
  โดยตั้ง `handle_unknown="ignore"` เพื่อรองรับหมวดหมู่ใหม่ตอนทำนาย

บันทึก pipeline ทั้งก้อนเป็น model artifact ใน MLflow เพื่อให้ทีม Serving โหลดตัวแปลงข้อมูล
และโมเดลชุดเดียวกับที่ใช้ตอนฝึก วิธีนี้เป็นกลไกป้องกัน training-serving skew ในระดับโค้ด
แต่ API ของทีม Serving ยังต้องเชื่อมและทดสอบการทำนายจริงแยกต่างหาก

## 2. Features ที่สร้างและการป้องกันข้อมูลรั่ว

ใน `build.py` สร้าง features ตาม Issue ดังนี้:

| Feature | วิธีคำนวณ |
|---|---|
| `lead_time_days` | จำนวนวันจาก `ScheduledDay` ถึง `AppointmentDay` โดยไม่ให้ค่าติดลบ |
| `appointment_weekday` | ชื่อวันในสัปดาห์ของวันนัด |
| `age_group` | แบ่งอายุเป็น child, teen, young adult, adult, senior และ elderly |
| `previous_appointments` | จำนวนนัดของผู้ป่วยคนเดิมที่ทราบผลแล้วก่อนวันจองนัดปัจจุบัน |
| `previous_noshow_rate` | สัดส่วน no-show ของนัดก่อนหน้าที่ทราบผลแล้ว |

สำหรับประวัติผู้ป่วย เราสมมติอย่างระมัดระวังว่าทราบผลนัดได้ตั้งแต่ **วันถัดจากวันนัด**
จึงนับเฉพาะ `AppointmentDay` ที่อยู่ **ก่อน** วันที่ของ `ScheduledDay` ปัจจุบันอย่างเคร่งครัด
นัดที่เกิดในวันจองเดียวกันหรือนัดในอนาคตไม่นับ เมื่อไม่มีประวัติให้จำนวนครั้งและอัตราเป็น 0
ระหว่าง validation, test และการทำนาย pipeline ใช้เฉพาะประวัติจากชุด train ที่เรียนไว้ใน `fit`
ไม่มีการอ่านผลจริงของ validation/test เพื่อสร้าง feature ให้แถวเหล่านั้น

ข้อจำกัด: ชุดข้อมูลไม่มี timestamp ที่บอกเวลาบันทึกผล no-show จริง สมมติฐานเรื่อง
"ทราบผลวันถัดไป" จึงควรตรวจซ้ำกับข้อมูลระบบคลินิกจริงก่อนนำไปใช้งานจริง
tests ใน `tests/test_features.py` ครอบคลุมกรณีจองนัดใหม่ก่อนนัดเก่าจะเกิดขึ้น
และกรณีนัดเก่าเกิดในวันเดียวกับวันจอง

## 3. Experiment 1 — Logistic Regression baseline

`src/noshow/models/train.py` เรียก `clean()` และ `time_split()` จากงาน Data & Validation
แบ่ง train/validation/test ตาม `AppointmentDay` ที่ระบุใน `configs/params.yaml` ไม่สุ่มข้อมูล
จากนั้นใช้ `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`
เป็น baseline และใช้ threshold 0.5 เพื่อรายงาน precision กับ recall

ผล validation: PR-AUC **0.2989**, no-show recall **0.7141**
ผล test: PR-AUC **0.2936**, no-show recall **0.6374**
ค่า validation PR-AUC ของ baseline ถูกตั้งเป็น `gate.min_pr_auc` ใน `configs/slo.yaml`

## 4. Experiment 2 — LightGBM พร้อม class weight

ใช้ `LGBMClassifier` จำนวน 200 trees, `class_weight="balanced"`, seed 42
และ threshold 0.5 การให้น้ำหนักคลาสช่วยรับมือข้อมูล no-show ที่มีสัดส่วนน้อยกว่าคลาสมาตามนัด

ผล validation: PR-AUC **0.3399**, no-show recall **0.8025**
ผล test: PR-AUC **0.3215**, no-show recall **0.7548**, precision **0.2841**
ทีมเลือก Exp 2 สำหรับการส่งเตือนที่เน้นจับผู้ป่วยเสี่ยงให้ได้มาก โดยใช้ threshold **0.5**
การเลือกนี้ถูกตั้งใน `configs/params.yaml` (`selected_experiment`, `selected_threshold`)
และ `train.py` ตรวจว่า validation recall และ PR-AUC ผ่านค่าขั้นต่ำใน `configs/slo.yaml`

## 5. Experiment 3 — LightGBM ปรับพารามิเตอร์และเลือก threshold

ใช้ LightGBM ที่ปรับพารามิเตอร์ไว้ใน `experiments()` เช่น `n_estimators=400`,
`learning_rate=0.03`, `max_depth=8`, `min_child_samples=30`, `reg_lambda=1.0`
และ `class_weight="balanced"` การปรับนี้เป็นการกำหนดชุดพารามิเตอร์สำหรับการทดลองที่สาม
ไม่ได้ใช้ grid search หรือ automated hyperparameter search

`choose_threshold()` ใน `src/noshow/models/evaluate.py` คำนวณจาก **validation set เท่านั้น**:
เลือก threshold ที่ precision สูงที่สุดในกลุ่มที่ no-show recall ไม่น้อยกว่า 0.60
ได้ threshold **0.5978**, validation recall **0.6025** และ validation PR-AUC **0.3452**
threshold ของ Exp 3 ถูก log เป็น metric ใน MLflow และอยู่ใน `docs/model_results.json`
เพื่อเปรียบเทียบกับ Exp 2 แต่ไม่ใช่ threshold ของโมเดลที่ทีมเลือกใช้

ผล test ที่ threshold เดิม: PR-AUC **0.3294**, recall **0.5404** และ precision **0.3279**
recall บน test ต่ำกว่า 0.60 จึงต้องรายงานตามจริง เราไม่ใช้ test set ปรับ threshold
เพราะจะทำให้การประเมินผลสุดท้ายไม่เป็นอิสระ

## 6. บันทึกการทดลองลง MLflow ครบ 6 อย่าง

ทุก experiment ใน `train.py` สร้าง MLflow run ภายใต้ experiment `clinic-noshow-modeling`
และบันทึกหลักฐานตาม Issue:

| สิ่งที่ต้องบันทึก | หลักฐานที่บันทึก |
|---|---|
| เวอร์ชันโค้ด | Git SHA เป็น tag `git_sha` |
| เวอร์ชันข้อมูล | SHA-256 ของ `data/raw/noshow.csv` เป็น tag `data_sha256` |
| ไฮเปอร์พารามิเตอร์ | ค่าจาก `estimator.get_params()` |
| ตัวชี้วัด | validation/test PR-AUC, no-show recall, precision และ threshold |
| ไฟล์ผลลัพธ์ | serialized pipeline, `metrics.json`, `data_split.json` และ SHAP ของ Exp 2 |
| สภาพแวดล้อม | `pyproject.toml`, `uv.lock` และ environment ของ model artifact |

SHA-256 ของข้อมูลรอบนี้คือ
`9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`
run ในเครื่องอยู่ใน `mlruns/` และ `mlflow.db` ซึ่ง Git ignore ไว้ ก่อนส่งหลักฐานให้ทีม
ควรตรวจว่าจะเก็บ MLflow artifacts ไว้ที่ใด เพราะไฟล์เหล่านี้ไม่ได้ไปกับ Pull Request
Git SHA ใน `docs/model_results.json` เป็น SHA ของ commit ที่มีอยู่ตอนรัน; เมื่อ commit งานใหม่นี้แล้ว
ควรรันอีกครั้งหากต้องการให้ MLflow อ้าง SHA ของโค้ดที่ส่ง PR ตรงกัน

## 7. SHAP และเหตุผลเลือกโมเดลสุดท้าย

Exp 2 ซึ่งเป็นโมเดลที่ทีมเลือก ใช้ `shap.TreeExplainer` กับข้อมูลตัวอย่างจาก validation set สูงสุด 1,000 แถว
สร้าง `shap_importance.csv` (ค่า mean absolute SHAP) และ `shap_summary.png`
แล้ว log ทั้งสองไฟล์เป็น artifact ของ run นั้น ค่า mean absolute SHAP สูงไม่ได้แปลว่า
feature เพิ่มหรือลดโอกาส no-show เสมอไป; ให้ดูทิศทางใน summary plot เพิ่มเติม

อันดับ feature ที่มีผลมากที่สุดใน run ล่าสุดคือ `lead_time_days`, `Age`,
`SMS_received`, `previous_appointments` และ `previous_noshow_rate`

| โมเดล | Validation PR-AUC | Validation recall | Threshold | Test PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2989 | 0.7141 | 0.5000 | 0.2936 |
| **LightGBM balanced (เลือกใช้)** | 0.3399 | **0.8025** | 0.5000 | 0.3215 |
| LightGBM tuned | **0.3452** | 0.6025 | 0.5978 | **0.3294** |

เลือก **Exp 2: LightGBM balanced** ตามมติทีมที่ให้ความสำคัญกับการจับผู้ป่วยเสี่ยง
เพื่อส่ง SMS/โทรเตือน Exp 2 มี validation recall 0.8025 และผ่านเกณฑ์ PR-AUC ขั้นต่ำ
แม้ Exp 3 จะมี PR-AUC สูงกว่าเล็กน้อย แต่ที่ threshold ที่ทดลองจับผู้ป่วย no-show ได้น้อยกว่า
บน test Exp 2 มี recall 0.7548; precision 0.2841 หมายความว่าใน 100 คนที่ถูกเลือกให้เตือน
จะไม่มาจริงประมาณ 28 คน จึงต้องคำนึงถึงต้นทุนและจำนวนการเตือนที่คลินิกรองรับได้
threshold ของแต่ละโมเดลต่างกัน การเปรียบเทียบที่งบส่งเตือนเท่ากันควรทำเพิ่มเติมก่อนใช้งานจริง
ผล test ใช้รายงาน ไม่ได้ใช้ปรับ threshold

## วิธีรันซ้ำและตรวจผล

รันจาก root ของ repository หลังติดตั้ง dependencies ด้วย `uv sync` และดาวน์โหลดข้อมูล:

```powershell
uv run python scripts/download_data.py
uv run python -m noshow.models.train
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

ผลสรุปที่อ่านง่ายอยู่ใน `docs/model_results.json` ขณะที่ MLflow เก็บรายละเอียดแต่ละ run
การตรวจล่าสุด: Ruff ผ่าน, format ผ่าน และ pytest ผ่าน **11 tests**
