# Pipeline / Registry Handoff

งาน Tracking, Registry และ Prefect Pipeline สำหรับ `clinic-noshow`
เชื่อมกับ Serving API ผ่าน `POST /reload`

## ก่อนใช้งาน

- ติดตั้ง dependencies: `uv sync --locked`
- เตรียมข้อมูล `data/raw/noshow.csv` ตาม SHA256 ใน `configs/params.yaml`
- เปิด MLflow ที่ `http://localhost:5001`
- เปิด Prefect ที่ `http://localhost:4200`
- เปิด Serving API ที่รองรับ `/health` และ `POST /reload`
- Pipeline ปัจจุบันใช้ MLflow URL `http://localhost:5001` ในโค้ด

## รัน pipeline บนเครื่อง Windows

ตั้งค่าใน PowerShell ที่โฟลเดอร์โปรเจกต์:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:NOSHOW_API_URL = "http://127.0.0.1:8000"
make pipeline
```

ถ้าไม่มี Make ใช้:

```powershell
uv run python -m noshow.pipeline.flow
```

Flow:
ingest → validate → split → preprocess → train → evaluate
→ register challenger → gate → promote → deploy

- Validation ล้ม: หยุด flow และเรียก failure hook
- Gate ไม่ผ่าน: flow จบโดยไม่ promote และไม่ deploy
- Gate ผ่าน: promote แล้วเรียก `/reload`
- Deploy ตรวจว่า API โหลด version ตรงกับ champion ที่ promote

## Registry และ Gate

- โมเดลหลัก: `clinic-noshow`
- Aliases: `challenger`, `champion`, `previous_champion`
- Threshold เก็บใน model-version tag `threshold`
- Gate อ่านเกณฑ์จาก `configs/slo.yaml`
- Challenger และ champion ถูกประเมินบน validation ชุดเดียวกัน
- เมื่อ `must_beat_production: true` challenger ต้องมี recall สูงกว่า champion
- Recall เท่ากันจะไม่ผ่าน Gate

ตรวจ Gate แยก:

```powershell
uv run python -m noshow.registry.gate --tracking-uri http://localhost:5001
```

คำสั่ง Gate จะ exit code 1 เมื่อโมเดลไม่ผ่าน

## Rollback โมเดลหลักพร้อม reload API

คำสั่งนี้เปลี่ยน champion จริง ใช้เมื่อจำเป็นต้องย้อนกลับเท่านั้น:

```powershell
uv run python -m noshow.registry.rollback --tracking-uri http://localhost:5001 --api-url http://127.0.0.1:8000
```

ต้องมี `previous_champion` และโมเดลเป้าหมายต้อง READY

หากไม่ระบุ `--api-url` จะเปลี่ยน alias ใน Registry อย่างเดียว

หาก promote/rollback สำเร็จ แต่ reload ล้มเหลว:
Registry เปลี่ยนแล้ว แต่ API อาจยังใช้โมเดลเดิม
แก้การเชื่อมต่อแล้วเรียก reload ซ้ำ และตรวจ `/health`:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/reload
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
```

ไม่มี automatic rollback เมื่อ deploy ล้มเหลว

## แจ้งเตือน Discord

Failure hook สร้าง Prefect Markdown Artifact และส่ง Discord webhook
เมื่อกำหนด `PIPELINE_DISCORD_WEBHOOK_URL`

ตั้งค่าใน Terminal ที่จะรัน pipeline:

```powershell
$env:PIPELINE_DISCORD_WEBHOOK_URL = (Read-Host "Paste Discord webhook URL").Trim()
```

เมื่อระบบถาม ให้วาง URL จาก Discord แล้วกด Enter
ห้ามใส่ URL จริงในโค้ดหรือ commit ขึ้น Git

ทดสอบด้วยข้อมูลว่าง:

```powershell
uv run python scripts/check_validation_failure.py
```

คาดหวัง:
- Flow เป็น Failed ด้วย `Dataset is empty`
- มีรายงานใน Prefect Artifacts
- Discord ได้รับข้อความชื่อ flow, run, run ID และสาเหตุ

ถ้าไม่มี webhook จะข้ามการส่ง Discord
หากส่งแจ้งเตือนล้มเหลว จะบันทึกข้อความใน log โดยไม่แทนที่สาเหตุเดิม
ตัวแปรที่ตั้งด้วย `$env:` ใช้เฉพาะ Terminal นั้น ต้องตั้งใหม่เมื่อเปิด Terminal ใหม่

## Demo promote → reload → rollback → reload

ใช้ Registry ชื่อแยกและ API พอร์ต 8001

เตรียมโมเดลต้นทาง:
- Baseline: URI ของ logged sklearn Pipeline ที่มี recall ต่ำกว่า candidate
- Candidate: URI ของ registered model version
- ทั้งสองต้องมีข้อมูลและ split ตรงกับ Gate
- Script setup ใช้ threshold 0.5 จึงต้องตรงกับ `val_threshold` ของทั้งสอง run

สร้าง demo โดยแทน URI ตัวอย่างด้วยค่าจริงของเครื่อง:

```powershell
uv run python scripts/setup_deployment_demo.py --baseline-uri "models:/<baseline-model-id>" --candidate-uri "models:/clinic-noshow/<candidate-version>"
```

จด `model_name` ที่ script แสดง

เปิด Terminal อีกอันใน checkout ที่มี Serving API:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
$env:MODEL_URI = "models:/<demo-model-name>@champion"
uv run uvicorn noshow.serving.app:app --port 8001
```

เปิด Terminal API ค้างไว้ แล้วใช้ Terminal ทดสอบ:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health | ConvertTo-Json
uv run python scripts/check_deployment_cycle.py --model-name "<demo-model-name>" --api-url http://127.0.0.1:8001
```

Script ใช้ Gate และ promote จริง จากนั้น reload, rollback และ reload อีกครั้ง
ตรวจว่า champion หลักไม่เปลี่ยน
ผลสำเร็จ: `DEPLOYMENT CYCLE VERIFIED; main champion unchanged`

## ผลทดสอบในเครื่อง

- Ruff check และ format check ผ่าน
- pytest: 16 passed
- `make pipeline` จบ Completed
- Challenger recall เท่า champion: ไม่ promote และ deployment เป็น None
- Demo ผ่าน Gate แล้ว API เปลี่ยน version 1 → 2
- Rollback demo แล้ว API เปลี่ยน version 2 → 1
- Champion หลักไม่เปลี่ยนระหว่าง demo
- Validation ข้อมูลว่างหยุดงาน และส่งแจ้งเตือน Discord จริงสำเร็จ

## จุดเชื่อมต่อกับทีม

- Serving: API ต้องโหลด `models:/clinic-noshow@champion` และ threshold
  และรองรับ `POST /reload`
- Integration: ตั้ง `NOSHOW_API_URL` ให้เข้าถึง API จากเครื่องที่รัน pipeline
- Monitoring: สามารถเรียก pipeline เพื่อ retrain ได้ โดยยังต้องผ่าน Gate
- CI/CD: unit tests ไม่ต้องใช้ Discord webhook จริง
- ผลทดสอบข้างต้นเป็นการทดสอบในเครื่อง ยังต้องตรวจ CI และระบบรวมหลัง merge