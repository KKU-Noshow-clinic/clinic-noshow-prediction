"""Check that the running API gives the same scores as the champion model run locally.

    uv run python scripts/check_serving_parity.py [--n 300] [--api http://localhost:8000]

Loads models:/clinic-noshow@champion from MLflow (default http://localhost:5001), scores N real
appointments from the test split locally, then sends the same rows to /predict_batch and
/predict and compares. Exit code 1 if any score differs by more than 1e-6 or any alert differs.
Use it after every serving change (workers, micro-batching) to show the model output did not
change.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noshow.data.clean import clean  # noqa: E402
from noshow.data.split import time_split  # noqa: E402

FIELDS = [
    "PatientId",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
]


def sample_rows(n: int) -> pd.DataFrame:
    config = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
    df = clean(pd.read_csv(ROOT / config["data"]["raw_path"]))
    _, _, test = time_split(df, config["split"]["train_end"], config["split"]["val_end"])
    return test.sample(n=min(n, len(test)), random_state=0)[FIELDS].reset_index(drop=True)


def to_payload(row: pd.Series) -> dict:
    record = row.to_dict()
    for col in ("ScheduledDay", "AppointmentDay"):
        record[col] = pd.Timestamp(record[col]).isoformat()
    for col in FIELDS[4:]:
        if col != "Neighbourhood":
            record[col] = int(record[col])
    record["PatientId"] = float(record["PatientId"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of test appointments")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("PARITY_TRACKING_URI", "http://localhost:5001")
    )
    args = parser.parse_args()

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    health = requests.get(f"{args.api}/health", timeout=10).json()
    if not health["model_loaded"]:
        print("API has no model loaded yet:", health)
        return 1
    uri = health["model_uri"]
    mlflow.set_tracking_uri(args.tracking_uri)
    name, alias = uri.removeprefix("models:/").split("@")
    version = MlflowClient().get_model_version_by_alias(name, alias)
    threshold = float(version.tags.get("threshold", 0.5))
    if str(version.version) != str(health["model_version"]):
        print(f"Registry champion is v{version.version} but API serves v{health['model_version']}")
        print("Wait for the workers to pick it up (MODEL_POLL_SECONDS), then rerun.")
        return 1

    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    rows = sample_rows(args.n)
    local = model.predict_proba(rows)[:, 1]
    payloads = [to_payload(row) for _, row in rows.iterrows()]

    served: list[dict] = []
    for start in range(0, len(payloads), 100):
        chunk = payloads[start : start + 100]
        resp = requests.post(f"{args.api}/predict_batch", json={"appointments": chunk}, timeout=60)
        resp.raise_for_status()
        served += resp.json()["predictions"]
    single = []
    for payload in payloads[:20]:
        resp = requests.post(f"{args.api}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        single.append(resp.json())

    batch_diff = max(abs(s["noshow_score"] - p) for s, p in zip(served, local, strict=True))
    single_diff = max(abs(s["noshow_score"] - p) for s, p in zip(single, local, strict=False))
    alert_mismatch = sum(s["alert"] != (p >= threshold) for s, p in zip(served, local, strict=True))
    versions = {s["model_version"] for s in served + single}

    print(f"model: {uri} -> v{version.version}, threshold {threshold}")
    print(f"rows compared: {len(served)} via /predict_batch, {len(single)} via /predict")
    print(f"max |score difference|: batch {batch_diff:.2e}, single {single_diff:.2e}")
    print(f"alert mismatches: {alert_mismatch}; versions seen in responses: {sorted(versions)}")
    ok = (
        batch_diff <= 1e-6
        and single_diff <= 1e-6
        and alert_mismatch == 0
        and not any(math.isnan(s["noshow_score"]) for s in served)
    )
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
