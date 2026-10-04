from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yaml
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NONE

from noshow.data.clean import clean
from noshow.data.schema import RAW_SCHEMA
from noshow.data.split import time_split
from noshow.features import build_preprocessor
from noshow.pipeline.modeling import evaluate_models, train_models
from noshow.registry.gate import check_gate
from noshow.registry.manage import register_selected
from noshow.registry.promote import promote

ROOT = Path(__file__).resolve().parents[3]


@task(name="ingest", cache_policy=NONE)
def ingest(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    config = yaml.safe_load((ROOT / "configs/params.yaml").read_text(encoding="utf-8"))
    # Hash and parse the same bytes.
    content = data_path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    expected = config["data"]["sha256"]
    if actual != expected:
        raise ValueError(f"Dataset SHA256 mismatch: {actual}")

    import io

    frame = pd.read_csv(io.BytesIO(content))
    get_run_logger().info("Loaded %s rows; SHA256 verified", len(frame))
    return frame


@task(name="validate", cache_policy=NONE)
def validate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Dataset is empty")

    validated = RAW_SCHEMA.validate(frame, lazy=True)
    for column in ("ScheduledDay", "AppointmentDay"):
        pd.to_datetime(validated[column], utc=True, errors="raise")

    get_run_logger().info("Raw schema and dates passed validation")
    return validated


@task(name="split", cache_policy=NONE)
def split_data(frame: pd.DataFrame) -> tuple:
    config = yaml.safe_load((ROOT / "configs/params.yaml").read_text(encoding="utf-8"))
    cleaned = clean(frame)
    train, val, test = time_split(
        cleaned,
        pd.Timestamp(config["split"]["train_end"], tz="UTC"),
        pd.Timestamp(config["split"]["val_end"], tz="UTC"),
    )
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train, validation, or test split is empty")

    target = config["target"]["column"]
    for label, partition in (("train", train), ("val", val), ("test", test)):
        if partition[target].nunique() != 2:
            raise ValueError(f"{label} split must contain both target classes")

    get_run_logger().info(
        "Cleaned rows: %s; train=%s, val=%s, test=%s",
        len(cleaned),
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


@task(name="preprocess", cache_policy=NONE)
def preprocess_data(train, val, test) -> dict:
    config = yaml.safe_load((ROOT / "configs/params.yaml").read_text(encoding="utf-8"))
    target = config["target"]["column"]
    positive = config["target"]["positive_label"]

    X_train = train.drop(columns=target)
    X_val = val.drop(columns=target)
    X_test = test.drop(columns=target)

    preprocessor = build_preprocessor()
    y_train = (train[target] == positive).astype(int)
    train_features = preprocessor.fit_transform(X_train, y_train)
    val_features = preprocessor.transform(X_val)
    test_features = preprocessor.transform(X_test)

    get_run_logger().info(
        "Feature shapes: train=%s, val=%s, test=%s",
        train_features.shape,
        val_features.shape,
        test_features.shape,
    )
    return {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "y_train": (train[target] == positive).astype(int),
        "y_val": (val[target] == positive).astype(int),
        "y_test": (test[target] == positive).astype(int),
    }


@task(name="register-challenger", cache_policy=NONE)
def register_model(summary: dict, tracking_uri: str) -> dict:
    registered = register_selected(tracking_uri)
    get_run_logger().info(
        "Registered challenger version %s from run %s",
        registered["version"],
        summary["selected_run_id"],
    )
    return registered


@task(name="gate", cache_policy=NONE)
def gate_model(registered: dict, tracking_uri: str) -> dict:
    result = check_gate(tracking_uri)
    if result["version"] != registered["version"]:
        raise RuntimeError("Challenger changed during this pipeline run")

    if result["passed"]:
        get_run_logger().info("Model gate passed")
    else:
        get_run_logger().warning("Model will not be promoted: %s", result["failures"])
    return result


@task(name="promote", cache_policy=NONE)
def promote_model(gate_result: dict, tracking_uri: str) -> dict:
    if not gate_result["passed"]:
        raise ValueError("Cannot promote a model that failed the gate")

    result = promote(tracking_uri)
    if result["champion_version"] != gate_result["version"]:
        raise RuntimeError("Promoted version differs from pipeline candidate")
    return result


@task(name="deploy", cache_policy=NONE)
def deploy_model(promotion: dict) -> dict:
    api_url = os.getenv("NOSHOW_API_URL", "http://localhost:8000").rstrip("/")
    request = Request(f"{api_url}/reload", method="POST")

    with urlopen(request, timeout=120) as response:
        result = json.load(response)

    expected = str(promotion["champion_version"])
    loaded = str(result.get("model_version"))
    if loaded != expected:
        raise RuntimeError(f"API loaded version {loaded}; expected champion version {expected}")

    get_run_logger().info("API loaded champion version %s", loaded)
    return result


def send_discord_alert(message: str) -> None:
    webhook = os.getenv("PIPELINE_DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print("Discord notification skipped: webhook is not configured")
        return

    separator = "&" if "?" in webhook else "?"
    payload = {
        "content": message[:2000],
        "allowed_mentions": {"parse": []},
    }

    try:
        request = Request(
            f"{webhook}{separator}wait=true",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Clinic-NoShow-Pipeline/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=15) as response:
            response.read()
        print("Discord failure notification sent")
    except Exception as exc:
        print(f"Discord notification failed ({type(exc).__name__})")


def notify_failure(flow, flow_run, state) -> None:
    message = (
        f"Flow: {flow.name}\n\n"
        f"Run: {flow_run.name}\n\n"
        f"Run ID: {flow_run.id}\n\n"
        f"State: {state.name}\n\n"
        f"Reason: {state.message}"
    )
    print(f"PIPELINE FAILURE\n{message}")

    try:
        create_markdown_artifact(
            key="clinic-noshow-pipeline-failure",
            markdown=message,
            description="Pipeline failure report",
        )
    except Exception as exc:
        print(f"Failure artifact creation failed ({type(exc).__name__})")

    send_discord_alert(f"PIPELINE FAILURE\n{message}")


@flow(
    name="clinic-noshow-pipeline",
    log_prints=True,
    on_failure=[notify_failure],
)
def pipeline() -> dict:
    config = yaml.safe_load((ROOT / "configs/params.yaml").read_text(encoding="utf-8"))
    data_path = ROOT / config["data"]["raw_path"]
    raw = ingest(data_path)
    validated = validate(raw)
    train, val, test = split_data(validated)
    prepared = preprocess_data(train, val, test)
    trained = train_models(prepared, data_path, "http://localhost:5001")
    evaluated = evaluate_models(trained, prepared, "http://localhost:5001")
    tracking_uri = "http://localhost:5001"
    registered = register_model(evaluated, tracking_uri)
    gate_result = gate_model(registered, tracking_uri)
    promotion = promote_model(gate_result, tracking_uri) if gate_result["passed"] else None
    deployment = deploy_model(promotion) if promotion is not None else None

    return {
        "validated_rows": len(validated),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "feature_count": prepared["train_features"].shape[1],
        "trained_models": len(trained["fitted"]),
        "selected_run_id": evaluated["selected_run_id"],
        "challenger_version": registered["version"],
        "gate_passed": gate_result["passed"],
        "promotion_status": ("promoted" if promotion is not None else "not_promoted"),
        "deployment": deployment,
    }


if __name__ == "__main__":
    print(pipeline())
