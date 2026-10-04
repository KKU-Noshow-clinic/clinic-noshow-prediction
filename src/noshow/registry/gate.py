from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient

from noshow.data.clean import clean
from noshow.data.schema import RAW_SCHEMA
from noshow.data.split import time_split
from noshow.models.evaluate import classification_metrics

ROOT = Path(__file__).resolve().parents[3]


def validation_data(config: dict) -> tuple:
    content = (ROOT / config["data"]["raw_path"]).read_bytes()
    data_sha = hashlib.sha256(content).hexdigest()
    if data_sha != config["data"]["sha256"]:
        raise ValueError("Dataset SHA256 mismatch")

    raw = RAW_SCHEMA.validate(pd.read_csv(io.BytesIO(content)), lazy=True)
    frame = clean(raw)
    train, val, test = time_split(
        frame,
        pd.Timestamp(config["split"]["train_end"], tz="UTC"),
        pd.Timestamp(config["split"]["val_end"], tz="UTC"),
    )
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A data split is empty")

    target = config["target"]["column"]
    y_val = (val[target] == config["target"]["positive_label"]).astype(int)
    if y_val.nunique() != 2:
        raise ValueError("Validation must contain both target classes")

    sizes = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    return val.drop(columns=target), y_val, data_sha, sizes


def evaluate_version(client, version, X_val, y_val, data_sha, sizes, config) -> dict:
    if version.status != "READY":
        raise ValueError(f"Model version {version.version} is not READY")

    run = client.get_run(version.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Training run is not FINISHED")
    if run.data.tags.get("data_sha256") != data_sha:
        raise ValueError("Model training data differs from this gate dataset")

    threshold = float(version.tags["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid model threshold")
    if run.data.metrics.get("val_threshold") != threshold:
        raise ValueError("Threshold differs from the training evaluation")

    with tempfile.TemporaryDirectory() as temp:
        split_path = client.download_artifacts(version.run_id, "evaluation/data_split.json", temp)
        recorded_split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        if any(recorded_split.get(key) != value for key, value in sizes.items()):
            raise ValueError("Recorded data split differs from this gate split")
        if "split_config" in recorded_split and recorded_split["split_config"] != config["split"]:
            raise ValueError("Recorded split configuration differs")

        model_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{version.name}/{version.version}",
                dst_path=temp,
            )
        )
        size_mb = (
            sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file()) / 1_000_000
        )

        model = mlflow.sklearn.load_model(str(model_path))
        probability = model.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val, probability, threshold)

    for key in ("pr_auc", "noshow_recall"):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"Invalid evaluation metric: {key}")

    return {
        "pr_auc": float(metrics["pr_auc"]),
        "recall": float(metrics["noshow_recall"]),
        "size_mb": size_mb,
        "threshold": threshold,
    }


def check_gate(tracking_uri: str, model_name: str | None = None) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    config = yaml.safe_load((ROOT / "configs/params.yaml").read_text(encoding="utf-8"))
    limits = yaml.safe_load((ROOT / "configs/slo.yaml").read_text(encoding="utf-8"))["gate"]

    name = model_name or config["registry"]["model_name"]
    candidate = client.get_model_version_by_alias(name, "challenger")
    registered = client.get_registered_model(name)
    champion_version = registered.aliases.get("champion")

    X_val, y_val, data_sha, sizes = validation_data(config)
    scores = evaluate_version(client, candidate, X_val, y_val, data_sha, sizes, config)
    failures = []
    if scores["pr_auc"] < float(limits["min_pr_auc"]):
        failures.append("PR-AUC is below the minimum")
    if scores["recall"] < float(limits["min_noshow_recall"]):
        failures.append("No-show recall is below the minimum")
    if scores["size_mb"] > float(limits["max_model_size_mb"]):
        failures.append("Model size exceeds the maximum")

    comparison = "First model: no champion to compare"
    production_recall = None
    if champion_version is not None and limits["must_beat_production"]:
        champion = client.get_model_version(name, champion_version)
        production = evaluate_version(client, champion, X_val, y_val, data_sha, sizes, config)
        production_recall = production["recall"]
        comparison = "Both models re-evaluated on the same validation rows"
        if scores["recall"] <= production_recall:
            failures.append("Recall must be higher than champion recall")

    return {
        "model_name": name,
        "version": candidate.version,
        "passed": not failures,
        "val_pr_auc": scores["pr_auc"],
        "val_noshow_recall": scores["recall"],
        "champion_recall": production_recall,
        "model_size_mb": round(scores["size_mb"], 4),
        "threshold": scores["threshold"],
        "evaluation_data_sha256": data_sha,
        "validation_rows": len(y_val),
        "comparison": comparison,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    args = parser.parse_args()
    result = check_gate(args.tracking_uri)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
