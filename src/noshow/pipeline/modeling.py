"""Modeling tasks for the Prefect pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from prefect import get_run_logger, task
from prefect.cache_policies import NONE
from sklearn.pipeline import Pipeline

from noshow.models.evaluate import choose_threshold, classification_metrics
from noshow.models.train import (
    ROOT,
    experiments,
    file_sha256,
    git_sha,
    load_config,
    make_shap_artifacts,
)


@task(name="train", cache_policy=NONE)
def train_models(
    prepared: dict,
    data_path: Path,
    tracking_uri: str,
) -> dict:
    config = load_config()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("clinic-noshow-modeling")

    source_sha = git_sha()
    data_sha = file_sha256(data_path)
    if data_sha != config["data"]["sha256"]:
        raise ValueError("Dataset changed before training")

    fitted = []
    for name, estimator, tune_threshold in experiments(config["seed"]):
        with mlflow.start_run(run_name=name) as active_run:
            mlflow.set_tags(
                {
                    "git_sha": source_sha,
                    "data_sha256": data_sha,
                    "selected_for_serving": str(
                        name == config["model"]["selected_experiment"]
                    ).lower(),
                }
            )
            mlflow.log_params(
                {key: value for key, value in estimator.get_params().items() if value is not None}
            )
            estimator.fit(
                prepared["train_features"],
                prepared["y_train"],
            )
            model = Pipeline(
                [
                    ("preprocess", prepared["preprocessor"]),
                    ("model", estimator),
                ]
            )
            fitted.append(
                {
                    "name": name,
                    "model": model,
                    "run_id": active_run.info.run_id,
                    "tune_threshold": tune_threshold,
                }
            )
            get_run_logger().info("Trained %s", name)

    return {
        "fitted": fitted,
        "git_sha": source_sha,
        "data_sha256": data_sha,
    }


@task(name="evaluate", cache_policy=NONE)
def evaluate_models(
    trained: dict,
    prepared: dict,
    tracking_uri: str,
) -> dict:
    config = load_config()
    mlflow.set_tracking_uri(tracking_uri)

    selected_name = config["model"]["selected_experiment"]
    selected_threshold = float(config["model"]["selected_threshold"])
    results = {}
    run_ids = {}
    model_uris = {}

    for item in trained["fitted"]:
        name = item["name"]
        model = item["model"]
        val_probability = model.predict_proba(prepared["X_val"])[:, 1]

        if name == selected_name:
            threshold = selected_threshold
        elif item["tune_threshold"]:
            threshold = choose_threshold(
                prepared["y_val"],
                val_probability,
                config["model"]["threshold_min_recall"],
            )
        else:
            threshold = 0.5

        val_metrics = classification_metrics(prepared["y_val"], val_probability, threshold)
        test_probability = model.predict_proba(prepared["X_test"])[:, 1]
        test_metrics = classification_metrics(prepared["y_test"], test_probability, threshold)
        metrics = {f"val_{key}": value for key, value in val_metrics.items()} | {
            f"test_{key}": value for key, value in test_metrics.items()
        }

        with mlflow.start_run(run_id=item["run_id"]):
            mlflow.log_metrics(metrics)
            example = prepared["X_train"].head(3).copy()
            for column in ("ScheduledDay", "AppointmentDay"):
                example[column] = example[column].astype(str)

            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            with tempfile.TemporaryDirectory() as temp:
                artifact_dir = Path(temp)
                (artifact_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
                split_info = {
                    "train_rows": len(prepared["y_train"]),
                    "val_rows": len(prepared["y_val"]),
                    "test_rows": len(prepared["y_test"]),
                    "split_config": config["split"],
                }
                (artifact_dir / "data_split.json").write_text(
                    json.dumps(split_info, indent=2), encoding="utf-8"
                )
                if name == selected_name:
                    make_shap_artifacts(model, prepared["X_val"], artifact_dir)
                mlflow.log_artifacts(artifact_dir, artifact_path="evaluation")

            for filename in ("pyproject.toml", "uv.lock"):
                mlflow.log_artifact(ROOT / filename, artifact_path="environment")
            mlflow.log_artifact(ROOT / "configs/params.yaml", artifact_path="environment")

        results[name] = metrics
        run_ids[name] = item["run_id"]
        model_uris[name] = model_info.model_uri
        get_run_logger().info(
            "%s: validation PR-AUC=%.4f, recall=%.4f",
            name,
            metrics["val_pr_auc"],
            metrics["val_noshow_recall"],
        )

    summary = {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selected_run_id": run_ids[selected_name],
        "selected_model_uri": model_uris[selected_name],
        "reason": "Team-selected model; pending registry gate.",
        "git_sha": trained["git_sha"],
        "data_sha256": trained["data_sha256"],
        "experiments": results,
    }
    (ROOT / "docs/model_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
