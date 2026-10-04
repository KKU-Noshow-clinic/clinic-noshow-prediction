from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from noshow.data.clean import clean
from noshow.data.split import time_split
from noshow.features import build_preprocessor
from noshow.models.evaluate import (
    choose_threshold,
    classification_metrics,
    validate_selected_model,
)

ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def load_config() -> dict:
    with (ROOT / "configs" / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def experiments(seed: int) -> list[tuple[str, object, bool]]:
    return [
        (
            "exp1_logistic_regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            False,
        ),
        (
            "exp2_lightgbm_balanced",
            lgb.LGBMClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
            ),
            False,
        ),
        (
            "exp3_lightgbm_tuned",
            lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=8,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
            ),
            True,
        ),
    ]


def make_shap_artifacts(model: Pipeline, X: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sample = X.sample(min(1000, len(X)), random_state=42)
    transformed = model.named_steps["preprocess"].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = model.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
    explainer = shap.TreeExplainer(model.named_steps["model"])
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(output_dir / "shap_importance.csv", index=False)
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def run(data_path: Path, tracking_uri: str | None = None) -> dict:
    config = load_config()
    raw = pd.read_csv(data_path)
    frame = clean(raw)
    train, val, test = time_split(
        frame,
        pd.Timestamp(config["split"]["train_end"], tz="UTC"),
        pd.Timestamp(config["split"]["val_end"], tz="UTC"),
    )
    target = config["target"]["column"]
    positive = config["target"]["positive_label"]
    X_train, y_train = train.drop(columns=target), (train[target] == positive).astype(int)
    X_val, y_val = val.drop(columns=target), (val[target] == positive).astype(int)
    X_test, y_test = test.drop(columns=target), (test[target] == positive).astype(int)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "The configured time split produced an empty train, validation, or test set"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("clinic-noshow-modeling")
    source_sha = git_sha()
    data_sha = file_sha256(data_path)
    selected_name = config["model"]["selected_experiment"]
    selected_threshold = float(config["model"]["selected_threshold"])
    results = {}
    run_ids = {}
    model_uris = {}

    for name, estimator, tune_threshold in experiments(config["seed"]):
        model = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=name) as active_run:
            model.fit(X_train, y_train)
            val_probability = model.predict_proba(X_val)[:, 1]
            if name == selected_name:
                threshold = selected_threshold
            elif tune_threshold:
                threshold = choose_threshold(
                    y_val, val_probability, config["model"]["threshold_min_recall"]
                )
            else:
                threshold = 0.5
            val_metrics = classification_metrics(y_val, val_probability, threshold)
            test_probability = model.predict_proba(X_test)[:, 1]
            test_metrics = {
                f"test_{key}": value
                for key, value in classification_metrics(
                    y_test, test_probability, threshold
                ).items()
            }
            metrics = {f"val_{key}": value for key, value in val_metrics.items()} | test_metrics
            params = estimator.get_params()
            mlflow.set_tags(
                {
                    "git_sha": source_sha,
                    "data_sha256": data_sha,
                    "selected_for_serving": str(name == selected_name).lower(),
                }
            )
            mlflow.log_params({key: value for key, value in params.items() if value is not None})
            mlflow.log_metrics(metrics)
            input_example = X_train.head(3).copy()
            input_example["ScheduledDay"] = input_example["ScheduledDay"].astype(str)
            input_example["AppointmentDay"] = input_example["AppointmentDay"].astype(str)
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            with tempfile.TemporaryDirectory() as temp:
                artifact_dir = Path(temp)
                (artifact_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
                (artifact_dir / "data_split.json").write_text(
                    json.dumps(
                        {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if name == selected_name:
                    make_shap_artifacts(model, X_val, artifact_dir)
                mlflow.log_artifacts(artifact_dir, artifact_path="evaluation")
            mlflow.log_artifact(ROOT / "pyproject.toml", artifact_path="environment")
            mlflow.log_artifact(ROOT / "uv.lock", artifact_path="environment")
            results[name] = metrics
            run_ids[name] = active_run.info.run_id
            model_uris[name] = model_info.model_uri

    with (ROOT / "configs" / "slo.yaml").open(encoding="utf-8") as stream:
        gate = yaml.safe_load(stream)["gate"]
    validate_selected_model(
        results,
        selected_name,
        min_recall=float(gate["min_noshow_recall"]),
        min_pr_auc=float(gate["min_pr_auc"]),
    )
    summary = {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selected_run_id": run_ids[selected_name],
        "selected_model_uri": model_uris[selected_name],
        "reason": (
            "Team selected higher no-show recall for reminder outreach; "
            "the selected run passes validation recall and PR-AUC gates."
        ),
        "git_sha": source_sha,
        "data_sha256": data_sha,
        "experiments": results,
    }
    output = ROOT / "docs" / "model_results.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "noshow.csv")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
