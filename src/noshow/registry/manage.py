from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[3]


def register_selected(tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    config = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "docs" / "model_results.json").read_text(encoding="utf-8"))

    name = config["registry"]["model_name"]
    run_id = summary["selected_run_id"]
    model_uri = summary["selected_model_uri"]
    threshold = float(summary["selected_threshold"])

    if summary["selected_model"] != config["model"]["selected_experiment"]:
        raise ValueError("Selected experiment does not match params.yaml")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1")
    if threshold != float(config["model"]["selected_threshold"]):
        raise ValueError("Threshold does not match params.yaml")

    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError("The selected training run has not finished")
    if run.data.tags.get("selected_for_serving") != "true":
        raise ValueError("This run was not selected for serving")
    if run.data.tags.get("data_sha256") != summary["data_sha256"]:
        raise ValueError("Dataset hash does not match the selected run")
    if run.data.metrics.get("val_threshold") != threshold:
        raise ValueError("Threshold does not match the selected run")

    # Verify that the logged model belongs to this training run.
    logged_model = mlflow.get_logged_model(model_uri.removeprefix("models:/"))
    if logged_model.source_run_id != run_id:
        raise ValueError("Model artifact does not belong to the selected run")

    # Reusing the same run and artifact should not create duplicate versions.
    existing = client.search_model_versions(f"name = '{name}'")
    version = next(
        (
            item
            for item in existing
            if item.run_id == run_id and item.tags.get("original_model_uri") == model_uri
        ),
        None,
    )

    if version is None:
        version = mlflow.register_model(model_uri=model_uri, name=name)

    tags = {
        "threshold": str(threshold),
        "selected_experiment": summary["selected_model"],
        "original_model_uri": model_uri,
        "git_sha": run.data.tags["git_sha"],
        "data_sha256": run.data.tags["data_sha256"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)

    client.set_registered_model_alias(name, "challenger", version.version)

    return {
        "model_name": name,
        "version": version.version,
        "alias": "challenger",
        "model_uri": f"models:/{name}/{version.version}",
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    args = parser.parse_args()
    result = register_selected(args.tracking_uri)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
