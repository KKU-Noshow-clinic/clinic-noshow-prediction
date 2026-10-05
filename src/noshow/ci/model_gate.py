"""CI model gate (Quality Gate): selected model metrics vs the gate in configs/slo.yaml.

    uv run python -m noshow.ci.model_gate [--results docs/model_results.json]

CI has no raw dataset and no MLflow server, so this checks the validation metrics that the
modeling run recorded in docs/model_results.json against the absolute thresholds
(min PR-AUC, min no-show recall). The relative threshold (must beat the champion) and the
real model size are checked by the registry gate in the Prefect pipeline.
Exit code 1 = gate failed, so the CI job fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def check_results(results: dict, params: dict, gate: dict) -> list[str]:
    """Return the list of failed checks (empty list = model passes the gate)."""
    failures = []

    # 1. The results must belong to the model and data the team selected in params.yaml.
    selected = results.get("selected_model")
    if selected != params["model"]["selected_experiment"]:
        failures.append(f"selected_model {selected!r} is not the model in params.yaml")
    if results.get("data_sha256") != params["data"]["sha256"]:
        failures.append("data_sha256 is not the dataset in params.yaml")

    metrics = results.get("experiments", {}).get(selected)
    if metrics is None:
        failures.append(f"no metrics recorded for {selected!r}")
        return failures

    # 2. Absolute thresholds from configs/slo.yaml.
    if metrics["val_pr_auc"] < gate["min_pr_auc"]:
        failures.append(f"PR-AUC {metrics['val_pr_auc']:.4f} < {gate['min_pr_auc']}")
    if metrics["val_noshow_recall"] < gate["min_noshow_recall"]:
        failures.append(
            f"no-show recall {metrics['val_noshow_recall']:.4f} < {gate['min_noshow_recall']}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "docs" / "model_results.json")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    gate = load_yaml("slo.yaml")["gate"]
    failures = check_results(results, load_yaml("params.yaml"), gate)

    metrics = results.get("experiments", {}).get(results.get("selected_model"), {})
    print(f"Model:  {results.get('selected_model')}  (from {args.results.name})")
    print(f"PR-AUC: {metrics.get('val_pr_auc')}  (gate >= {gate['min_pr_auc']})")
    print(f"Recall: {metrics.get('val_noshow_recall')}  (gate >= {gate['min_noshow_recall']})")
    if failures:
        print("Model gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Model gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
