import json
from pathlib import Path

import pytest

from noshow.ci.model_gate import ROOT, check_results, load_yaml, main

BAD_RESULTS = Path(__file__).parent / "fixtures" / "bad_model_results.json"
PARAMS = load_yaml("params.yaml")
GATE = load_yaml("slo.yaml")["gate"]


def committed_results() -> dict:
    return json.loads((ROOT / "docs" / "model_results.json").read_text(encoding="utf-8"))


def selected_metrics(results: dict) -> dict:
    return results["experiments"][results["selected_model"]]


def test_committed_model_passes_the_gate():
    assert check_results(committed_results(), PARAMS, GATE) == []


@pytest.mark.parametrize(
    "metric, value, message",
    [("val_pr_auc", 0.25, "PR-AUC"), ("val_noshow_recall", 0.59, "recall")],
)
def test_each_threshold_blocks_a_weak_model(metric, value, message):
    results = committed_results()
    selected_metrics(results)[metric] = value
    failures = check_results(results, PARAMS, GATE)
    assert len(failures) == 1
    assert message in failures[0]


def test_results_must_belong_to_the_selected_model_and_data():
    results = committed_results()
    results["selected_model"] = "exp1_logistic_regression"
    results["data_sha256"] = "other-dataset"
    failures = check_results(results, PARAMS, GATE)
    assert any("params.yaml" in failure for failure in failures)
    assert any("data_sha256" in failure for failure in failures)


def test_cli_exit_codes(capsys):
    assert main([]) == 0
    assert main(["--results", str(BAD_RESULTS)]) == 1
    assert "Model gate FAILED" in capsys.readouterr().out
