import numpy as np
import pytest

from noshow.models.evaluate import (
    choose_threshold,
    classification_metrics,
    validate_selected_model,
)


def test_threshold_meets_recall_constraint():
    y_true = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.7, 0.1])
    threshold = choose_threshold(y_true, probability, min_recall=1.0)
    metrics = classification_metrics(y_true, probability, threshold)
    assert threshold == 0.6
    assert metrics["noshow_recall"] == 1.0


def test_team_selected_model_must_pass_validation_gates():
    results = {
        "exp2_lightgbm_balanced": {"val_noshow_recall": 0.80, "val_pr_auc": 0.34},
        "exp3_lightgbm_tuned": {"val_noshow_recall": 0.60, "val_pr_auc": 0.35},
    }
    assert (
        validate_selected_model(results, "exp2_lightgbm_balanced", 0.60, 0.30)
        is results["exp2_lightgbm_balanced"]
    )
    with pytest.raises(ValueError, match="PR-AUC"):
        validate_selected_model(results, "exp2_lightgbm_balanced", 0.60, 0.36)
