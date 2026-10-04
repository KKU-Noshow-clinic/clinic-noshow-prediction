import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score


def choose_threshold(y_true, probabilities, min_recall: float = 0.60) -> float:
    """Choose the highest-precision threshold whose recall meets the gate."""
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    feasible = []
    for threshold in candidates:
        prediction = np.asarray(probabilities) >= threshold
        recall = recall_score(y_true, prediction, zero_division=0)
        if recall >= min_recall:
            precision = precision_score(y_true, prediction, zero_division=0)
            feasible.append((precision, threshold))
    return float(max(feasible, default=(0.0, 0.5))[1])


def classification_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    prediction = np.asarray(probabilities) >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "noshow_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "noshow_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
    }


def validate_selected_model(
    results: dict[str, dict[str, float]],
    selected_name: str,
    min_recall: float,
    min_pr_auc: float,
) -> dict[str, float]:
    """Check that the team's selected experiment passes validation gates."""
    if selected_name not in results:
        raise ValueError(f"Selected experiment {selected_name!r} was not run")
    metrics = results[selected_name]
    if metrics["val_noshow_recall"] < min_recall:
        raise ValueError(f"{selected_name} did not meet validation recall >= {min_recall}")
    if metrics["val_pr_auc"] < min_pr_auc:
        raise ValueError(f"{selected_name} did not meet validation PR-AUC >= {min_pr_auc}")
    return metrics
