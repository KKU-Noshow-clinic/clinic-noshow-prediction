"""Run batch monitoring for the clinic no-show model.

Data drift is reported by Evidently and summarized with PSI for a stable,
explicit alert threshold. Concept drift is evaluated only when delayed labels
and the corresponding model scores are supplied in the feedback CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
except ImportError as exc:  # pragma: no cover - depends on the user's venv
    raise SystemExit(
        "Evidently is not installed in this environment. Activate the project's "
        "venv and run: python -m pip install evidently"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "monitoring_policy.json"
TARGET_CANDIDATES = ("No-show", "No_show", "NoShow", "no_show")
SCORE_CANDIDATES = ("noshow_score", "no_show_score", "y_proba", "score")
RAW_FEATURES = (
    "Age",
    "Gender",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Hypertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "Handicap",
    "SMS_received",
    "lead_time_days",
    "appointment_weekday",
    "age_group",
)
DATE_FEATURES = ("ScheduledDay", "AppointmentDay")


def load_policy(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_label(values: pd.Series) -> pd.Series:
    """Convert Kaggle's Yes/No target or numeric 0/1 values to binary labels."""
    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_numeric(values, errors="coerce")
        return result.where(result.isin([0, 1]))
    normalized = values.astype("string").str.strip().str.casefold()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    return normalized.map(mapping)


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select appointment features and derive lead time if the CSV has dates."""
    data = frame.copy()
    if "lead_time_days" not in data and all(c in data for c in DATE_FEATURES):
        scheduled = pd.to_datetime(data["ScheduledDay"], errors="coerce", utc=True)
        appointment = pd.to_datetime(data["AppointmentDay"], errors="coerce", utc=True)
        data["lead_time_days"] = (appointment - scheduled).dt.total_seconds() / 86400
    if "appointment_weekday" not in data and "AppointmentDay" in data:
        appointment = pd.to_datetime(data["AppointmentDay"], errors="coerce", utc=True)
        data["appointment_weekday"] = appointment.dt.dayofweek

    present = [column for column in RAW_FEATURES if column in data.columns]
    if not present:
        raise ValueError(
            "No supported clinic features found. Expected columns such as Age, "
            "Neighbourhood, SMS_received, or ScheduledDay/AppointmentDay."
        )

    features = data[present].copy()
    for column in features.columns:
        if column in {"Neighbourhood", "Gender", "age_group"}:
            features[column] = features[column].astype("string").fillna("<missing>")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using reference quantiles or category bins."""
    if pd.api.types.is_numeric_dtype(expected):
        ref = pd.to_numeric(expected, errors="coerce").dropna().to_numpy(dtype=float)
        cur = pd.to_numeric(actual, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(ref) or not len(cur):
            return 0.0
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        expected_counts = np.histogram(ref, bins=edges)[0].astype(float)
        actual_counts = np.histogram(cur, bins=edges)[0].astype(float)
    else:
        ref_values = expected.astype("string").fillna("<missing>")
        cur_values = actual.astype("string").fillna("<missing>")
        categories = sorted(set(ref_values) | set(cur_values))
        expected_counts = np.array(
            [(ref_values == value).sum() for value in categories], dtype=float
        )
        actual_counts = np.array([(cur_values == value).sum() for value in categories], dtype=float)
    epsilon = 1e-6
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _sms_rates(frame: pd.DataFrame, label_column: str) -> dict[str, float]:
    if "SMS_received" not in frame:
        return {}
    labels = normalize_label(frame[label_column])
    sms = pd.to_numeric(frame["SMS_received"], errors="coerce")
    rates: dict[str, float] = {}
    for group in (0, 1):
        mask = (sms == group) & labels.notna()
        if mask.any():
            rates[str(group)] = float(labels[mask].mean())
    return rates


def _positive_prediction_rate(frame: pd.DataFrame, threshold: float) -> float | None:
    score_column = find_column(frame, SCORE_CANDIDATES)
    if score_column is None:
        return None
    scores = pd.to_numeric(frame[score_column], errors="coerce").dropna()
    if scores.empty:
        return None
    return float((scores >= threshold).mean())


def _performance(feedback: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    target_column = find_column(feedback, TARGET_CANDIDATES)
    score_column = find_column(feedback, SCORE_CANDIDATES)
    if target_column is None or score_column is None:
        raise ValueError(
            "Feedback CSV must include actual No-show labels and prediction scores. "
            "Required columns: No-show and noshow_score."
        )
    labels = normalize_label(feedback[target_column])
    scores = pd.to_numeric(feedback[score_column], errors="coerce")
    valid = labels.notna() & scores.notna()
    y_true = labels[valid].astype(int)
    y_score = scores[valid].astype(float)
    y_pred = (y_score >= float(policy["score_threshold"])).astype(int)
    enough_rows = len(y_true) >= int(policy["min_labeled_rows"])
    auc: float | None = None
    if y_true.nunique() == 2:
        auc = float(roc_auc_score(y_true, y_score))
    precision = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else None
    recall = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None
    accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else None

    return {
        "labeled_rows": int(len(y_true)),
        "enough_labeled_rows": bool(enough_rows),
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "positive_prediction_rate": float(y_pred.mean()) if len(y_pred) else None,
        "sms_no_show_rate": _sms_rates(feedback.loc[valid], target_column),
    }


def run(
    reference_path: Path,
    current_path: Path,
    feedback_path: Path | None,
    report_dir: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    reference_raw = pd.read_csv(reference_path)
    current_raw = pd.read_csv(current_path)
    reference = make_features(reference_raw)
    current = make_features(current_raw)
    columns = [column for column in reference.columns if column in current.columns]
    if not columns:
        raise ValueError("Reference and current CSVs have no common supported features.")
    reference, current = reference[columns], current[columns]
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Reference and current CSVs each need at least two rows.")

    report_dir.mkdir(parents=True, exist_ok=True)
    is_numeric = [column for column in columns if pd.api.types.is_numeric_dtype(reference[column])]
    is_categorical = [column for column in columns if column not in is_numeric]
    definition = DataDefinition(numerical_columns=is_numeric, categorical_columns=is_categorical)
    snapshot = Report([DataDriftPreset(drift_share=float(policy["drift_share_threshold"]))]).run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )
    snapshot.save_html(str(report_dir / "evidently_data_drift.html"))

    psi_by_feature = {
        column: round(_psi(reference[column], current[column]), 6) for column in columns
    }
    drifted = [
        column
        for column, value in psi_by_feature.items()
        if value >= float(policy["feature_psi_threshold"])
    ]
    drift_share = len(drifted) / len(columns)
    data_drift_alert = drift_share >= float(policy["drift_share_threshold"])

    reference_label = find_column(reference_raw, TARGET_CANDIDATES)
    baseline_sms = _sms_rates(reference_raw, reference_label) if reference_label else {}
    performance = None
    concept_drift = False
    performance_alert = False
    sms_effect_delta = None
    label_state = "waiting_for_labeled_feedback"

    if feedback_path is not None and feedback_path.exists():
        feedback = pd.read_csv(feedback_path)
        performance = _performance(feedback, policy)
        label_state = "enough_labels" if performance["enough_labeled_rows"] else "not_enough_labels"
        enough = performance["enough_labeled_rows"]
        auc = performance["roc_auc"]
        baseline_auc = policy.get("baseline_roc_auc")
        if enough and auc is not None:
            performance_alert = auc < float(policy["min_roc_auc"])
            if baseline_auc is not None:
                performance_alert = performance_alert or (
                    float(baseline_auc) - auc >= float(policy["max_roc_auc_drop"])
                )

        current_sms = performance["sms_no_show_rate"]
        comparable_groups = [
            group for group in ("0", "1") if group in baseline_sms and group in current_sms
        ]
        if len(comparable_groups) == 2:
            baseline_effect = baseline_sms["1"] - baseline_sms["0"]
            current_effect = current_sms["1"] - current_sms["0"]
            sms_effect_delta = abs(current_effect - baseline_effect)
            min_group_rows = int(policy["min_sms_group_rows"])
            if "SMS_received" in feedback:
                feedback_sms = pd.to_numeric(feedback["SMS_received"], errors="coerce")
                feedback_labels = normalize_label(
                    feedback[find_column(feedback, TARGET_CANDIDATES)]
                )
                feedback_score_column = find_column(feedback, SCORE_CANDIDATES)
                feedback_scores = pd.to_numeric(feedback[feedback_score_column], errors="coerce")
                group_counts = [
                    int(
                        (
                            (feedback_sms == group)
                            & feedback_labels.notna()
                            & feedback_scores.notna()
                        ).sum()
                    )
                    for group in (0, 1)
                ]
                if enough and min(group_counts) >= min_group_rows:
                    concept_drift = sms_effect_delta >= float(policy["sms_effect_delta_threshold"])

        concept_drift = concept_drift or performance_alert

    current_prediction_rate = _positive_prediction_rate(
        current_raw, float(policy["score_threshold"])
    )
    reference_prediction_rate = _positive_prediction_rate(
        reference_raw, float(policy["score_threshold"])
    )
    prediction_rate_delta = (
        abs(current_prediction_rate - reference_prediction_rate)
        if current_prediction_rate is not None and reference_prediction_rate is not None
        else None
    )
    prediction_drift_alert = prediction_rate_delta is not None and prediction_rate_delta >= float(
        policy["prediction_positive_rate_delta_threshold"]
    )

    if concept_drift:
        action = "RETRAIN"
        reason = (
            "Concept/performance drift confirmed with enough delayed labels. "
            "Run the existing train-evaluate-gate pipeline."
        )
    elif data_drift_alert or prediction_drift_alert:
        action = "WATCH"
        reason = (
            "Input/prediction drift detected. Investigate and collect delayed labels; "
            "do not retrain on unlabeled drift alone."
        )
    elif label_state == "not_enough_labels":
        action = "WAIT_FOR_LABELS"
        reason = "Not enough delayed labels for a reliable performance/concept-drift decision."
    else:
        action = "OK"
        reason = "No alert crossed the configured thresholds."

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_file": str(reference_path),
        "current_file": str(current_path),
        "feedback_file": str(feedback_path) if feedback_path and feedback_path.exists() else None,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_checked": columns,
        "feature_psi": psi_by_feature,
        "drifted_features": drifted,
        "drift_share": round(drift_share, 6),
        "data_drift_alert": data_drift_alert,
        "prediction_positive_rate": current_prediction_rate,
        "prediction_positive_rate_delta": prediction_rate_delta,
        "prediction_drift_alert": bool(prediction_drift_alert),
        "label_state": label_state,
        "performance": performance,
        "sms_effect_delta": sms_effect_delta,
        "concept_drift_suspected": concept_drift,
        "performance_alert": performance_alert,
        "action": action,
        "reason": reason,
        "policy": policy,
    }
    target_path = report_dir / "monitoring_latest.json"
    target_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evidently and label-based monitoring for clinic no-show."
    )
    parser.add_argument(
        "--reference", help="CSV from the model's healthy/training reference period."
    )
    parser.add_argument("--current", help="Current unlabeled appointment feature CSV.")
    parser.add_argument(
        "--feedback", help="Delayed-label CSV with No-show and noshow_score columns."
    )
    parser.add_argument(
        "--policy", default=str(POLICY_PATH), help="Path to monitoring_policy.json."
    )
    parser.add_argument("--report-dir", help="Directory for HTML/JSON monitoring reports.")
    args = parser.parse_args()

    policy = load_policy(resolve_path(args.policy))
    reference_path = resolve_path(args.reference or policy["reference_csv"])
    current_path = resolve_path(args.current or policy["current_csv"])
    default_feedback = resolve_path(policy["feedback_csv"])
    feedback_path = (
        resolve_path(args.feedback)
        if args.feedback
        else (default_feedback if default_feedback.exists() else None)
    )
    report_dir = resolve_path(args.report_dir or policy["report_dir"])
    result = run(reference_path, current_path, feedback_path, report_dir, policy)
    print(f"Action: {result['action']}")
    n_drifted = len(result["drifted_features"])
    n_checked = len(result["features_checked"])
    print(f"Data drift share: {result['drift_share']:.1%} ({n_drifted}/{n_checked} features)")
    print(f"Drifted features: {', '.join(result['drifted_features']) or 'none'}")
    if result["performance"]:
        perf = result["performance"]
        print(f"Delayed-label rows: {perf['labeled_rows']}; ROC-AUC: {perf['roc_auc']}")
    else:
        print("Concept/performance drift: waiting for delayed labels and prediction scores.")
    print(f"Reports: {report_dir}")
    return 10 if result["action"] == "RETRAIN" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Monitoring failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
