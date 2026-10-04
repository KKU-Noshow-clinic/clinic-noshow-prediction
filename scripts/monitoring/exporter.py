"""Expose the latest batch-monitoring JSON as Prometheus metrics (port 9101)."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "monitoring" / "monitoring_latest.json"
DEFAULT_POLICY = ROOT / "configs" / "monitoring_policy.json"


def escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def metric(name: str, value: Any, labels: dict[str, Any] | None = None) -> str:
    if value is None:
        return ""
    label_text = ""
    if labels:
        label_text = "{" + ",".join(f'{key}="{escape_label(val)}"' for key, val in labels.items()) + "}"
    return f"{name}{label_text} {float(value):.8g}"


def render_metrics(report_path: Path, policy_path: Path) -> str:
    lines = [
        "# HELP noshow_monitor_report_available Whether a monitoring report is present.",
        "# TYPE noshow_monitor_report_available gauge",
    ]
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    except (OSError, json.JSONDecodeError):
        policy = {}
        report = None

    for name, key in (
        ("noshow_monitor_drift_share_threshold", "drift_share_threshold"),
        ("noshow_monitor_min_roc_auc_threshold", "min_roc_auc"),
        ("noshow_monitor_sms_effect_delta_threshold", "sms_effect_delta_threshold"),
        ("noshow_monitor_p95_latency_threshold_seconds", "p95_latency_seconds"),
        ("noshow_monitor_error_rate_threshold", "error_rate_threshold"),
        ("noshow_monitor_min_labeled_rows_threshold", "min_labeled_rows"),
    ):
        lines.append(metric(name, policy.get(key)))

    if report is None:
        lines.append("noshow_monitor_report_available 0")
        return "\n".join(line for line in lines if line) + "\n"

    lines.append("noshow_monitor_report_available 1")
    definitions = {
        "noshow_monitor_data_drift_share": report.get("drift_share"),
        "noshow_monitor_concept_drift_suspected": int(bool(report.get("concept_drift_suspected"))),
        "noshow_monitor_performance_alert": int(bool(report.get("performance_alert"))),
        "noshow_monitor_retrain_recommended": int(report.get("action") == "RETRAIN"),
        "noshow_monitor_labeled_rows": (report.get("performance") or {}).get("labeled_rows"),
        "noshow_monitor_roc_auc": (report.get("performance") or {}).get("roc_auc"),
        "noshow_monitor_precision": (report.get("performance") or {}).get("precision"),
        "noshow_monitor_recall": (report.get("performance") or {}).get("recall"),
        "noshow_monitor_accuracy": (report.get("performance") or {}).get("accuracy"),
        "noshow_monitor_prediction_positive_rate": report.get("prediction_positive_rate"),
        "noshow_monitor_prediction_drift_alert": int(bool(report.get("prediction_drift_alert"))),
        "noshow_monitor_sms_effect_delta": report.get("sms_effect_delta"),
    }
    for name, value in definitions.items():
        lines.append(metric(name, value))

    feature_psi = report.get("feature_psi") or {}
    if feature_psi:
        lines.extend([
            "# HELP noshow_monitor_feature_psi Population Stability Index by input feature.",
            "# TYPE noshow_monitor_feature_psi gauge",
        ])
        lines.extend(metric("noshow_monitor_feature_psi", value, {"feature": feature})
                     for feature, value in feature_psi.items())

    timestamp = report.get("generated_at")
    if timestamp:
        from datetime import datetime
        try:
            seconds = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            lines.append(metric("noshow_monitor_last_run_timestamp_seconds", seconds))
        except ValueError:
            pass
    return "\n".join(line for line in lines if line) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/healthz":
                body = b"ok\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            elif self.path == "/metrics":
                body = render_metrics(args.report, args.policy).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            else:
                body = b"not found\n"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Monitoring exporter listening at http://{args.host}:{args.port}/metrics")
    server.serve_forever()


if __name__ == "__main__":
    main()
