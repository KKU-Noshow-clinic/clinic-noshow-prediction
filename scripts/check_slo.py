"""Compare Locust runs against configs/slo.yaml and write docs/loadtest_report.md.

    uv run python scripts/check_slo.py loadtest/results/run [--label "2 workers"]
        [--compare "1 worker (baseline)=loadtest/results/baseline" ...] [--strict]

Reads <prefix>_stats.csv produced by `locust --csv <prefix>`. Each --compare adds an earlier
run to the before/after table. Anything written below the NOTES_MARKER line in the existing
report (analysis, SLO proposal) is kept when the report is regenerated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTES_MARKER = "<!-- notes: everything below is kept when this report is regenerated -->"


def load_rows(prefix: str) -> dict[str, dict]:
    path = Path(f"{prefix}_stats.csv")
    with path.open(encoding="utf-8") as f:
        rows = {f"{r['Type']} {r['Name']}".strip(): r for r in csv.DictReader(f)}
    if "POST /predict" not in rows or int(rows["Aggregated"]["Request Count"]) == 0:
        sys.exit(f"No /predict requests in {path} - did locust run?")
    return rows


def summarize(row: dict) -> dict:
    total = int(row["Request Count"])
    failures = int(row["Failure Count"])
    return {
        "requests": total,
        "rps": float(row["Requests/s"]),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
        "error_rate": failures / total if total else 0.0,
    }


def _fmt(value: float, unit: str) -> str:
    return f"{value:.0f} {unit}" if unit else f"{value:.2%}"


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def comparison(runs: list[tuple[str, dict[str, dict]]], slo: dict) -> list[str]:
    lines = [
        "## Before / after optimization",
        "",
        "| Configuration | /predict p50 | /predict p95 | /predict p99 | Total req/s | Errors "
        f"| p50 <= {slo['latency_p50_ms']} | p95 <= {slo['latency_p95_ms']} |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for label, rows in runs:
        p = summarize(rows["POST /predict"])
        total = summarize(rows["Aggregated"])
        lines.append(
            f"| {label} | {p['p50_ms']:.0f} ms | {p['p95_ms']:.0f} ms | {p['p99_ms']:.0f} ms "
            f"| {total['rps']:.1f} | {total['error_rate']:.2%} "
            f"| {_mark(p['p50_ms'] <= slo['latency_p50_ms'])} "
            f"| {_mark(p['p95_ms'] <= slo['latency_p95_ms'])} |"
        )
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="the --csv prefix given to locust (current run)")
    parser.add_argument("--label", default="current", help="name of the current configuration")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=PREFIX",
        help="earlier run to show in the before/after table (repeatable)",
    )
    parser.add_argument("--users", default="50")
    parser.add_argument("--duration", default="2m")
    parser.add_argument("--out", default=str(ROOT / "docs" / "loadtest_report.md"))
    parser.add_argument("--strict", action="store_true", help="exit 1 if an SLO is missed")
    args = parser.parse_args()

    slo = yaml.safe_load((ROOT / "configs" / "slo.yaml").read_text(encoding="utf-8"))
    rows = load_rows(args.prefix)
    predict = summarize(rows["POST /predict"])
    overall = summarize(rows["Aggregated"])

    checks = [
        ("p50 latency /predict", predict["p50_ms"], slo["latency_p50_ms"], "ms", "<="),
        ("p95 latency /predict", predict["p95_ms"], slo["latency_p95_ms"], "ms", "<="),
        ("error rate (all requests)", overall["error_rate"], slo["error_rate_max"], "", "<="),
        ("availability", 1 - overall["error_rate"], slo["availability_min"], "", ">="),
    ]

    lines = [
        "# Load test report (Serving)",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Configuration: {args.label}",
        f"- Tool: Locust, {args.users} concurrent users, duration {args.duration}",
        "- Traffic mix: /predict 10 : /predict_batch (20 rows) 1 : /health 1",
        f"- Raw results: `{args.prefix}_stats.csv`",
        "",
        f"## SLO check: {args.label} (targets from `configs/slo.yaml`)",
        "",
        "| SLO | Measured | Target | Result |",
        "|---|---:|---:|:---:|",
    ]
    all_ok = True
    for name, value, target, unit, op in checks:
        ok = value <= target if op == "<=" else value >= target
        all_ok &= ok
        shown, goal = _fmt(value, unit), _fmt(target, unit)
        lines.append(f"| {name} | {shown} | {op} {goal} | {_mark(ok)} |")

    lines += [
        "",
        f"**Overall: {'all SLOs met' if all_ok else 'SLO missed - see FAIL rows'}**",
        "",
    ]

    if args.compare:
        runs = []
        for spec in args.compare:
            label, _, prefix = spec.partition("=")
            if not prefix:
                sys.exit(f"--compare needs LABEL=PREFIX, got {spec!r}")
            runs.append((label.strip(), load_rows(prefix.strip())))
        runs.append((args.label, rows))
        lines += comparison(runs, slo)

    lines += [
        f"## Per-endpoint latency: {args.label}",
        "",
        "| Endpoint | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in rows.items():
        s = summarize(row)
        lines.append(
            f"| {key} | {s['requests']} | {s['rps']:.1f} | {s['p50_ms']:.0f} | "
            f"{s['p95_ms']:.0f} | {s['p99_ms']:.0f} | {s['error_rate']:.2%} |"
        )

    out = Path(args.out)
    notes = ""
    if out.exists() and NOTES_MARKER in (old := out.read_text(encoding="utf-8")):
        notes = old.split(NOTES_MARKER, 1)[1].strip("\n")
    lines += ["", NOTES_MARKER, "", notes or "## Analysis\n\n(write the findings here)", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if (args.strict and not all_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
