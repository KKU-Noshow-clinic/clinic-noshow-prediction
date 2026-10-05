"""Where does serving time go? HTTP vs feature builder vs ColumnTransformer vs LightGBM.

Two modes (details and the exact commands in docs/serving.md, "Latency breakdown"):

1) offline - run INSIDE the api container, no HTTP, no load. Loads the champion the same way the
   API does and times every stage of one pipeline call, for 1 / 8 / 32 rows:
     docker compose exec -T api python - offline < scripts/profile_serving.py > OUT.md

2) load - run on the host while the API runs with SERVING_PROFILE=1. Snapshots /metrics, runs
   Locust, snapshots again, and reports server-side stage times from the histogram deltas next
   to the latency the client saw. client - in-app (means) = HTTP + network + client overhead.
     uv run python scripts/profile_serving.py load --users 50 --run-time 2m --name prof50

3) http - plain sequential HTTP timing, no Locust. Run it on the host AND inside the container:
   inside, requests never cross Docker Desktop's port forwarding, so the difference between the
   two is the cost of that network path.
     uv run python scripts/profile_serving.py http --rounds 200
     docker compose exec -T api python - http --rounds 200 < scripts/profile_serving.py

Both print Markdown (machine, workers, method, rounds, table) to stdout; `load` also writes
loadtest/results/profile_<name>.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

NEIGHBOURHOODS = ["JARDIM CAMBURI", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM DA PENHA", "CENTRO"]
PIPELINE_STAGES = ["frame", "preprocess.features", "preprocess.columns", "model"]


def _appointment(rng: random.Random) -> dict:
    scheduled = datetime(2016, 4, 25, tzinfo=UTC) + timedelta(
        days=rng.randint(0, 40), hours=rng.randint(7, 17)
    )
    appointment = scheduled.replace(hour=0) + timedelta(days=rng.choice([0, 1, 3, 7, 14, 30]))
    return {
        "PatientId": float(rng.randint(1, 50_000)),
        "Gender": rng.choice(["F", "M"]),
        "ScheduledDay": scheduled.isoformat().replace("+00:00", "Z"),
        "AppointmentDay": appointment.isoformat().replace("+00:00", "Z"),
        "Age": rng.randint(0, 95),
        "Neighbourhood": rng.choice(NEIGHBOURHOODS),
        "Scholarship": int(rng.random() < 0.1),
        "Hipertension": int(rng.random() < 0.2),
        "Diabetes": int(rng.random() < 0.07),
        "Alcoholism": int(rng.random() < 0.03),
        "Handcap": int(rng.random() < 0.02),
        "SMS_received": int(rng.random() < 0.32),
    }


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _cpu_quota() -> str:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return "no limit" if quota == "max" else f"{int(quota) / int(period):.1f} CPUs"
    except (OSError, ValueError):
        return "unknown"


# --------------------------------------------------------------------------- offline (container)
def offline(args: argparse.Namespace) -> None:
    import lightgbm
    import pandas as pd
    import sklearn

    from noshow.serving import profiling
    from noshow.serving.model import load_bundle
    from noshow.serving.schemas import Appointment

    bundle = load_bundle()
    rng = random.Random(0)
    lines = [
        "## Offline stage breakdown (inside the api container, no HTTP, no load)",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Model: `{bundle.uri}` v{bundle.version}",
        f"- Container: {platform.platform()}, Python {platform.python_version()}, "
        f"{os.cpu_count()} CPUs visible, cgroup quota {_cpu_quota()}",
        f"- OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'unset')}, "
        f"WEB_CONCURRENCY={os.getenv('WEB_CONCURRENCY', 'unset')} "
        "(workers keep running idle while this measures)",
        f"- pandas {pd.__version__}, scikit-learn {sklearn.__version__}, "
        f"lightgbm {lightgbm.__version__}",
        f"- Method: single process, sequential calls, {args.warmup} warm-up calls discarded, "
        f"then {args.rounds} timed calls per batch size; time.perf_counter per stage. "
        "`total` = the API's normal path (one predict_proba) timed separately.",
        "",
        "| Rows/call | Stage | p50 (ms) | p95 (ms) | mean (ms) | share of stages |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rows in args.batch_sizes:
        payloads = [[_appointment(rng) for _ in range(rows)] for _ in range(args.rounds)]
        samples: dict[str, list[float]] = {s: [] for s in ["schema", *PIPELINE_STAGES, "total"]}
        for i in range(args.warmup + args.rounds):
            payload = payloads[i % args.rounds]
            start = time.perf_counter()
            records = [Appointment(**p).to_record() for p in payload]
            schema = time.perf_counter() - start
            start = time.perf_counter()
            frame = pd.DataFrame.from_records(records)
            frame_s = time.perf_counter() - start
            proba, timings = profiling.timed_predict_proba(bundle.model, frame)
            start = time.perf_counter()
            expected = bundle.model.predict_proba(pd.DataFrame.from_records(records))
            total = time.perf_counter() - start
            if not (proba == expected).all():
                sys.exit("Step-by-step scores differ from predict_proba - stop and investigate")
            if i < args.warmup:
                continue
            samples["schema"].append(schema)
            samples["frame"].append(frame_s)
            for stage, seconds in timings.items():
                samples[stage].append(seconds)
            samples["total"].append(total)
        stage_sum = sum(statistics.mean(samples[s]) for s in PIPELINE_STAGES)
        for stage, values in samples.items():
            share = f"{statistics.mean(values) / stage_sum:.0%}" if stage in PIPELINE_STAGES else ""
            lines.append(
                f"| {rows} | {stage} | {_pct(values, 0.5) * 1e3:.1f} | "
                f"{_pct(values, 0.95) * 1e3:.1f} | {statistics.mean(values) * 1e3:.1f} | {share} |"
            )
    lines += ["", "Step-by-step scores were identical to predict_proba on every call.", ""]
    print("\n".join(lines))


# --------------------------------------------------------------------------- http (both sides)
def http_timing(args: argparse.Namespace) -> None:
    import http.client
    from urllib.parse import urlparse

    url = urlparse(args.host)
    rng = random.Random(1)
    inside = Path("/.dockerenv").exists()
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)
    headers = {"Content-Type": "application/json"}
    samples: dict[str, list[float]] = {"GET /health": [], "POST /predict": []}
    for i in range(args.warmup + args.rounds):
        for name in samples:
            body = json.dumps(_appointment(rng)).encode() if name.startswith("POST") else None
            start = time.perf_counter()
            conn.request(name.split()[0], name.split()[1], body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()
            elapsed = time.perf_counter() - start
            if resp.status != 200:
                sys.exit(f"{name} returned {resp.status}")
            if i >= args.warmup:
                samples[name].append(elapsed)
    conn.close()
    local = url.hostname in ("localhost", "127.0.0.1")
    if inside and local:
        where = "inside the api container itself (loopback, no Docker networking)"
    elif inside:
        where = "another container on the Docker network (no Windows port forwarding)"
    else:
        where = "the host (through Docker Desktop port forwarding)"
    lines = [
        f"## Sequential HTTP timing from {where}",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Client: {platform.platform()}, Python {platform.python_version()}, "
        f"{os.cpu_count()} CPUs; target {args.host}",
        f"- Method: one keep-alive connection, Python http.client, one request at a time, "
        f"{args.warmup} warm-up rounds discarded, {args.rounds} timed rounds of each request",
        "",
        "| Request | p50 (ms) | p95 (ms) | mean (ms) |",
        "|---|---:|---:|---:|",
    ]
    for name, values in samples.items():
        lines.append(
            f"| {name} | {_pct(values, 0.5) * 1e3:.1f} | {_pct(values, 0.95) * 1e3:.1f} | "
            f"{statistics.mean(values) * 1e3:.1f} |"
        )
    print("\n".join(lines + [""]))


# --------------------------------------------------------------------------- load (host)
def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode()


def _histograms(metrics_text: str, name: str) -> dict[str, dict[float, float]]:
    """{stage: {le: cumulative count}} for one histogram from Prometheus text format."""
    out: dict[str, dict[float, float]] = {}
    for line in metrics_text.splitlines():
        if not line.startswith(f"{name}_bucket{{"):
            continue
        labels, value = line[len(name) + 8 :].rsplit("} ", 1)
        parts = dict(p.split("=", 1) for p in labels.split(","))
        stage = parts.get("stage", '"rows"').strip('"')
        le = float(parts["le"].strip('"').replace("+Inf", "inf"))
        out.setdefault(stage, {})[le] = float(value)
    return out


def _sums(metrics_text: str, name: str) -> dict[str, float]:
    out = {}
    for line in metrics_text.splitlines():
        if line.startswith(f"{name}_sum"):
            labels, value = line.rsplit(" ", 1)
            stage = labels.split('stage="')[1].split('"')[0] if "stage=" in labels else "rows"
            out[stage] = float(value)
    return out


def _quantile(buckets: dict[float, float], q: float) -> float:
    """Approximate quantile from cumulative buckets (linear inside the bucket)."""
    total = buckets.get(float("inf"), 0.0)
    if total <= 0:
        return float("nan")
    target, prev_le, prev_count = q * total, 0.0, 0.0
    for le in sorted(buckets):
        count = buckets[le]
        if count >= target:
            if le == float("inf"):
                return prev_le
            span = count - prev_count
            return prev_le + (le - prev_le) * ((target - prev_count) / span if span else 0)
        prev_le, prev_count = le, count
    return prev_le


def _delta(after: dict, before: dict) -> dict:
    return {
        k: {le: c - before.get(k, {}).get(le, 0.0) for le, c in v.items()} for k, v in after.items()
    }


def _machine() -> list[str]:
    try:
        docker = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "{{.NCPU}} CPUs, {{.MemTotal}} bytes RAM, {{.OperatingSystem}}, "
                "kernel {{.KernelVersion}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        docker = "unknown"
    return [
        f"- Host: {platform.platform()}, {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical CPUs",
        f"- Docker engine: {docker or 'unknown'}",
    ]


def load(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "loadtest" / "results"
    results.mkdir(parents=True, exist_ok=True)
    health = json.loads(_get(f"{args.host}/health"))
    if not health.get("model_loaded"):
        sys.exit(f"API has no model loaded: {health}")
    if not health.get("profiling"):
        sys.exit("API runs without SERVING_PROFILE=1 - set API_PROFILE=1 and recreate the api")

    prefix = results / args.name
    for attempt in (1, 2):
        before = _get(f"{args.host}/metrics")
        options = {
            "-f": str(root / "loadtest" / "locustfile.py"),
            "--host": args.host,
            "--users": str(args.users),
            "--spawn-rate": str(args.spawn_rate),
            "--run-time": args.run_time,
            "--csv": str(prefix),
        }
        cmd = [sys.executable, "-m", "locust", "--headless"]
        for flag, value in options.items():
            cmd += [flag, value]
        with open(f"{prefix}_locust.log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        after = _get(f"{args.host}/metrics")
        with open(f"{prefix}_stats.csv", encoding="utf-8") as f:
            rows = {f"{r['Type']} {r['Name']}".strip(): r for r in csv.DictReader(f)}
        if "POST /predict" in rows and int(rows["POST /predict"]["Request Count"]) > 0:
            break
        print(f"Locust sent no requests (attempt {attempt}), see {prefix}_locust.log", flush=True)
    else:
        sys.exit("Locust sent no requests twice - check the log")

    client = rows["POST /predict"]
    stages = _delta(
        _histograms(after, "noshow_stage_seconds"), _histograms(before, "noshow_stage_seconds")
    )
    sums_after = _sums(after, "noshow_stage_seconds")
    sums_before = _sums(before, "noshow_stage_seconds")
    rows_hist = "noshow_batch_rows"
    batch = _delta(_histograms(after, rows_hist), _histograms(before, rows_hist))
    batch_sum = _sums(after, rows_hist).get("rows", 0) - _sums(before, rows_hist).get("rows", 0)

    def stat(stage: str) -> tuple[float, float, float, float]:
        b = stages.get(stage, {})
        n = b.get(float("inf"), 0.0)
        mean = (sums_after.get(stage, 0) - sums_before.get(stage, 0)) / n if n else float("nan")
        return n, _quantile(b, 0.5) * 1e3, _quantile(b, 0.95) * 1e3, mean * 1e3

    calls = batch.get("rows", {}).get(float("inf"), 0.0)
    lines = [
        f"## Under load: {args.users} users ({args.name})",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        *_machine(),
        f"- API: {health['workers']} uvicorn worker(s), model v{health['model_version']}, "
        "micro-batching as configured (API_BATCH), SERVING_PROFILE=1",
        f"- Method: Locust {args.users} users, spawn {args.spawn_rate}/s, {args.run_time}, "
        "traffic mix /predict 10 : /predict_batch (20 rows) 1 : /health 1. Server stages from "
        "Prometheus histogram deltas before/after the run (percentiles interpolated inside "
        "buckets, so approximate); client numbers from Locust.",
        f"- Requests: {client['Request Count']} /predict, {client['Failure Count']} failed; "
        f"{calls:.0f} pipeline calls, {batch_sum / calls if calls else float('nan'):.1f} rows "
        "per call on average",
        "",
        "| Where | Count | p50 (ms) | p95 (ms) | mean (ms) |",
        "|---|---:|---:|---:|---:|",
        f"| **client sees** /predict (Locust) | {client['Request Count']} | {client['50%']} | "
        f"{client['95%']} | {float(client['Average Response Time']):.1f} |",
    ]
    order = ["in_app_/predict", "queue_wait", *PIPELINE_STAGES, "in_app_/predict_batch"]
    labels = {
        "in_app_/predict": "inside app, whole /predict request",
        "queue_wait": "waiting for micro-batch (per request)",
        "frame": "build DataFrame (per call)",
        "preprocess.features": "feature builder (per call)",
        "preprocess.columns": "ColumnTransformer (per call)",
        "model": "LightGBM predict_proba (per call)",
        "in_app_/predict_batch": "inside app, whole /predict_batch request",
    }
    for stage in order:
        n, p50, p95, mean = stat(stage)
        if n:
            lines.append(f"| {labels[stage]} | {n:.0f} | {p50:.1f} | {p95:.1f} | {mean:.1f} |")
    _, _, _, in_app_mean = stat("in_app_/predict")
    http = float(client["Average Response Time"]) - in_app_mean
    lines += [
        "",
        f"HTTP + network + client overhead (mean client - mean in-app) ~ **{http:.1f} ms**. "
        "Per-call stages are shared by all rows in a micro-batch, so they do not add up to "
        "per-request latency under load.",
        "",
    ]
    text = "\n".join(lines)
    Path(f"{prefix}.md").with_name(f"profile_{args.name}.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    off = sub.add_parser("offline", help="inside the container, no HTTP")
    off.add_argument("--rounds", type=int, default=300)
    off.add_argument("--warmup", type=int, default=30)
    off.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ld = sub.add_parser("load", help="on the host, Locust + /metrics deltas")
    ld.add_argument("--users", type=int, default=50)
    ld.add_argument("--spawn-rate", type=int, default=10)
    ld.add_argument("--run-time", default="2m")
    ld.add_argument("--name", default="prof")
    ld.add_argument("--host", default="http://localhost:8000")
    ht = sub.add_parser("http", help="sequential HTTP timing (run on host and in container)")
    ht.add_argument("--rounds", type=int, default=200)
    ht.add_argument("--warmup", type=int, default=20)
    ht.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()
    if args.mode == "offline":
        offline(args)
    elif args.mode == "http":
        http_timing(args)
    else:
        load(args)


if __name__ == "__main__":
    main()
