"""Optional per-stage timing for the serving path (SERVING_PROFILE=1, off by default).

Off: ModelBundle.predict calls model.predict_proba once, exactly as before.
On:  the same pipeline steps are called one by one (what Pipeline.predict_proba does inside),
     so each stage can be timed. Scores are identical; tests/test_serving_profiling.py and
     scripts/check_serving_parity.py check that.

Stages recorded in the `noshow_stage_seconds` histogram (label `stage`):
  frame                         list of records -> pandas DataFrame (per pipeline call)
  preprocess.features           AppointmentFeatureBuilder.transform (per pipeline call)
  preprocess.columns            ColumnTransformer: impute / one-hot / scale (per pipeline call)
  model                         LightGBM predict_proba (per pipeline call)
  queue_wait                    request waiting for its micro-batch to start (per request)
  in_app_/predict, ..._batch    whole request inside the app, incl. validation (per request)
`noshow_batch_rows` records how many rows each pipeline call scored.
"""

from __future__ import annotations

import os
import time

from prometheus_client import Histogram

ENABLED = os.getenv("SERVING_PROFILE", "0") == "1"

_MS = (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 150, 200, 300, 500, 1000, 2500)
STAGE_BUCKETS = tuple(ms / 1000 for ms in _MS)
STAGE_SECONDS = Histogram(
    "noshow_stage_seconds", "Time per serving stage", ["stage"], buckets=STAGE_BUCKETS
)
BATCH_ROWS = Histogram(
    "noshow_batch_rows", "Rows per pipeline call", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 1000)
)


def flatten_steps(model) -> list[tuple[str, object]]:
    """[(name, step), ...] of a (nested) sklearn Pipeline; the last one is the estimator."""
    from sklearn.pipeline import Pipeline

    if not isinstance(model, Pipeline):
        return [("model", model)]
    steps: list[tuple[str, object]] = []
    for name, step in model.steps:
        if step is None or step == "passthrough":
            continue
        if isinstance(step, Pipeline):
            steps += [(f"{name}.{inner}", s) for inner, s in flatten_steps(step)]
        else:
            steps.append((name, step))
    return steps


def timed_predict_proba(model, frame) -> tuple[object, dict[str, float]]:
    """Same result as model.predict_proba(frame), plus seconds spent in each step."""
    steps = flatten_steps(model)
    timings: dict[str, float] = {}
    data = frame
    for name, step in steps[:-1]:
        start = time.perf_counter()
        data = step.transform(data)
        timings[name] = time.perf_counter() - start
    name, estimator = steps[-1]
    start = time.perf_counter()
    proba = estimator.predict_proba(data)
    timings[name] = time.perf_counter() - start
    return proba, timings


def observe(stage: str, seconds: float) -> None:
    STAGE_SECONDS.labels(stage=stage).observe(seconds)
