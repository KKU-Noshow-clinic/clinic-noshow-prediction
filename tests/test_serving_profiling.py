"""Per-stage timing (SERVING_PROFILE=1) must give exactly the same scores as the normal path."""

import asyncio

import numpy as np
import pandas as pd
from prometheus_client import REGISTRY

from noshow.serving import profiling
from noshow.serving.batching import MicroBatcher
from noshow.serving.schemas import Appointment


def _frame(valid, n=50):
    return pd.DataFrame(
        [
            Appointment(**{**valid, "PatientId": 500.0 + i, "Age": i % 90}).to_record()
            for i in range(n)
        ]
    )


def _count(stage):
    return REGISTRY.get_sample_value("noshow_stage_seconds_count", {"stage": stage}) or 0.0


def test_flatten_steps_matches_pipeline_structure(fitted_pipeline):
    names = [name for name, _ in profiling.flatten_steps(fitted_pipeline)]
    assert names == ["preprocess.features", "preprocess.columns", "model"]


def test_step_by_step_equals_predict_proba(fitted_pipeline, valid):
    frame = _frame(valid)
    proba, timings = profiling.timed_predict_proba(fitted_pipeline, frame)
    assert np.array_equal(proba, fitted_pipeline.predict_proba(frame))  # bit-for-bit
    assert set(timings) == {"preprocess.features", "preprocess.columns", "model"}
    assert all(seconds >= 0 for seconds in timings.values())


def test_bundle_scores_identical_with_profiling_on(bundle, valid, monkeypatch):
    records = _frame(valid, 20).to_dict("records")
    off = bundle.predict(records)
    before = _count("preprocess.features")
    monkeypatch.setattr(profiling, "ENABLED", True)
    on = bundle.predict(records)
    assert on == off
    assert _count("preprocess.features") == before + 1
    assert _count("frame") >= 1 and _count("model") >= 1


def test_queue_wait_recorded_per_request(bundle, valid, monkeypatch):
    monkeypatch.setattr(profiling, "ENABLED", True)
    records = _frame(valid, 5).to_dict("records")
    before = _count("queue_wait")

    async def main():
        batcher = MicroBatcher(max_size=32)
        return await asyncio.gather(*(batcher.submit(bundle, [r]) for r in records))

    asyncio.run(main())
    assert _count("queue_wait") == before + 5
