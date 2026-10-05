"""Performance features must not change model output: micro-batching and champion polling."""

import asyncio

import pandas as pd
import pytest

from noshow.serving.batching import MicroBatcher
from noshow.serving.model import ModelBundle, ModelHolder
from noshow.serving.schemas import Appointment


def _records(valid, n):
    return [
        Appointment(**{**valid, "PatientId": 1000.0 + i, "Age": 10 + i}).to_record()
        for i in range(n)
    ]


def _run_concurrently(batcher, bundle, requests):
    async def main():
        return await asyncio.gather(*(batcher.submit(bundle, recs) for recs in requests))

    return asyncio.run(main())


def test_batched_scores_equal_one_by_one(bundle, valid):
    records = _records(valid, 40)
    expected = [bundle.predict([r])[0] for r in records]
    batcher = MicroBatcher(max_size=32)
    results = _run_concurrently(batcher, bundle, [[r] for r in records])
    got = [r[0] for r in results]
    assert [g["PatientId"] for g in got] == [e["PatientId"] for e in expected]  # order kept
    assert [g["noshow_score"] for g in got] == pytest.approx(
        [e["noshow_score"] for e in expected], abs=1e-6
    )
    assert max(batcher.batch_sizes) > 1  # concurrent requests really shared a call
    assert sum(batcher.batch_sizes) == 40


def test_batch_request_is_not_split_and_matches_pipeline(bundle, fitted_pipeline, valid):
    records = _records(valid, 50)  # bigger than max_size: still one call
    batcher = MicroBatcher(max_size=8)
    (results,) = _run_concurrently(batcher, bundle, [records])
    local = fitted_pipeline.predict_proba(pd.DataFrame(records))[:, 1]
    assert [r["noshow_score"] for r in results] == pytest.approx(list(local), abs=1e-6)
    assert batcher.batch_sizes == [50]


def test_batching_can_be_turned_off(bundle, valid):
    batcher = MicroBatcher(max_size=1)
    _run_concurrently(batcher, bundle, [[r] for r in _records(valid, 5)])
    assert batcher.batch_sizes == [1, 1, 1, 1, 1]


def test_requests_for_different_model_versions_are_scored_separately(bundle, valid):
    other = ModelBundle(model=bundle.model, threshold=0.0, version="other", uri="memory://other")
    record = _records(valid, 1)[0]

    async def main():
        batcher = MicroBatcher(max_size=32)
        return await asyncio.gather(
            batcher.submit(bundle, [record]), batcher.submit(other, [record])
        )

    first, second = asyncio.run(main())
    assert first[0]["model_version"] == "test" and second[0]["model_version"] == "other"
    assert second[0]["alert"] is True  # threshold 0.0 of the other bundle was used


def test_prediction_error_reaches_the_caller(bundle, valid):
    class Broken:
        def predict_proba(self, _):
            raise RuntimeError("boom")

    broken = ModelBundle(model=Broken(), threshold=0.5, version="x", uri="x")
    with pytest.raises(RuntimeError, match="boom"):
        _run_concurrently(MicroBatcher(), broken, [_records(valid, 1)])


def test_holder_reloads_when_champion_moves(bundle):
    newer = ModelBundle(model=bundle.model, threshold=0.4, version="2", uri=bundle.uri)
    loads = iter([bundle, newer])
    registry = {"version": "test"}
    seen = []
    holder = ModelHolder(
        loader=lambda: next(loads),
        version_of=lambda uri: registry["version"],
        on_load=seen.append,
    )
    assert holder.check_for_update() is True  # first load
    assert holder.check_for_update() is False  # alias unchanged -> nothing to do
    registry["version"] = "2"  # promote/rollback moved the alias
    assert holder.check_for_update() is True
    assert holder.get().version == "2"
    assert [b.version for b in seen] == ["test", "2"]


def test_holder_keeps_model_when_registry_unreachable(bundle):
    def down(uri):
        raise ConnectionError("mlflow down")

    holder = ModelHolder(loader=lambda: bundle, version_of=down)
    holder.reload()
    assert holder.check_for_update() is False
    assert holder.get() is bundle
    assert "mlflow down" in holder.last_error
