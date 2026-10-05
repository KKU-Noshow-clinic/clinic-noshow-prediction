"""Load the serving model (sklearn Pipeline) and its decision threshold.

Source of truth is the MLflow registry written by the registry workstream:
  models:/clinic-noshow@champion   + model-version tag `threshold`

Environment variables
  MODEL_URI            default models:/<registry.model_name>@champion
  MLFLOW_TRACKING_URI  e.g. http://mlflow:5000 inside docker compose
  MODEL_PATH           optional local MLflow model directory (offline dev / tests);
                       overrides MODEL_URI when set
  MODEL_THRESHOLD      optional override; otherwise registry tag -> params.yaml
  MODEL_POLL_SECONDS   how often each worker checks whether the champion alias moved (30)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from noshow.serving import profiling

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = ROOT / "configs" / "params.yaml"
RETRY_SECONDS = 15.0
POLL_SECONDS = float(os.getenv("MODEL_POLL_SECONDS", "30"))


def _params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))


def default_model_uri() -> str:
    return f"models:/{_params()['registry']['model_name']}@champion"


def default_threshold() -> float:
    return float(_params()["model"]["selected_threshold"])


@dataclass(frozen=True)
class ModelBundle:
    model: Any  # fitted sklearn Pipeline: features -> columns -> classifier
    threshold: float
    version: str
    uri: str

    def predict(self, records: list[dict]) -> list[dict]:
        if profiling.ENABLED:
            start = time.perf_counter()
            frame = pd.DataFrame.from_records(records)
            profiling.observe("frame", time.perf_counter() - start)
            proba, timings = profiling.timed_predict_proba(self.model, frame)
            for stage, seconds in timings.items():
                profiling.observe(stage, seconds)
            profiling.BATCH_ROWS.observe(len(records))
            scores = proba[:, 1]
        else:
            frame = pd.DataFrame.from_records(records)
            scores = self.model.predict_proba(frame)[:, 1]
        return [
            {
                "PatientId": record["PatientId"],
                "noshow_score": round(float(score), 6),
                "alert": bool(score >= self.threshold),
                "threshold": self.threshold,
                "model_version": self.version,
            }
            for record, score in zip(records, scores, strict=True)
        ]


def _registry_info(uri: str) -> tuple[str, float | None]:
    """Return (version, threshold tag) for a models:/name@alias or models:/name/version URI."""
    from mlflow import MlflowClient

    ref = uri.removeprefix("models:/")
    client = MlflowClient()
    if "@" in ref:
        name, alias = ref.split("@", 1)
        mv = client.get_model_version_by_alias(name, alias)
    else:
        name, version = ref.rsplit("/", 1)
        mv = client.get_model_version(name, version)
    tag = mv.tags.get("threshold")
    return str(mv.version), (float(tag) if tag is not None else None)


def load_bundle() -> ModelBundle:
    import mlflow.sklearn

    local_path = os.getenv("MODEL_PATH")
    uri = local_path or os.getenv("MODEL_URI") or default_model_uri()

    model = mlflow.sklearn.load_model(uri)

    version, tag_threshold = "local", None
    if uri.startswith("models:/"):
        version, tag_threshold = _registry_info(uri)

    env_threshold = os.getenv("MODEL_THRESHOLD")
    if env_threshold is not None:
        threshold = float(env_threshold)
    elif tag_threshold is not None:
        threshold = tag_threshold
    else:
        threshold = default_threshold()
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    log.info("Loaded model %s (version %s, threshold %.3f)", uri, version, threshold)
    return ModelBundle(model=model, threshold=threshold, version=version, uri=uri)


def champion_version(uri: str) -> str | None:
    """Current registry version behind an alias URI (models:/name@alias); None otherwise."""
    if not uri.startswith("models:/") or "@" not in uri:
        return None
    return _registry_info(uri)[0]


class ModelHolder:
    """Holds the current model for one worker process.

    Loading never happens inside a request (it would blow the latency SLO). A background thread
    first retries until MLflow is up and a champion exists, so `docker compose up` works even
    before the first model is registered. It then keeps polling the registry: when the champion
    alias moves (promote or rollback) this worker reloads by itself. With several uvicorn
    workers that matters, because POST /reload only reaches the one worker that received it.
    """

    def __init__(self, loader=load_bundle, version_of=champion_version, on_load=None) -> None:
        self._loader = loader
        self._version_of = version_of
        self._on_load = on_load
        self._lock = threading.Lock()
        self.bundle: ModelBundle | None = None
        self.last_error: str | None = None

    def get(self) -> ModelBundle | None:
        return self.bundle

    def check_for_update(self) -> bool:
        """Reload if the registry alias now points at another version. True if reloaded."""
        bundle = self.bundle
        if bundle is None:
            return self.reload() is not None
        try:
            latest = self._version_of(bundle.uri)
        except Exception as exc:  # registry unreachable: keep serving the current model
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        if latest is None or latest == bundle.version:
            return False
        log.info("Champion moved %s -> %s, reloading", bundle.version, latest)
        return self.reload() is not bundle

    def load_in_background(
        self, retry_seconds: float = RETRY_SECONDS, poll_seconds: float = POLL_SECONDS
    ) -> threading.Thread:
        def _run() -> None:
            while True:
                self.check_for_update()
                time.sleep(retry_seconds if self.bundle is None else poll_seconds)

        thread = threading.Thread(target=_run, name="model-loader", daemon=True)
        thread.start()
        return thread

    def reload(self) -> ModelBundle | None:
        with self._lock:
            try:
                self.bundle = self._loader()
                self.last_error = None
                if self._on_load is not None:
                    self._on_load(self.bundle)
            except Exception as exc:  # keep serving the old model if reload fails
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Model load failed: %s", self.last_error)
        return self.bundle

    def set(self, bundle: ModelBundle | None) -> None:
        self.bundle = bundle
        self.last_error = None
