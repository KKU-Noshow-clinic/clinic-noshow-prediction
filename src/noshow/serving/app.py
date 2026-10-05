"""FastAPI serving app for the clinic no-show model.

Endpoints
  GET  /health         liveness + which model version is loaded (by the worker that answers)
  POST /predict        score one appointment
  POST /predict_batch  score up to 1000 appointments
  POST /reload         re-read the champion from the MLflow registry (after promote/rollback)
  GET  /metrics        Prometheus metrics (latency, requests, predictions, scores)

Scaling (see docs/serving.md)
  WEB_CONCURRENCY           uvicorn worker processes (Dockerfile default 2)
  PROMETHEUS_MULTIPROC_DIR  set in Docker so /metrics sums all workers
  MICROBATCH_*              concurrent requests share one pipeline call (serving/batching.py)
  SERVING_PROFILE=1         per-stage timing histograms (serving/profiling.py), off by default
"""

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from noshow import __version__
from noshow.serving import profiling
from noshow.serving.batching import MicroBatcher
from noshow.serving.model import ModelBundle, ModelHolder
from noshow.serving.schemas import Appointment, BatchRequest, BatchResponse, Prediction

REQUESTS = Counter("noshow_requests_total", "HTTP requests", ["path", "method", "status"])
LATENCY = Histogram(
    "noshow_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("noshow_predictions_total", "Scored appointments", ["alert"])
SCORES = Histogram(
    "noshow_prediction_score",
    "Distribution of no-show scores (for drift monitoring)",
    buckets=[i / 10 for i in range(1, 10)],
)
# "max" across workers: 1 for every version that at least one worker is serving.
MODEL_INFO = Gauge(
    "noshow_model_info",
    "Currently loaded model",
    ["version", "threshold"],
    multiprocess_mode="max",
)

_info_lock = threading.Lock()
_published: tuple[str, str] | None = None


def publish_model_info(bundle: ModelBundle | None) -> None:
    """Mark the served version 1 and the previous one 0 (clear() does not work across processes)."""
    global _published
    if bundle is None:
        return
    labels = (bundle.version, str(bundle.threshold))
    with _info_lock:
        if labels == _published:
            return
        if _published is not None:
            MODEL_INFO.labels(*_published).set(0)
        MODEL_INFO.labels(*labels).set(1)
        _published = labels


holder = ModelHolder(on_load=publish_model_info)
batcher = MicroBatcher()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("MODEL_LOAD_ON_STARTUP", "1") == "1":
        holder.load_in_background()
    yield


app = FastAPI(title="Clinic No-Show Prediction API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    elapsed = time.perf_counter() - start
    LATENCY.labels(path=path).observe(elapsed)
    if profiling.ENABLED and path in ("/predict", "/predict_batch"):
        profiling.observe(f"in_app_{path}", elapsed)
    REQUESTS.labels(path=path, method=request.method, status=response.status_code).inc()
    return response


def _require_model() -> ModelBundle:
    bundle = holder.get()
    if bundle is None:
        detail = "Model not loaded yet"
        if holder.last_error:
            detail += f" ({holder.last_error})"
        raise HTTPException(status_code=503, detail=detail)
    return bundle


async def _score(appointments: list[Appointment]) -> list[dict]:
    bundle = _require_model()
    results = await batcher.submit(bundle, [a.to_record() for a in appointments])
    for item in results:
        PREDICTIONS.labels(alert=str(item["alert"]).lower()).inc()
        SCORES.observe(item["noshow_score"])
    return results


@app.get("/health")
def health() -> dict:
    bundle = holder.get()
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
        "model_uri": bundle.uri if bundle else None,
        "threshold": bundle.threshold if bundle else None,
        "last_error": holder.last_error,
        "pid": os.getpid(),
        "workers": int(os.getenv("WEB_CONCURRENCY", "1")),
        "profiling": profiling.ENABLED,
    }


@app.post("/predict", response_model=Prediction)
async def predict(appointment: Appointment) -> dict:
    return (await _score([appointment]))[0]


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest) -> dict:
    results = await _score(batch.appointments)
    return {"predictions": results, "n_alerts": sum(r["alert"] for r in results)}


@app.post("/reload")
def reload() -> dict:
    """Reloads THIS worker now; other workers follow within MODEL_POLL_SECONDS."""
    old = holder.get()
    new = holder.reload()
    if new is None or (new is old and holder.last_error):
        raise HTTPException(status_code=503, detail=f"Reload failed: {holder.last_error}")
    return {"previous_version": old.version if old else None, "model_version": new.version}


@app.get("/metrics")
def metrics() -> Response:
    publish_model_info(holder.get())
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
