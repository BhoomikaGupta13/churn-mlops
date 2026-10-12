"""FastAPI service that serves the champion churn model."""

import csv
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.config import ROOT
from src.features import FEATURES
from src.serving.model_loader import ModelStore
from src.serving.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    ModelInfo,
    Prediction,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("churn.api")

THRESHOLD = 0.5
PREDICTION_LOG = Path(
    os.getenv("PREDICTION_LOG_PATH", ROOT / "data" / "monitoring" / "predictions.csv")
)
LOG_COLUMNS = [
    "timestamp",
    "model_version",
    "churn_probability",
    "churn_prediction",
    *FEATURES,
]

store = ModelStore()
_log_lock = threading.Lock()
_recent: deque[int] = deque(maxlen=200)

# ---------- Prometheus metrics ----------
REQUESTS = Counter(
    "predict_requests_total", "Prediction requests", ["endpoint", "status"]
)
LATENCY = Histogram("predict_latency_seconds", "Prediction latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by outcome", ["outcome"])
CHURN_RATE = Gauge(
    "predicted_churn_rate", "Share of churn predictions over the last 200"
)
MODEL_VERSION = Gauge("model_version_info", "Loaded model version (0 = fallback)")


def _set_version_gauge(version: str) -> None:
    try:
        MODEL_VERSION.set(float(version))
    except ValueError:
        MODEL_VERSION.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _set_version_gauge(store.load().version)
    except Exception:
        log.exception("Startup model load failed; /ready will return 503")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("RELOAD_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _risk_level(p: float) -> str:
    return "low" if p < 0.3 else "medium" if p < 0.6 else "high"


def _log_predictions(rows: list[dict], probs, version: str) -> None:
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _log_lock:
        is_new = not PREDICTION_LOG.exists()
        with open(PREDICTION_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if is_new:
                writer.writeheader()
            for row, p in zip(rows, probs):
                writer.writerow(
                    {
                        "timestamp": ts,
                        "model_version": version,
                        "churn_probability": round(float(p), 4),
                        "churn_prediction": int(p >= THRESHOLD),
                        **row,
                    }
                )


def _predict(customers: list[CustomerFeatures], endpoint: str) -> list[Prediction]:
    loaded = store.current
    if loaded is None:
        REQUESTS.labels(endpoint, "unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        rows = [c.model_dump() for c in customers]
        df = pd.DataFrame(rows)[FEATURES]
        probs = loaded.pipeline.predict_proba(df)[:, 1]
    except Exception:
        REQUESTS.labels(endpoint, "error").inc()
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    LATENCY.labels(endpoint).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint, "ok").inc()

    results = []
    for p in probs:
        churn = bool(p >= THRESHOLD)
        PREDICTIONS.labels("churn" if churn else "stay").inc()
        _recent.append(int(churn))
        results.append(
            Prediction(
                churn_probability=round(float(p), 4),
                churn_prediction=churn,
                risk_level=_risk_level(float(p)),
                model_version=loaded.version,
            )
        )
    CHURN_RATE.set(sum(_recent) / len(_recent))

    try:
        _log_predictions(rows, probs, loaded.version)
    except Exception:
        log.exception("Could not write prediction log (prediction still returned)")
    return results


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if store.current is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": store.current.version}


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures):
    return _predict([customer], "predict")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_predict(req.customers, "predict_batch"))


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    m = store.current
    if m is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_name=m.name,
        model_version=m.version,
        source=m.source,
        run_id=m.run_id,
        loaded_at=m.loaded_at,
    )


@app.post(
    "/model/reload", response_model=ModelInfo, dependencies=[Depends(require_api_key)]
)
def model_reload():
    """Re-fetch the champion from MLflow. A failed reload keeps the current model."""
    try:
        m = store.load(allow_fallback=False)
    except Exception as exc:
        log.exception("Reload failed; keeping current model")
        raise HTTPException(status_code=502, detail=f"Reload failed: {exc}")
    _set_version_gauge(m.version)
    return ModelInfo(
        model_name=m.name,
        model_version=m.version,
        source=m.source,
        run_id=m.run_id,
        loaded_at=m.loaded_at,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
