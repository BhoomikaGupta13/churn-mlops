"""Loads the champion model from the MLflow registry, with a local-file fallback."""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import ROOT, load_params

# Fail fast if MLflow is down (default retry/backoff can hang for minutes).
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

log = logging.getLogger("churn.model")


@dataclass
class LoadedModel:
    pipeline: object
    name: str
    version: str
    source: str  # "mlflow:champion" or "fallback:local-file"
    run_id: str | None
    loaded_at: str


class ModelStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def load(self, allow_fallback: bool = True) -> LoadedModel:
        params = load_params()
        name = params["mlflow"]["registered_model_name"]
        uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            mlflow.set_tracking_uri(uri)
            mv = MlflowClient(tracking_uri=uri).get_model_version_by_alias(
                name, "champion"
            )
            # Load by explicit version so the alias can't change under us mid-load.
            pipeline = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
            loaded = LoadedModel(
                pipeline, name, str(mv.version), "mlflow:champion", mv.run_id, now
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            log.warning(
                "Could not load champion from MLflow (%s). Trying local fallback.", exc
            )
            fallback = ROOT / "models" / "model.joblib"
            if not fallback.exists():
                raise RuntimeError(
                    "No champion in MLflow and no local fallback model"
                ) from exc
            loaded = LoadedModel(
                joblib.load(fallback),
                name,
                "fallback",
                "fallback:local-file",
                None,
                now,
            )

        with self._lock:
            self._current = loaded
        log.info(
            "Loaded model %s v%s from %s", loaded.name, loaded.version, loaded.source
        )
        return loaded
