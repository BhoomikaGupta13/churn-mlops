"""Train the churn model, log everything to MLflow, register it, and write metrics.json."""

import json
import os
import subprocess

import joblib
import matplotlib
import importlib.metadata as md

import pandas as pd

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import ROOT, load_params
from src.features import build_pipeline
from src.models.evaluate import compute_metrics, load_xy
from src.models.registry import get_alias_version, get_client, resolve_tracking_uri


def build_model(params: dict):
    cfg = params["train"]
    seed = params["data"]["random_state"]
    if cfg["model_type"] == "random_forest":
        return RandomForestClassifier(
            **cfg["random_forest"], random_state=seed, n_jobs=-1
        )
    if cfg["model_type"] == "logistic_regression":
        return LogisticRegression(**cfg["logistic_regression"], random_state=seed)
    raise ValueError(f"Unknown model_type: {cfg['model_type']}")


def get_git_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_data_version(params: dict) -> str:
    """The md5 DVC recorded for the raw data = exact data version."""
    dvc_file = ROOT / (params["data"]["raw_path"] + ".dvc")
    try:
        return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    except Exception:
        return "unknown"


def main() -> None:
    params = load_params()
    processed = ROOT / params["data"]["processed_dir"]
    X_train, y_train = load_xy(processed / "train.csv")
    X_test, y_test = load_xy(processed / "test.csv")

    tracking_uri = resolve_tracking_uri(params)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    name = params["mlflow"]["registered_model_name"]

    model_type = params["train"]["model_type"]
    pipeline = build_pipeline(build_model(params))

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params["train"][model_type])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.set_tags(
            {"git_sha": get_git_sha(), "data_version": get_data_version(params)}
        )
        # Log the datasets so the MLflow UI shows lineage in the "Dataset" column
        train_ds = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1),
            source=str(processed / "train.csv"),
            name="train",
            targets="Churn",
        )
        test_ds = mlflow.data.from_pandas(
            pd.concat([X_test, y_test], axis=1),
            source=str(processed / "test.csv"),
            name="test",
            targets="Churn",
        )
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(test_ds, context="evaluation")

        pipeline.fit(X_train, y_train)
        metrics = compute_metrics(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        mlflow.log_artifact(str(ROOT / "params.yaml"))

        signature = infer_signature(X_train, pipeline.predict(X_train))
        pip_reqs = [
            f"{p}=={md.version(p)}" for p in ("scikit-learn", "pandas", "numpy")
        ]
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=name,
            pip_requirements=pip_reqs,
        )
        run_id = run.info.run_id

    # --- registry aliases ---
    client = get_client(tracking_uri)
    version = client.search_model_versions(f"run_id='{run_id}'")[0].version
    client.set_registered_model_alias(name, "challenger", version)
    if get_alias_version(client, name, "champion") is None:
        client.set_registered_model_alias(name, "champion", version)
        print(f"No champion existed -> version {version} is now champion (bootstrap)")

    # --- outputs DVC tracks ---
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(
        pipeline, ROOT / "models" / "model.joblib"
    )  # fallback copy for the API image

    print(f"Registered '{name}' version {version} as challenger | metrics: {metrics}")


if __name__ == "__main__":
    main()
