import csv

import pytest
from fastapi.testclient import TestClient

from src.features import FEATURES
from src.serving import app as app_module
from src.serving.model_loader import LoadedModel
from src.serving.schemas import EXAMPLE


@pytest.fixture()
def client(trained_pipeline, tmp_path, monkeypatch):
    """API with a small fake-data model loaded and its prediction log sent to a temp file."""
    monkeypatch.setattr(app_module, "PREDICTION_LOG", tmp_path / "predictions.csv")
    app_module.store._current = LoadedModel(
        pipeline=trained_pipeline,
        name="churn-classifier",
        version="7",
        source="test",
        run_id=None,
        loaded_at="now",
    )
    # No "with" block on purpose: it would run the startup hook that contacts MLflow.
    yield TestClient(app_module.app)
    app_module.store._current = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_when_a_model_is_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == "7"


def test_ready_is_503_without_a_model(client):
    app_module.store._current = None
    assert client.get("/ready").status_code == 503


def test_predict_returns_a_valid_prediction(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] == "7"


def test_predict_rejects_an_invalid_contract(client):
    assert (
        client.post("/predict", json={**EXAMPLE, "Contract": "Weekly"}).status_code
        == 422
    )


def test_predict_rejects_negative_tenure(client):
    assert client.post("/predict", json={**EXAMPLE, "tenure": -1}).status_code == 422


def test_predict_rejects_a_missing_field(client):
    payload = {k: v for k, v in EXAMPLE.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_is_503_without_a_model(client):
    app_module.store._current = None
    assert client.post("/predict", json=EXAMPLE).status_code == 503


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE, EXAMPLE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_batch_rejects_an_empty_list(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["model_name"] == "churn-classifier"
    assert body["model_version"] == "7"


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=EXAMPLE)
    text = client.get("/metrics").text
    assert "predict_requests_total" in text
    assert "predicted_churn_rate" in text


def test_predictions_are_logged_for_drift_monitoring(client, tmp_path):
    client.post("/predict", json=EXAMPLE)
    with open(tmp_path / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    for col in ["timestamp", "model_version", "churn_probability", *FEATURES]:
        assert col in rows[0]


def test_reload_requires_the_api_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setenv("RELOAD_API_KEY", "secret")
    assert client.post("/model/reload").status_code == 401


def test_a_failed_reload_keeps_the_current_model(client, monkeypatch):
    monkeypatch.delenv("RELOAD_API_KEY", raising=False)

    def boom(allow_fallback=True):
        raise RuntimeError("mlflow is down")

    monkeypatch.setattr(app_module.store, "load", boom)
    assert client.post("/model/reload").status_code == 502
    assert client.get("/model/info").json()["model_version"] == "7"


@pytest.mark.parametrize(
    "p,expected", [(0.05, "low"), (0.45, "medium"), (0.90, "high")]
)
def test_risk_level_bands(p, expected):
    assert app_module._risk_level(p) == expected
