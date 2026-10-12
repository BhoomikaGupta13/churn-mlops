import copy

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import load_params
from src.features import FEATURES, TARGET, build_pipeline
from src.models.evaluate import compute_metrics
from src.models.train import build_model


def test_smoke_train_and_evaluate(customers):
    X = customers[FEATURES]
    y = (customers[TARGET] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    pipe = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipe.fit(X_tr, y_tr)
    metrics = compute_metrics(pipe, X_te, y_te)

    assert set(metrics) == {"f1", "precision", "recall", "roc_auc", "accuracy"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # The fake data has a real signal, so a working pipeline must beat a coin flip.
    assert metrics["roc_auc"] > 0.6


def test_both_models_in_params_yaml_can_be_built_and_fitted(customers):
    params = copy.deepcopy(load_params())
    params["train"]["random_forest"]["n_estimators"] = 20  # keep the test fast
    X = customers[FEATURES]
    y = (customers[TARGET] == "Yes").astype(int)

    for model_type in ("random_forest", "logistic_regression"):
        params["train"]["model_type"] = model_type
        pipe = build_pipeline(build_model(params))
        pipe.fit(X, y)
        assert pipe.predict_proba(X.head(3)).shape == (3, 2)


def test_unknown_model_type_raises():
    params = copy.deepcopy(load_params())
    params["train"]["model_type"] = "svm"
    with pytest.raises(ValueError):
        build_model(params)
