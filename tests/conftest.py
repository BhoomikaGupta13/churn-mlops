import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import FEATURES, TARGET, build_pipeline
from tests.data_factory import make_customers


@pytest.fixture(scope="session")
def customers() -> pd.DataFrame:
    """Shared fake dataset. Tests that change it must call .copy() first."""
    return make_customers(1000)


@pytest.fixture(scope="session")
def trained_pipeline(customers):
    X = customers[FEATURES]
    y = (customers[TARGET] == "Yes").astype(int)
    pipe = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    return pipe.fit(X, y)
