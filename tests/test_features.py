import numpy as np
import pandas as pd

from src.features import FEATURES, coerce_numeric


def test_coerce_numeric_turns_blanks_into_zero(customers):
    out = coerce_numeric(customers)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert not out["TotalCharges"].isna().any()
    assert out.loc[0, "TotalCharges"] == 0.0


def test_coerce_numeric_does_not_change_its_input(customers):
    before = customers["TotalCharges"].copy()
    coerce_numeric(customers)
    assert customers["TotalCharges"].equals(before)


def test_pipeline_survives_an_unseen_category(trained_pipeline, customers):
    row = customers[FEATURES].head(1).copy()
    row["Contract"] = "Weekly"  # never seen during training
    proba = trained_pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


def test_pipeline_survives_a_missing_number(trained_pipeline, customers):
    row = customers[FEATURES].head(1).copy()
    row["MonthlyCharges"] = np.nan
    proba = trained_pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= proba[0] <= 1.0
