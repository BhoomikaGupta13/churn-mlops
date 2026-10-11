"""Data quality checks. Exits non-zero if the data is not fit for training."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import load_params

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_dataframe(df: pd.DataFrame, max_null_rate: float = 0.05) -> list[str]:
    """Return a list of error messages (empty list = data is valid)."""
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors  # can't check the rest safely

    if len(df) == 0:
        errors.append("Dataset is empty")
        return errors

    # TotalCharges has blank strings in the raw file -> coerce, then count as nulls
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tc_null_rate = total_charges.isna().mean()
    if tc_null_rate > max_null_rate:
        errors.append(f"TotalCharges null rate {tc_null_rate:.3f} > {max_null_rate}")

    other_null_rate = df.drop(columns=["TotalCharges"]).isna().mean()
    bad_cols = other_null_rate[other_null_rate > max_null_rate]
    if not bad_cols.empty:
        errors.append(f"Columns above null threshold: {bad_cols.to_dict()}")

    if (pd.to_numeric(df["tenure"], errors="coerce") < 0).any():
        errors.append("tenure contains negative values")

    if (pd.to_numeric(df["MonthlyCharges"], errors="coerce") <= 0).any():
        errors.append("MonthlyCharges contains non-positive values")

    bad_labels = set(df["Churn"].dropna().unique()) - {"Yes", "No"}
    if bad_labels:
        errors.append(f"Churn has unexpected labels: {bad_labels}")

    if df["customerID"].duplicated().any():
        errors.append("Duplicate customerID values found")

    return errors


def main(path: str, report_path: str | None = None) -> int:
    params = load_params()
    df = pd.read_csv(path)
    errors = validate_dataframe(df, params["validate"]["max_null_rate"])

    if report_path:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {"path": path, "rows": len(df), "passed": not errors, "errors": errors},
                indent=2,
            )
        )

    if errors:
        print(f"VALIDATION FAILED for {path}")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"Validation passed for {path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    sys.exit(main(args.path, args.report))
