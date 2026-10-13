"""Detect data drift between the reference (training) data and a new batch, using Evidently."""

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import ROOT, load_params
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    coerce_numeric,
)

# SeniorCitizen is a 0/1 flag, so it is tested as a category rather than as a number.
DRIFT_NUMERIC = [c for c in NUMERIC_FEATURES if c != "SeniorCitizen"]
DRIFT_CATEGORICAL = [*CATEGORICAL_FEATURES, "SeniorCitizen"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model input features (no ID, no target) with TotalCharges made numeric."""
    return coerce_numeric(df)[FEATURES].reset_index(drop=True)


def _drift_table(report_dict: dict) -> dict:
    """Find the per-column drift table among the preset's metrics (order differs by version)."""
    for metric in report_dict["metrics"]:
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            return result
    raise RuntimeError(
        "Evidently report has no per-column drift table; check the version"
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
    html_path: Path | None = None,
) -> dict:
    mapping = ColumnMapping(
        numerical_features=DRIFT_NUMERIC, categorical_features=DRIFT_CATEGORICAL
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=_prepare(reference),
        current_data=_prepare(current),
        column_mapping=mapping,
    )

    result = _drift_table(report.as_dict())
    n_features = int(result["number_of_columns"])
    n_drifted = int(result["number_of_drifted_columns"])
    share = n_drifted / n_features if n_features else 0.0
    drifted = sorted(
        c for c, info in result["drift_by_columns"].items() if info["drift_detected"]
    )

    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

    return {
        "drift_detected": bool(share >= threshold),
        "share_drifted": round(share, 4),
        "n_drifted": n_drifted,
        "n_features": n_features,
        "threshold": threshold,
        "drifted_features": drifted,
        "reference_rows": len(reference),
        "current_rows": len(current),
    }


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=params["data"]["reference_path"])
    parser.add_argument("--current", default=params["data"]["latest_path"])
    parser.add_argument("--output-dir", default=params["drift"]["output_dir"])
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = detect_drift(
        pd.read_csv(ROOT / args.reference),
        pd.read_csv(ROOT / args.current),
        params["drift"]["share_drifted_features_threshold"],
        html_path=out_dir / "drift_report.html",
    )
    summary["current_path"] = args.current
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Drift detected: {summary['drift_detected']} | "
        f"{summary['n_drifted']}/{summary['n_features']} features drifted "
        f"(share {summary['share_drifted']}, threshold {summary['threshold']})"
    )
    print(f"Drifted features: {summary['drifted_features']}")
    print(f"Report: {out_dir / 'drift_report.html'}")


if __name__ == "__main__":
    main()
