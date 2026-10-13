"""Create an artificially drifted copy of a batch (for testing drift detection)."""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path


def inject_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    # 1) Customers are now paying a lot more
    out["MonthlyCharges"] = (out["MonthlyCharges"] * 1.4).round(2)

    # 2) Customers are much newer
    out["tenure"] = (out["tenure"] * 0.4).round().astype(int)

    # 3) Most customers shift to month-to-month contracts
    mask = rng.random(len(out)) < 0.6
    out.loc[mask, "Contract"] = "Month-to-month"

    # 4) More fiber optic users
    mask = rng.random(len(out)) < 0.4
    out.loc[mask, "InternetService"] = "Fiber optic"

    # Keep TotalCharges roughly consistent with the changed columns
    out["TotalCharges"] = (out["tenure"] * out["MonthlyCharges"]).round(2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    drifted = inject_drift(df, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    drifted.to_csv(args.output, index=False)
    print(f"Wrote drifted data ({len(drifted)} rows) -> {args.output}")


if __name__ == "__main__":
    main()
