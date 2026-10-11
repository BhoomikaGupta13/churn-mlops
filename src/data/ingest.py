"""Split raw data into a reference (training) set and sequential 'incoming' batches."""

import numpy as np
import pandas as pd

from src.config import ROOT, load_params


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    ing_cfg = params["ingest"]

    df = pd.read_csv(ROOT / data_cfg["raw_path"])

    # Shuffle with a fixed seed so the split is reproducible
    df = df.sample(frac=1.0, random_state=data_cfg["random_state"]).reset_index(
        drop=True
    )

    n_ref = int(len(df) * ing_cfg["reference_fraction"])
    reference = df.iloc[:n_ref]
    incoming = df.iloc[n_ref:]

    out_dir = ROOT / data_cfg["batches_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    reference.to_csv(ROOT / data_cfg["reference_path"], index=False)
    print(f"reference.csv -> {len(reference)} rows")

    batches = np.array_split(incoming, ing_cfg["n_batches"])
    for i, batch in enumerate(batches, start=1):
        batch.to_csv(out_dir / f"batch_{i}.csv", index=False)
        print(f"batch_{i}.csv -> {len(batch)} rows")

    # 'latest' is the batch the monitoring/CT jobs look at (starts as batch_1)
    batches[0].to_csv(ROOT / data_cfg["latest_path"], index=False)
    print("latest.csv    -> copy of batch_1")


if __name__ == "__main__":
    main()
