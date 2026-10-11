"""Drop the ID column, encode the target, and make a stratified train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_params
from src.features import ID_COLUMN, TARGET


def main() -> None:
    params = load_params()
    cfg = params["data"]

    df = pd.read_csv(ROOT / cfg["reference_path"])
    df = df.drop(columns=[ID_COLUMN])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    train, test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df[TARGET],
    )

    out = ROOT / cfg["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(
        f"train: {len(train)} rows | test: {len(test)} rows | churn rate: {df[TARGET].mean():.3f}"
    )


if __name__ == "__main__":
    main()
