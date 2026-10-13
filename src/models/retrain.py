"""Train a challenger on reference + new data, compare it with the champion, register it.

This script never promotes anything. Promotion is a separate, gated step (promote.py).
"""

import argparse
import importlib.metadata as md
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_params
from src.data.validate import validate_dataframe
from src.features import FEATURES, ID_COLUMN, TARGET, build_pipeline, coerce_numeric
from src.models.evaluate import compute_metrics
from src.models.registry import get_alias_version, get_client, resolve_tracking_uri
from src.models.train import build_model, get_data_version, get_git_sha


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data", default=params["data"]["latest_path"])
    parser.add_argument("--trigger", default="manual", help="why this retrain happened")
    args = parser.parse_args()

    seed = params["data"]["random_state"]
    processed = ROOT / params["data"]["processed_dir"]
    name = params["mlflow"]["registered_model_name"]

    # 1) validate and prepare the new batch (it must carry labels)
    new_raw = pd.read_csv(ROOT / args.new_data)
    errors = validate_dataframe(new_raw, params["validate"]["max_null_rate"])
    if errors:
        print(f"New data failed validation: {errors}")
        raise SystemExit(1)

    new = new_raw.drop(columns=[ID_COLUMN])
    new[TARGET] = (new[TARGET] == "Yes").astype(int)
    new_fit, new_eval = train_test_split(
        new,
        test_size=params["retrain"]["new_data_eval_fraction"],
        stratify=new[TARGET],
        random_state=seed,
    )

    # 2) challenger trains on old train + half the new batch;
    #    BOTH models are judged on old test + the other half (same data for both)
    # coerce_numeric makes TotalCharges numeric everywhere (blank strings -> 0), so the
    # old CSVs (strings) and the new batch (numbers) can be combined and logged safely
    fit_df = coerce_numeric(
        pd.concat([pd.read_csv(processed / "train.csv"), new_fit], ignore_index=True)
    )
    eval_df = coerce_numeric(
        pd.concat([pd.read_csv(processed / "test.csv"), new_eval], ignore_index=True)
    )
    X_fit, y_fit = fit_df[FEATURES], fit_df[TARGET]
    X_eval, y_eval = eval_df[FEATURES], eval_df[TARGET]

    model_type = params["train"]["model_type"]
    challenger = build_pipeline(build_model(params)).fit(X_fit, y_fit)
    challenger_metrics = compute_metrics(challenger, X_eval, y_eval)

    # 3) score the current champion on the same evaluation data
    tracking_uri = resolve_tracking_uri(params)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    client = get_client(tracking_uri)

    champion_mv = get_alias_version(client, name, "champion")
    champion_metrics = None
    if champion_mv is not None:
        champion = mlflow.sklearn.load_model(f"models:/{name}/{champion_mv.version}")
        champion_metrics = compute_metrics(champion, X_eval, y_eval)

    # 4) log and register the challenger
    with mlflow.start_run(run_name=f"retrain-{model_type}") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params["train"][model_type])
        mlflow.log_param("fit_rows", len(X_fit))
        mlflow.log_param("eval_rows", len(X_eval))
        mlflow.set_tags(
            {
                "run_kind": "retrain",
                "trigger": args.trigger,
                "new_data": args.new_data,
                "git_sha": get_git_sha(),
                "data_version": get_data_version(params),
            }
        )
        mlflow.log_metrics(challenger_metrics)
        if champion_metrics:
            mlflow.log_metrics(
                {f"champion_{k}": v for k, v in champion_metrics.items()}
            )

        pip_reqs = [
            f"{p}=={md.version(p)}" for p in ("scikit-learn", "pandas", "numpy")
        ]
        mlflow.sklearn.log_model(
            challenger,
            artifact_path="model",
            signature=infer_signature(X_fit, challenger.predict(X_fit)),
            registered_model_name=name,
            pip_requirements=pip_reqs,
        )
        run_id = run.info.run_id

    version = client.search_model_versions(f"run_id='{run_id}'")[0].version
    client.set_registered_model_alias(name, "challenger", version)

    # 5) hand the comparison to the gate
    result = {
        "challenger_version": str(version),
        "challenger_metrics": challenger_metrics,
        "champion_version": str(champion_mv.version) if champion_mv else None,
        "champion_metrics": champion_metrics,
        "eval_rows": len(X_eval),
        "trigger": args.trigger,
        "new_data": args.new_data,
    }
    out = ROOT / params["drift"]["output_dir"] / "retrain_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"Challenger v{version}: {challenger_metrics}")
    print(f"Champion   v{result['champion_version']}: {champion_metrics}")


if __name__ == "__main__":
    main()
