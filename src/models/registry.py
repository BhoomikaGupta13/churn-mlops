"""MLflow registry helpers. Run as a script to print the current registry state."""

import os

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import load_params


def resolve_tracking_uri(params: dict | None = None) -> str:
    """Env var wins (used in CI/Docker), otherwise params.yaml."""
    params = params or load_params()
    return os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri or resolve_tracking_uri())


def get_alias_version(client: MlflowClient, name: str, alias: str):
    """Return the ModelVersion an alias points to, or None if the alias doesn't exist."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def main() -> None:
    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    client = get_client()

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        print(f"No versions registered for '{name}' yet.")
        return

    print(f"Registry: {name}")
    print(f"{'version':<9}{'aliases':<26}{'f1':<9}{'roc_auc':<9}model_type")
    for mv in sorted(versions, key=lambda v: int(v.version)):
        run = client.get_run(mv.run_id)
        m, p = run.data.metrics, run.data.params
        print(
            f"{mv.version:<9}{','.join(mv.aliases) or '-':<26}"
            f"{m.get('f1', float('nan')):<9.4f}{m.get('roc_auc', float('nan')):<9.4f}"
            f"{p.get('model_type', '?')}"
        )


if __name__ == "__main__":
    main()
