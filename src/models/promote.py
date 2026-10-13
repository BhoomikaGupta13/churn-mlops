"""Apply the promotion gate to the latest retrain result and move the champion alias if it passes."""

import json
import os

from src.config import ROOT, load_params
from src.models.gate import should_promote
from src.models.registry import get_alias_version, get_client


def main() -> None:
    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    out_dir = ROOT / params["drift"]["output_dir"]
    result = json.loads((out_dir / "retrain_result.json").read_text())

    ok, reason = should_promote(
        result["champion_metrics"], result["challenger_metrics"], params["promotion"]
    )

    client = get_client()
    version = result["challenger_version"]
    previous = get_alias_version(client, name, "champion")
    previous_version = str(previous.version) if previous else None

    if ok:
        client.set_registered_model_alias(name, "champion", version)
        client.set_model_version_tag(name, version, "promotion", "promoted")
        if previous_version:
            client.set_model_version_tag(
                name, version, "previous_champion", previous_version
            )
    else:
        client.set_model_version_tag(name, version, "promotion", "rejected")
    client.set_model_version_tag(name, version, "promotion_reason", reason)

    decision = {
        "promoted": ok,
        "reason": reason,
        "challenger_version": version,
        "previous_champion_version": previous_version,
        "challenger_metrics": result["challenger_metrics"],
        "champion_metrics": result["champion_metrics"],
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    # Lets a GitHub workflow read the outcome as a step output.
    gh_output = os.getenv("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"promoted={str(ok).lower()}\n")

    print(f"{'PROMOTED' if ok else 'REJECTED'}: challenger v{version}. {reason}")
    if ok and previous_version:
        print(
            f"Rollback: point the 'champion' alias back to version {previous_version}"
        )


if __name__ == "__main__":
    main()
