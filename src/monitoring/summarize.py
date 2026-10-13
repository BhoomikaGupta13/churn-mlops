"""Write a human-readable markdown summary of a continuous-training run."""

import json
import os
from pathlib import Path

from src.config import ROOT, load_params


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def build_summary(drift, retrain, decision) -> str:
    lines = ["## Continuous training run", ""]
    if drift is None:
        lines.append("No drift summary was produced (an earlier step failed).")
        return "\n".join(lines)

    flag = "YES" if drift["drift_detected"] else "no"
    lines.append(
        f"**Drift detected:** {flag} | {drift['n_drifted']}/{drift['n_features']} features "
        f"drifted (share {drift['share_drifted']}, threshold {drift['threshold']})"
    )
    if drift["drifted_features"]:
        lines.append("")
        lines.append(
            "Drifted features: "
            + ", ".join(f"`{f}`" for f in drift["drifted_features"])
        )
    lines.append("")

    if retrain is None:
        lines.append("No retraining was triggered.")
        return "\n".join(lines)

    champion_metrics = retrain.get("champion_metrics") or {}
    lines += [
        "### Champion vs challenger (same evaluation data)",
        "",
        "| metric | champion | challenger |",
        "|---|---|---|",
    ]
    for metric, value in retrain["challenger_metrics"].items():
        lines.append(
            f"| {metric} | {_fmt(champion_metrics.get(metric))} | {_fmt(value)} |"
        )
    lines.append("")

    if decision:
        verdict = "PROMOTED" if decision["promoted"] else "REJECTED"
        lines.append(
            f"**Gate decision: {verdict}** (challenger v{decision['challenger_version']})"
        )
        lines.append("")
        lines.append(decision["reason"])
        if decision["promoted"] and decision.get("previous_champion_version"):
            lines.append("")
            lines.append(
                f"Rollback: point the `champion` alias back to version "
                f"{decision['previous_champion_version']}."
            )
    return "\n".join(lines)


def main() -> None:
    out_dir = ROOT / load_params()["drift"]["output_dir"]
    text = build_summary(
        _load(out_dir / "drift_summary.json"),
        _load(out_dir / "retrain_result.json"),
        _load(out_dir / "decision.json"),
    )
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
