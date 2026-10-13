"""The promotion gate: decides whether a challenger may replace the champion."""


def should_promote(
    champion_metrics: dict | None, challenger_metrics: dict, cfg: dict
) -> tuple[bool, str]:
    metric = cfg["primary_metric"]
    min_gain = cfg["min_improvement"]
    floor = cfg["min_absolute_f1"]

    new = challenger_metrics[metric]
    if new < floor:
        return False, f"{metric}={new:.4f} is below the minimum floor {floor}"

    if champion_metrics is None:
        return (
            True,
            f"no champion exists and {metric}={new:.4f} clears the floor {floor}",
        )

    old = champion_metrics[metric]
    gain = round(new - old, 4)  # rounding avoids float noise such as 0.00999999
    if gain < min_gain:
        return (
            False,
            f"{metric} gain {gain:+.4f} ({old:.4f} -> {new:.4f}) is below the required {min_gain}",
        )
    return (
        True,
        f"{metric} improved {old:.4f} -> {new:.4f} ({gain:+.4f}, required {min_gain})",
    )
