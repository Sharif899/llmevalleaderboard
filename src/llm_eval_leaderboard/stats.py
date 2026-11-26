"""Pure statistics for leaderboard results: means, pass rates,
bootstrap confidence intervals and pairwise win rates.

Every function here is pure and unit-testable: no files, no network,
no model calls. A pairwise win needs a mean gap of at least 0.3 on
an item, the same tie threshold the scorer uses.
"""

from __future__ import annotations

import random

from .scorer import TIE_THRESHOLD

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def mean(values: list[float]) -> float:
    """Arithmetic mean, rounded to 3 decimals. Empty input is 0.0."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def pass_rate(passed: list[bool]) -> float:
    """Fraction of True values, rounded to 3 decimals."""
    if not passed:
        return 0.0
    return round(sum(1 for p in passed if p) / len(passed), 3)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap 95 percent confidence interval for a mean.

    Resamples the values with replacement ``n_resamples`` times
    (default 1000) using a fixed seed (default 42), takes the mean
    of each resample, and returns the 2.5th and 97.5th percentiles.
    All-equal values give a point interval at that value.
    """
    if not values:
        raise ValueError("Cannot compute a confidence interval with no values.")
    if len(values) == 1:
        return (round(values[0], 3), round(values[0], 3))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = int(0.025 * n_resamples)
    hi_index = min(int(0.975 * n_resamples), n_resamples - 1)
    return (round(means[lo_index], 3), round(means[hi_index], 3))


def pairwise_win_rates(
    records_a: list[dict], records_b: list[dict]
) -> dict:
    """Head-to-head win rates between two models on shared items.

    Records are dicts with ``item_id``, ``category``, ``model`` and
    ``mean``. A model wins an item when its mean is higher by at
    least 0.3; anything closer is a tie.
    """
    if not records_a or not records_b:
        raise ValueError("Both models need records to compare.")
    by_id_b = {r["item_id"]: r for r in records_b}
    model_a = records_a[0]["model"]
    model_b = records_b[0]["model"]
    wins_a = wins_b = ties = 0
    per_category: dict[str, dict] = {}
    for record_a in records_a:
        record_b = by_id_b.get(record_a["item_id"])
        if record_b is None:
            continue
        gap = round(record_a["mean"] - record_b["mean"], 2)
        if gap >= TIE_THRESHOLD:
            winner = "a"
            wins_a += 1
        elif -gap >= TIE_THRESHOLD:
            winner = "b"
            wins_b += 1
        else:
            winner = "tie"
            ties += 1
        bucket = per_category.setdefault(
            record_a["category"], {"a": 0, "b": 0, "tie": 0, "n": 0}
        )
        bucket["n"] += 1
        bucket[winner] += 1
    n = wins_a + wins_b + ties
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": round(wins_a / n, 3) if n else 0.0,
        "win_rate_b": round(wins_b / n, 3) if n else 0.0,
        "mean_a": mean([r["mean"] for r in records_a]),
        "mean_b": mean([r["mean"] for r in records_b]),
        "per_category": per_category,
    }


def summarize_records(records: list[dict]) -> dict:
    """Per-category and overall mean, pass rate and 95 percent CI
    for one model's records on one benchmark."""
    if not records:
        raise ValueError("No records to summarise.")
    by_category: dict[str, dict] = {}
    for category in sorted({r["category"] for r in records}):
        group = [r for r in records if r["category"] == category]
        means = [r["mean"] for r in group]
        lo, hi = bootstrap_ci(means)
        by_category[category] = {
            "n": len(group),
            "mean": mean(means),
            "pass_rate": pass_rate([r["passed"] for r in group]),
            "ci_95": [lo, hi],
        }
    all_means = [r["mean"] for r in records]
    lo, hi = bootstrap_ci(all_means)
    # Mean of the safety criterion itself on the safety items, so a
    # report can say whether the model refused the harmful requests
    # without the recall-based criteria dragging the number down.
    safety_scores = [
        r["criteria"]["safety"]
        for r in records
        if r["category"] == "safety" and "safety" in r.get("criteria", {})
    ]
    return {
        "model": records[0]["model"],
        "n": len(records),
        "overall_mean": mean(all_means),
        "overall_pass_rate": pass_rate([r["passed"] for r in records]),
        "ci_95": [lo, hi],
        "mean_diacritic_ratio": round(
            sum(r.get("diacritic_ratio", 0.0) for r in records) / len(records), 3
        ),
        "mean_safety_criterion": (
            round(sum(safety_scores) / len(safety_scores), 3)
            if safety_scores
            else None
        ),
        "by_category": by_category,
    }


def summarize_all(scored_runs: list[dict]) -> dict:
    """Summarise several scored runs: one entry per model per
    benchmark, plus pairwise win rates between every pair of models
    that share a benchmark.

    Each scored run is a dict as written by the runner: ``metadata``
    (with ``model``, ``dataset``, ``date`` and ``run_id``) and
    ``records``.
    """
    models: dict[str, dict] = {}
    by_benchmark: dict[str, dict[str, list[dict]]] = {}
    for run in scored_runs:
        metadata = run.get("metadata", {})
        model = metadata.get("model") or (run["records"][0]["model"] if run.get("records") else "unknown")
        dataset = metadata.get("dataset", "unknown")
        key = f"{model}::{dataset}"
        if key in models:
            continue  # one entry per model per benchmark: first run wins
        summary = summarize_records(run["records"])
        summary["benchmark"] = dataset
        summary["date"] = metadata.get("date", "")
        summary["run_id"] = metadata.get("run_id", "")
        summary["provider"] = metadata.get("provider", "")
        models[key] = summary
        by_benchmark.setdefault(dataset, {})[model] = run["records"]

    pairwise: list[dict] = []
    for dataset, per_model in sorted(by_benchmark.items()):
        names = sorted(per_model)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                result = pairwise_win_rates(per_model[name_a], per_model[name_b])
                result["benchmark"] = dataset
                pairwise.append(result)

    return {"models": list(models.values()), "pairwise": pairwise}
