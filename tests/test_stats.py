"""Tests for leaderboard statistics: bootstrap CI and pairwise win rates."""

from __future__ import annotations

import pytest

from llm_eval_leaderboard.stats import (
    bootstrap_ci,
    mean,
    pass_rate,
    pairwise_win_rates,
    summarize_records,
)


def _record(item_id, category, model, score, passed=True):
    return {
        "item_id": item_id,
        "category": category,
        "model": model,
        "mean": score,
        "passed": passed,
        "diacritic_ratio": 0.0,
    }


def test_mean_and_pass_rate():
    assert mean([4.0, 5.0, 3.0]) == 4.0
    assert mean([]) == 0.0
    assert pass_rate([True, True, False]) == pytest.approx(0.667, abs=0.001)
    assert pass_rate([]) == 0.0


def test_bootstrap_ci_all_equal_is_a_point():
    lo, hi = bootstrap_ci([4.0] * 20)
    assert lo == 4.0
    assert hi == 4.0


def test_bootstrap_ci_contains_sample_mean():
    values = [3.0, 4.0, 5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 5.0]
    lo, hi = bootstrap_ci(values)
    assert lo <= mean(values) <= hi
    assert lo <= hi


def test_bootstrap_ci_wider_spread_gives_wider_interval():
    tight = [4.0] * 19 + [4.2]
    wide = [1.0, 5.0] * 10
    lo_t, hi_t = bootstrap_ci(tight)
    lo_w, hi_w = bootstrap_ci(wide)
    assert (hi_w - lo_w) > (hi_t - lo_t)


def test_bootstrap_ci_is_deterministic_with_fixed_seed():
    values = [2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_pairwise_win_rates_clear_winner():
    a = [_record(f"i{n}", "factual_qa", "model-a", 5.0) for n in range(5)]
    b = [_record(f"i{n}", "factual_qa", "model-b", 4.0) for n in range(5)]
    result = pairwise_win_rates(a, b)
    assert result["wins_a"] == 5
    assert result["wins_b"] == 0
    assert result["ties"] == 0
    assert result["n"] == 5
    assert result["win_rate_a"] == 1.0


def test_pairwise_tie_rule_gap_under_0_3_is_a_tie():
    a = [_record("i1", "factual_qa", "model-a", 4.2)]
    b = [_record("i1", "factual_qa", "model-b", 4.0)]
    result = pairwise_win_rates(a, b)
    assert result["ties"] == 1
    assert result["wins_a"] == 0
    assert result["wins_b"] == 0


def test_pairwise_tie_rule_gap_of_exactly_0_3_is_a_win():
    a = [_record("i1", "factual_qa", "model-a", 4.3)]
    b = [_record("i1", "factual_qa", "model-b", 4.0)]
    result = pairwise_win_rates(a, b)
    assert result["wins_a"] == 1
    assert result["ties"] == 0


def test_pairwise_only_shared_items_count():
    a = [_record("i1", "factual_qa", "model-a", 5.0),
         _record("i2", "factual_qa", "model-a", 5.0)]
    b = [_record("i1", "factual_qa", "model-b", 4.0)]
    result = pairwise_win_rates(a, b)
    assert result["n"] == 1


def test_pairwise_empty_raises():
    with pytest.raises(ValueError):
        pairwise_win_rates([], [_record("i1", "factual_qa", "model-b", 4.0)])


def test_summarize_records_mean_safety_criterion():
    records = [
        {**_record("i1", "factual_qa", "model-a", 5.0, True),
         "criteria": {"safety": 5}},
        {**_record("i2", "safety", "model-a", 3.0, False),
         "criteria": {"safety": 5}},
        {**_record("i3", "safety", "model-a", 2.8, False),
         "criteria": {"safety": 5}},
    ]
    summary = summarize_records(records)
    # The safety criterion is 5 on both safety items even though
    # their overall means are 3.0 and 2.8.
    assert summary["mean_safety_criterion"] == 5.0
    assert summary["by_category"]["safety"]["mean"] == 2.9


def test_summarize_records_mean_safety_criterion_none_without_criteria():
    records = [_record("i1", "factual_qa", "model-a", 5.0, True)]
    assert summarize_records(records)["mean_safety_criterion"] is None
