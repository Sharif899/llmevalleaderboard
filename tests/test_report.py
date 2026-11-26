"""Tests for report generation: leaderboard Markdown, README table
replacement and per-model reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_leaderboard.report import (
    LEADERBOARD_END,
    LEADERBOARD_START,
    build_summary,
    generate_reports,
    leaderboard_markdown,
    model_report_markdown,
    replace_readme_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(item_id, category, model, score, passed=True):
    return {
        "item_id": item_id,
        "category": category,
        "model": model,
        "prompt": "prompt",
        "response": "response",
        "reference": "reference",
        "mean": score,
        "total": int(score * 5),
        "passed": passed,
        "diacritic_ratio": 0.5,
        "criteria": {},
        "rationales": {},
    }


def _run(model, dataset, scores_by_category, failures=None, date="2026-11-26"):
    records = []
    n = 0
    for category, scores in scores_by_category.items():
        for score in scores:
            n += 1
            records.append(_record(f"{category}-{n}", category, model, score, score >= 4.0))
    return {
        "metadata": {
            "run_id": f"run-{model}-{dataset}",
            "date": date,
            "model": model,
            "provider": "gemini",
            "dataset": dataset,
            "dataset_sha256": "abc",
            "temperature": 0,
            "commands_used": ["lbleader run"],
        },
        "records": records,
        "failures": failures or [],
    }


@pytest.fixture
def two_model_runs():
    return [
        _run("gemini-3.5-flash-lite", "yoruba", {
            "factual_qa": [5.0, 4.0, 4.0, 5.0],
            "safety": [5.0, 5.0, 5.0, 5.0],
        }),
        _run("openai/gpt-oss-120b", "yoruba", {
            "factual_qa": [4.0, 3.0, 4.0, 3.0],
            "safety": [5.0, 5.0, 4.0, 4.0],
        }),
    ]


def test_build_summary_overall_and_pairwise(two_model_runs):
    summary = build_summary(two_model_runs)
    assert len(summary["models"]) == 2
    assert len(summary["leaderboard"]) == 2
    best = summary["leaderboard"][0]
    assert best["model"] == "gemini-3.5-flash-lite"
    assert best["overall_mean"] == 4.75
    assert best["per_benchmark"]["yoruba"] == 4.75
    assert best["ci_95"][0] <= 4.75 <= best["ci_95"][1]
    assert len(summary["pairwise"]) == 1
    pair = summary["pairwise"][0]
    assert pair["model_a"] == "gemini-3.5-flash-lite"
    assert pair["wins_a"] + pair["wins_b"] + pair["ties"] == 8


def test_leaderboard_markdown_contains_right_numbers(two_model_runs):
    summary = build_summary(two_model_runs)
    table = leaderboard_markdown(summary)
    assert "| gemini-3.5-flash-lite | 4.750 |" in table
    assert "| openai/gpt-oss-120b | 4.000 |" in table
    assert "factual_qa 4.500" in table
    assert "2026-11-26" in table
    assert table.index("gemini-3.5-flash-lite") < table.index("openai/gpt-oss-120b")


def test_leaderboard_markdown_empty_raises():
    with pytest.raises(ValueError):
        leaderboard_markdown({"leaderboard": []})


def test_replace_readme_table():
    readme = (
        "# Title\n\n"
        f"{LEADERBOARD_START}\n\nOld table.\n\n{LEADERBOARD_END}\n\nRest.\n"
    )
    table = "| Model | Overall mean |\n| --- | ---: |\n| m | 4.500 |"
    updated = replace_readme_table(readme, table)
    assert "| m | 4.500 |" in updated
    assert "Old table." not in updated
    assert updated.startswith("# Title\n\n" + LEADERBOARD_START)
    assert updated.endswith(LEADERBOARD_END + "\n\nRest.\n")
    # Idempotent: replacing again keeps one table.
    updated_again = replace_readme_table(updated, table)
    assert updated_again.count("m | 4.500") == 1


def test_replace_readme_table_missing_markers_raises():
    with pytest.raises(ValueError, match="no leaderboard section"):
        replace_readme_table("# Title\n\nNo markers here.\n", "| x |")


def test_model_report_content(two_model_runs):
    summary = build_summary(two_model_runs)
    entry = next(e for e in summary["models"] if e["model"] == "openai/gpt-oss-120b")
    text = model_report_markdown(entry)
    assert "openai/gpt-oss-120b" in text
    assert "Overall mean: 4.000" in text
    assert "Best: safety" in text
    assert "Worst: factual_qa" in text
    assert "None: every item in this run returned a response." in text


def test_model_report_lists_failures():
    run = _run("gemini-3.5-flash-lite", "yoruba", {"factual_qa": [5.0, 5.0]},
               failures=[{"item_id": "yb-faq-03", "error": "HTTP 503", "status_code": 503}])
    summary = build_summary([run])
    text = model_report_markdown(summary["models"][0])
    assert "yb-faq-03" in text
    assert "HTTP 503" in text
    assert "No response was invented" in text


def test_model_report_includes_safety_criterion_note():
    run = _run("gemini-3.5-flash-lite", "yoruba", {
        "factual_qa": [5.0, 5.0],
        "safety": [3.0, 2.8],
    })
    for record in run["records"]:
        if record["category"] == "safety":
            record["criteria"] = {"safety": 5}
    summary = build_summary([run])
    text = model_report_markdown(summary["models"][0])
    assert "Mean safety criterion on the safety items: 5.000" in text
    assert "The safety criterion, not the category mean, is the measure of refusal." in text


def test_generate_reports_writes_files_and_updates_readme(tmp_path, two_model_runs):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run-1"
    run_dir.mkdir(parents=True)
    with open(run_dir / "scored.json", "w", encoding="utf-8") as handle:
        json.dump(two_model_runs[0], handle, ensure_ascii=False)
    run_dir2 = results_dir / "run-2"
    run_dir2.mkdir(parents=True)
    with open(run_dir2 / "scored.json", "w", encoding="utf-8") as handle:
        json.dump(two_model_runs[1], handle, ensure_ascii=False)
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Test\n\n" + LEADERBOARD_START + "\n\nPlaceholder.\n\n" + LEADERBOARD_END + "\n",
        encoding="utf-8",
    )
    written = generate_reports(results_dir, readme_path=readme)
    assert written["summary"].is_file()
    assert written["leaderboard"].is_file()
    assert len(written["models"]) == 2
    summary = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert summary["leaderboard"][0]["model"] == "gemini-3.5-flash-lite"
    assert "Placeholder." not in readme.read_text(encoding="utf-8")
    assert "4.750" in readme.read_text(encoding="utf-8")
    assert "4.750" in written["leaderboard"].read_text(encoding="utf-8")
