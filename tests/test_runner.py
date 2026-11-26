"""Tests for the runner: item loading, resume, failure recording,
and provider response parsing (via the providers module)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_leaderboard.providers import ProviderError
from llm_eval_leaderboard.runner import (
    dataset_sha256,
    load_items,
    load_response_rows,
    run_evaluation,
    score_responses_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    """A test provider: returns a canned response per item id, or
    raises for ids listed in ``fail_ids``."""

    name = "fake"
    model = "fake-model"
    temperature = 0.0

    def __init__(self, responses=None, fail_ids=None, fail_status=500):
        self.responses = responses or {}
        self.fail_ids = set(fail_ids or [])
        self.fail_status = fail_status
        self.calls: list[str] = []

    def complete(self, prompt, max_tokens=1024):
        self.calls.append(prompt)
        for item_id, response in self.responses.items():
            if item_id in prompt:
                return response
        raise ProviderError("fake failure", status_code=self.fail_status)


class ReferenceProvider:
    """Returns the item reference: looks it up by prompt text."""

    name = "fake"
    model = "fake-model"
    temperature = 0.0

    def __init__(self, items, fail_ids=None):
        self.by_prompt = {item.prompt: item.reference for item in items}
        self.fail_ids = set(fail_ids or [])
        self.by_prompt_fail = {}
        self.calls: list[str] = []

    def complete(self, prompt, max_tokens=1024):
        self.calls.append(prompt)
        if prompt in self.by_prompt_fail:
            raise ProviderError("boom", status_code=500)
        return self.by_prompt[prompt]


def _write_items(tmp_path, rows):
    path = tmp_path / "items.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_load_items_tolerates_extra_fields():
    path = REPO_ROOT / "data" / "yoruba.jsonl"
    items = load_items(path)
    assert len(items) == 60
    assert items[0].extra.get("prompt_gloss")
    assert items[0].extra.get("reference_gloss")


def test_load_items_tolerates_scoring_notes():
    path = REPO_ROOT / "data" / "general.jsonl"
    items = load_items(path)
    assert len(items) == 40
    assert items[0].extra.get("scoring_notes")


def test_load_items_rejects_missing_field(tmp_path):
    path = _write_items(tmp_path, [{"id": "x1", "category": "c", "prompt": "p"}])
    with pytest.raises(ValueError, match="reference"):
        load_items(path)


def test_load_items_rejects_duplicate_id(tmp_path):
    row = {"id": "x1", "category": "c", "prompt": "p", "reference": "r"}
    path = _write_items(tmp_path, [row, row])
    with pytest.raises(ValueError, match="duplicated"):
        load_items(path)


def test_dataset_sha256_matches_recorded_value():
    assert dataset_sha256(REPO_ROOT / "data" / "yoruba.jsonl") == (
        "4b9a5b5e428bdf172db8f08ed9ea763c2477c3a9e896adf50b6e53d6a4a874dc"
    )


def test_run_scores_items_and_writes_files(tmp_path):
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:4]
    provider = ReferenceProvider(items)
    result = run_evaluation(
        provider, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl",
        "run-1", tmp_path / "results", throttle_seconds=0, sleep_fn=lambda s: None,
    )
    assert len(result["records"]) == 4
    assert result["failures"] == []
    assert result["metadata"]["dataset"] == "yoruba"
    assert result["metadata"]["dataset_sha256"] == dataset_sha256(
        REPO_ROOT / "data" / "yoruba.jsonl"
    )
    assert result["metadata"]["model"] == "fake-model"
    run_dir = tmp_path / "results" / "run-1"
    assert (run_dir / "responses.jsonl").is_file()
    assert (run_dir / "scored.json").is_file()
    assert (run_dir / "state.json").is_file()
    raw = [json.loads(line) for line in open(run_dir / "responses.jsonl", encoding="utf-8")]
    assert len(raw) == 4
    assert raw[0]["response"] == items[0].reference
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["completed"] == sorted(item.id for item in items)
    assert state["remaining"] == 0


def test_run_records_failure_and_never_invents_a_response(tmp_path):
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:3]
    provider = ReferenceProvider(items)
    provider.by_prompt_fail = {items[1].prompt}
    result = run_evaluation(
        provider, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl",
        "run-1", tmp_path / "results", throttle_seconds=0, sleep_fn=lambda s: None,
    )
    assert len(result["records"]) == 2
    assert len(result["failures"]) == 1
    assert result["failures"][0]["item_id"] == items[1].id
    assert result["failures"][0]["status_code"] == 500
    raw = [json.loads(line) for line in open(
        tmp_path / "results" / "run-1" / "responses.jsonl", encoding="utf-8")]
    assert all(row["item_id"] != items[1].id for row in raw)
    assert all(row["response"] for row in raw)


def test_run_retries_once_on_503_then_succeeds(tmp_path):
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:2]

    class FlakyProvider(ReferenceProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0

        def complete(self, prompt, max_tokens=1024):
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderError("overloaded", status_code=503)
            return super().complete(prompt, max_tokens)

    provider = FlakyProvider(items)
    result = run_evaluation(
        provider, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl",
        "run-1", tmp_path / "results", throttle_seconds=0, sleep_fn=lambda s: None,
    )
    assert len(result["records"]) == 2
    assert result["failures"] == []
    assert provider.attempts == 3  # 1 failed + 1 retry + 1 second item


def test_run_resumes_and_skips_done_items(tmp_path):
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:3]
    failing = ReferenceProvider(items)
    failing.by_prompt_fail = {items[2].prompt}
    run_evaluation(
        failing, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl",
        "run-1", tmp_path / "results", throttle_seconds=0, sleep_fn=lambda s: None,
    )
    good = ReferenceProvider(items)
    result = run_evaluation(
        good, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl",
        "run-1", tmp_path / "results", throttle_seconds=0, sleep_fn=lambda s: None,
    )
    # The two already-scored items are not called again; the failed
    # item is retried and now scores.
    assert len(good.calls) == 1
    assert len(result["records"]) == 3
    assert result["failures"] == []


def test_score_responses_file_scores_without_live_calls(tmp_path):
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:2]
    responses_path = tmp_path / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({
                "prompt_id": item.id, "model": "my-model",
                "response": item.reference,
            }, ensure_ascii=False) + "\n")
    scored = score_responses_file(
        responses_path, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl"
    )
    assert len(scored["records"]) == 2
    assert scored["records"][0]["mean"] >= 4.0
    assert scored["metadata"]["model"] == "my-model"


def test_load_response_rows_rejects_unknown_prompt_id(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"prompt_id": "nope", "model": "m", "response": "x"}) + "\n",
        encoding="utf-8",
    )
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:1]
    with pytest.raises(ValueError, match="Unknown prompt_id"):
        score_responses_file(
            path, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl"
        )


def test_load_response_rows_rejects_empty_response(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"prompt_id": "yb-faq-01", "model": "m", "response": "  "}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_response_rows(path)


def test_load_response_rows_accepts_item_id_alias(tmp_path):
    # The live runner writes the item id as 'item_id'; the score
    # command must be able to re-score a run's own responses file.
    path = tmp_path / "responses.jsonl"
    items = load_items(REPO_ROOT / "data" / "yoruba.jsonl")[:1]
    path.write_text(
        json.dumps({"item_id": items[0].id, "model": "m",
                    "response": items[0].reference}) + "\n",
        encoding="utf-8",
    )
    rows = load_response_rows(path)
    assert rows[0]["prompt_id"] == items[0].id
    scored = score_responses_file(
        path, items, "yoruba", REPO_ROOT / "data" / "yoruba.jsonl"
    )
    assert len(scored["records"]) == 1
