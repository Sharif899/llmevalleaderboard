"""Running live evaluations: call a provider, score, save, resume.

The runner never invents a response. A call that fails after one
retry is recorded as a failure, with its error and HTTP status, and
the run moves on. Progress is written to disk after every item, so
re-running the same run id resumes and skips items already done.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .providers import RETRYABLE_STATUSES, Provider, ProviderError
from .scorer import score_response

DEFAULT_THROTTLE_SECONDS = 8.5
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class Item:
    """One benchmark item. Extra fields from the JSONL (prompt_gloss,
    reference_gloss, scoring_notes and anything else) are kept in
    ``extra`` and ignored for scoring."""

    id: str
    category: str
    prompt: str
    reference: str
    extra: dict = field(default_factory=dict)


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 of a dataset file, for pinning a run to its data."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(path: str | Path) -> list[Item]:
    """Load items from a JSONL file.

    Each line needs a non-empty ``id``, ``category``, ``prompt`` and
    ``reference``. Extra fields are tolerated and kept. A malformed
    line, a missing field or a duplicated id raises ValueError
    naming the line, because silently skipping bad rows would make
    results unreproducible.
    """
    items_path = Path(path)
    items: list[Item] = []
    seen: set[str] = set()
    with open(items_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{items_path}:{line_no}: not valid JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{items_path}:{line_no}: each item must be a JSON object"
                )
            for field_name in ("id", "category", "prompt", "reference"):
                if field_name not in raw or not str(raw[field_name]).strip():
                    raise ValueError(
                        f"{items_path}:{line_no}: item is missing a "
                        f"non-empty '{field_name}'"
                    )
            if raw["id"] in seen:
                raise ValueError(
                    f"{items_path}:{line_no}: duplicated item id '{raw['id']}'"
                )
            seen.add(raw["id"])
            extra = {
                key: value
                for key, value in raw.items()
                if key not in ("id", "category", "prompt", "reference")
            }
            items.append(
                Item(
                    id=raw["id"],
                    category=raw["category"],
                    prompt=raw["prompt"],
                    reference=raw["reference"],
                    extra=extra,
                )
            )
    if not items:
        raise ValueError(f"{items_path}: no items found")
    return items


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(model: str, item: Item, response: str, result) -> dict:
    return {
        "item_id": item.id,
        "category": item.category,
        "model": model,
        "prompt": item.prompt,
        "response": response,
        "reference": item.reference,
        "mean": result.mean,
        "total": result.total,
        "passed": result.passed,
        "diacritic_ratio": result.diacritic_ratio,
        "criteria": {c.criterion: c.score for c in result.criteria},
        "rationales": {c.criterion: c.rationale for c in result.criteria},
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _call_with_retry(
    provider: Provider,
    prompt: str,
    max_tokens: int,
    retry_backoff: float,
    sleep_fn,
) -> str:
    """One provider call, with a single retry on 429/503."""
    try:
        return provider.complete(prompt, max_tokens=max_tokens)
    except ProviderError as exc:
        if exc.status_code in RETRYABLE_STATUSES:
            sleep_fn(retry_backoff)
            return provider.complete(prompt, max_tokens=max_tokens)
        raise


def run_evaluation(
    provider: Provider,
    items: list[Item],
    dataset_name: str,
    dataset_path: str | Path,
    run_id: str,
    out_dir: str | Path,
    throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
    max_tokens: int = 1024,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    commands_used: list[str] | None = None,
    sleep_fn=time.sleep,
) -> dict:
    """Run one provider over one item set and save the run.

    Writes ``results/<run-id>/`` under ``out_dir``:
    ``responses.jsonl`` (raw responses verbatim), ``scored.json``
    (per-item criteria, mean, passed, plus run metadata) and
    ``state.json`` (progress, so a re-run resumes). Returns the
    scored data dict. Failures are recorded in ``scored.json`` under
    ``failures`` and in ``state.json``; no response is ever invented.
    """
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_path = run_dir / "responses.jsonl"
    scored_path = run_dir / "scored.json"
    state_path = run_dir / "state.json"

    started = _now_iso()
    model = provider.model
    temperature = getattr(provider, "temperature", None)
    dataset_sha = dataset_sha256(dataset_path)

    # Resume: keep records and raw responses already on disk.
    existing = _read_json(scored_path) or {}
    records: list[dict] = list(existing.get("records", []))
    failures: list[dict] = list(existing.get("failures", []))
    completed = {r["item_id"] for r in records}
    metadata = existing.get("metadata") or {
        "run_id": run_id,
        "date": started[:10],
        "started_at": started,
        "model": model,
        "provider": provider.name,
        "dataset": dataset_name,
        "dataset_sha256": dataset_sha,
        "temperature": temperature,
        "commands_used": list(commands_used or []),
    }

    def save_progress() -> None:
        _write_json(
            scored_path,
            {"metadata": metadata, "records": records, "failures": failures},
        )
        _write_json(
            state_path,
            {
                "run_id": run_id,
                "total": len(items),
                "completed": sorted(completed),
                "failed": sorted({f["item_id"] for f in failures}),
                "remaining": len(items) - len(completed),
                "updated_at": _now_iso(),
            },
        )

    save_progress()
    first_call = True
    for item in items:
        if item.id in completed:
            continue
        if not first_call and throttle_seconds > 0:
            sleep_fn(throttle_seconds)
        first_call = False
        try:
            response = _call_with_retry(
                provider, item.prompt, max_tokens, retry_backoff, sleep_fn
            )
        except ProviderError as exc:
            # A failed item is retried on the next run of this run
            # id: drop any earlier failure entry for it first.
            failures = [f for f in failures if f["item_id"] != item.id]
            failures.append(
                {
                    "item_id": item.id,
                    "category": item.category,
                    "error": str(exc),
                    "status_code": exc.status_code,
                }
            )
            save_progress()
            continue
        except Exception as exc:  # noqa: BLE001 - record, do not invent
            failures = [f for f in failures if f["item_id"] != item.id]
            failures.append(
                {
                    "item_id": item.id,
                    "category": item.category,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status_code": None,
                }
            )
            save_progress()
            continue
        timestamp = _now_iso()
        with open(responses_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "item_id": item.id,
                        "prompt": item.prompt,
                        "response": response,
                        "model": model,
                        "timestamp": timestamp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        result = score_response(item.prompt, response, item.reference, item.category)
        records.append(_record(model, item, response, result))
        completed.add(item.id)
        # A retried item that now succeeds leaves the failure list.
        failures = [f for f in failures if f["item_id"] != item.id]
        save_progress()

    metadata["finished_at"] = _now_iso()
    metadata["n_scored"] = len(records)
    metadata["n_failed"] = len(failures)
    save_progress()
    return {"metadata": metadata, "records": records, "failures": failures}


def score_response_rows(
    rows: list[dict], items: list[Item], model: str
) -> list[dict]:
    """Score already-collected response rows against items.

    Each row needs ``prompt_id`` (an item id), ``model`` and
    ``response``. Rows are scored in file order.
    """
    by_id = {item.id: item for item in items}
    records: list[dict] = []
    for row in rows:
        item = by_id[row["prompt_id"]]
        result = score_response(
            item.prompt, row["response"], item.reference, item.category
        )
        records.append(_record(model, item, row["response"], result))
    return records


def load_response_rows(path: str | Path) -> list[dict]:
    """Load and validate a JSONL file of collected responses.

    Strict: a malformed line, an unknown prompt id, a missing field
    or an empty response raises ValueError naming the line.
    """
    rows_path = Path(path)
    rows: list[dict] = []
    with open(rows_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{rows_path}:{line_no}: not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{rows_path}:{line_no}: each row must be a JSON object "
                    "with 'prompt_id', 'model' and 'response'"
                )
            # The live runner writes the item id as 'item_id'; accept
            # it as an alias so `lbleader score` can re-score a run's
            # own responses file.
            if "prompt_id" not in row and "item_id" in row:
                row["prompt_id"] = row["item_id"]
            for field_name in ("prompt_id", "model", "response"):
                if field_name not in row:
                    raise ValueError(
                        f"{rows_path}:{line_no}: missing field '{field_name}'"
                    )
            if not isinstance(row["response"], str) or not row["response"].strip():
                raise ValueError(
                    f"{rows_path}:{line_no}: 'response' must be a non-empty string"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{rows_path}: no responses found")
    return rows


def score_responses_file(
    responses_path: str | Path,
    items: list[Item],
    dataset_name: str,
    dataset_path: str | Path,
    model: str | None = None,
) -> dict:
    """Score a responses file and return scored data (no live calls)."""
    rows = load_response_rows(responses_path)
    unknown = [r["prompt_id"] for r in rows if r["prompt_id"] not in {i.id for i in items}]
    if unknown:
        raise ValueError(f"Unknown prompt_id(s) in {responses_path}: {', '.join(sorted(set(unknown)))}")
    model_name = model or str(rows[0]["model"])
    records = score_response_rows(rows, items, model_name)
    return {
        "metadata": {
            "date": _now_iso()[:10],
            "model": model_name,
            "dataset": dataset_name,
            "dataset_sha256": dataset_sha256(dataset_path),
            "source_responses": str(responses_path),
            "n_scored": len(records),
        },
        "records": records,
        "failures": [],
    }


def load_scored(path: str | Path) -> dict:
    """Load a scored.json file written by this runner."""
    with open(Path(path), encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "records" not in data:
        raise ValueError(f"{path}: not a scored results file (expected 'records')")
    return data
