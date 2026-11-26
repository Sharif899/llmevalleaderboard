"""Command line interface: run, score, report, compare.

The CLI is thin: the work lives in runner, stats and report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .providers import ProviderError, get_provider
from .report import build_summary, generate_reports, load_scored_runs
from .runner import (
    DEFAULT_THROTTLE_SECONDS,
    dataset_sha256,
    load_items,
    run_evaluation,
    score_responses_file,
)

BENCHMARKS = {
    "yoruba": ("yoruba", "data/yoruba.jsonl"),
    "general": ("general", "data/general.jsonl"),
}


def _repo_root() -> Path:
    """The repo root: the parent of the directory holding this file's
    package when installed editable, else the current directory."""
    here = Path(__file__).resolve()
    for parent in [here.parent.parent.parent.parent, Path.cwd()]:
        if (parent / "data" / "yoruba.jsonl").is_file():
            return parent
    return Path.cwd()


def _benchmark_path(name: str) -> tuple[str, Path]:
    dataset_name, rel = BENCHMARKS[name]
    path = _repo_root() / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark '{name}' data not found at {path}. "
            "For 'general', data/general.jsonl is written separately."
        )
    return dataset_name, path


def _default_run_id(model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model}"


def cmd_run(args: argparse.Namespace) -> int:
    try:
        provider = get_provider(args.provider)
    except (ProviderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    names = ["yoruba", "general"] if args.benchmark == "all" else [args.benchmark]
    commands_used = [" ".join(sys.argv)]
    for name in names:
        try:
            dataset_name, path = _benchmark_path(name)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        items = load_items(path)
        run_id = args.run_id
        if args.benchmark == "all":
            run_id = f"{run_id}-{name}"
        print(
            f"Running {provider.model} ({provider.name}) on {dataset_name}: "
            f"{len(items)} items, dataset SHA-256 "
            f"{dataset_sha256(path)[:12]}..., run id {run_id}."
        )
        try:
            result = run_evaluation(
                provider,
                items,
                dataset_name,
                path,
                run_id,
                args.out,
                throttle_seconds=args.throttle,
                max_tokens=args.max_tokens,
                commands_used=commands_used,
            )
        except ProviderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        metadata = result["metadata"]
        print(
            f"Done: {metadata['n_scored']} scored, "
            f"{metadata['n_failed']} failed. "
            f"Results in {Path(args.out) / run_id}."
        )
        if metadata["n_failed"]:
            print(
                "Failed items were recorded, not scored. "
                "Re-run the same run id to resume them."
            )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    try:
        dataset_name, path = _benchmark_path(args.benchmark)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    items = load_items(path)
    try:
        scored = score_responses_file(
            args.responses, items, dataset_name, path, model=args.model
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"scored-{scored['metadata']['model']}-{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(scored, handle, ensure_ascii=False, indent=2)
    print(f"Scored {len(scored['records'])} responses. Wrote {out_path}.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        written = generate_reports(args.results, readme_path=args.readme)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote summary: {written['summary']}")
    print(f"Wrote leaderboard: {written['leaderboard']}")
    for path in written["models"]:
        print(f"Wrote model report: {path}")
    if "readme" in written:
        print(f"Updated README leaderboard table: {written['readme']}")
    else:
        print("No README.md found next to the results directory; table not updated.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        runs = load_scored_runs(args.results)
        summary = build_summary(runs)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not summary["pairwise"]:
        print("Only one model (or no shared benchmark) found; nothing to compare.")
        return 0
    for result in summary["pairwise"]:
        print(
            f"{result['benchmark']}: {result['model_a']} vs {result['model_b']}: "
            f"{result['wins_a']}-{result['wins_b']} with {result['ties']} ties "
            f"on {result['n']} shared items "
            f"(means {result['mean_a']:.3f} vs {result['mean_b']:.3f})."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lbleader",
        description="Live LLM evaluation leaderboard: run models, score responses, build reports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a live evaluation against a provider.")
    run_p.add_argument("--provider", required=True, choices=["gemini", "groq"])
    run_p.add_argument("--benchmark", required=True, choices=["yoruba", "general", "all"])
    run_p.add_argument("--run-id", default=None, help="Run id (default: timestamp plus model).")
    run_p.add_argument("--out", default="results", help="Results directory (default: results).")
    run_p.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE_SECONDS,
                       help="Seconds between provider calls (default 8.5).")
    run_p.add_argument("--max-tokens", type=int, default=1024)
    run_p.set_defaults(func=cmd_run)

    score_p = sub.add_parser("score", help="Score already-collected responses (no live calls).")
    score_p.add_argument("--responses", required=True, help="JSONL file of responses.")
    score_p.add_argument("--benchmark", required=True, choices=["yoruba", "general"])
    score_p.add_argument("--model", default=None, help="Model name (default: from the responses file).")
    score_p.add_argument("--out", default="results", help="Directory for the scored JSON.")
    score_p.set_defaults(func=cmd_score)

    report_p = sub.add_parser("report", help="Generate summary, leaderboard and per-model reports.")
    report_p.add_argument("--results", required=True, help="Results directory holding scored runs.")
    report_p.add_argument("--readme", default=None, help="README.md to update (default: next to the results directory).")
    report_p.set_defaults(func=cmd_report)

    compare_p = sub.add_parser("compare", help="Show pairwise win rates between models.")
    compare_p.add_argument("--results", required=True, help="Results directory holding scored runs.")
    compare_p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not args.run_id:
        # Default run id needs the provider model name, which needs
        # the key, so resolve it here after parsing.
        try:
            provider = get_provider(args.provider)
        except (ProviderError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        args.run_id = _default_run_id(provider.model)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
