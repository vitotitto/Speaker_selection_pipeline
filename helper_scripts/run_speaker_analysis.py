"""Stage 2: LLM-based speaker classification for DementiaNet transcripts.

Usage:
    # Single run:
    python run_speaker_analysis.py --run-dir runs/v2_test

    # Batch all runs:
    python run_speaker_analysis.py --batch

    # Dry run (pre-filter only, no LLM):
    python run_speaker_analysis.py --batch --dry-run

    # Override provider:
    python run_speaker_analysis.py --batch --provider claude --model claude-sonnet-4-20250514

    # Force reprocess:
    python run_speaker_analysis.py --run-dir runs/v2_test --force
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from speaker_analysis.config import SpeakerAnalysisConfig, LLMProviderConfig
from speaker_analysis.discovery import discover_single_run
from speaker_analysis.runner import process_single_run, process_batch

# Provider defaults
PROVIDER_DEFAULTS = {
    "gemini": {"model_name": "gemini-2.0-flash", "api_key_env": "GEMINI_API_KEY"},
    "claude": {"model_name": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
    "openai": {"model_name": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
}


def main():
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description="Stage 2: LLM speaker classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=str, help="Path to a single run directory")
    mode.add_argument("--batch", action="store_true", help="Process all discovered runs")

    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude", "openai"])
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--dry-run", action="store_true", help="Pre-filter only, no LLM calls")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--v2-only", action="store_true", help="Only process v2 pipeline runs")
    parser.add_argument("--runs-root", default=None, help="Override runs root directory")
    parser.add_argument("--csv-dir", default=None, help="Override CSV sources directory")
    parser.add_argument(
        "--ignore-content-screening",
        action="store_true",
        help="Force speaker analysis even when content_screening marks run unusable.",
    )
    parser.add_argument(
        "--disable-named-turn-guard",
        action="store_true",
        help="Disable deterministic named-turn guard demotions for this run.",
    )
    parser.add_argument(
        "--disable-subject-anchor",
        action="store_true",
        help="Disable subject-anchor enforcement inside the named-turn guard.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("speaker_analysis.log"),
        ],
    )

    # Build config
    helper_dir = Path(__file__).resolve().parent
    project_root = helper_dir.parent
    runs_root = Path(args.runs_root) if args.runs_root else project_root / "runs"
    csv_dir = Path(args.csv_dir) if args.csv_dir else project_root / "csv_sources"
    if not runs_root.is_absolute():
        cwd_candidate = (Path.cwd() / runs_root).resolve()
        project_candidate = (project_root / runs_root).resolve()
        helper_candidate = (helper_dir / runs_root).resolve()
        if cwd_candidate.exists():
            runs_root = cwd_candidate
        elif project_candidate.exists():
            runs_root = project_candidate
        else:
            runs_root = helper_candidate
    if not csv_dir.is_absolute():
        cwd_candidate = (Path.cwd() / csv_dir).resolve()
        project_candidate = (project_root / csv_dir).resolve()
        helper_candidate = (helper_dir / csv_dir).resolve()
        if cwd_candidate.exists():
            csv_dir = cwd_candidate
        elif project_candidate.exists():
            csv_dir = project_candidate
        else:
            csv_dir = helper_candidate

    # Provider config
    prov_defaults = PROVIDER_DEFAULTS.get(args.provider, {})
    llm_config = LLMProviderConfig(
        provider=args.provider,
        model_name=args.model or prov_defaults.get("model_name", "gemini-2.0-flash"),
        api_key_env=prov_defaults.get("api_key_env", "GEMINI_API_KEY"),
    )

    config = SpeakerAnalysisConfig(
        llm=llm_config,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        ignore_content_screening=args.ignore_content_screening,
    )
    if args.disable_named_turn_guard:
        config.post_validation.enable_named_turn_guard = False
    if args.disable_subject_anchor:
        config.post_validation.enforce_subject_anchor_when_cued = False

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            cwd_candidate = (Path.cwd() / run_dir).resolve()
            project_candidate = (project_root / run_dir).resolve()
            helper_candidate = (helper_dir / run_dir).resolve()
            if cwd_candidate.exists():
                run_dir = cwd_candidate
            elif project_candidate.exists():
                run_dir = project_candidate
            else:
                run_dir = helper_candidate
        run_info = discover_single_run(run_dir, runs_root)
        if run_info is None:
            logging.error(f"No processable run found at {run_dir}")
            return
        result = process_single_run(run_info, config, csv_dir)
        logging.info(result)
    else:
        process_batch(runs_root, config, csv_dir, v2_only=args.v2_only)


if __name__ == "__main__":
    main()
