"""Agent 1: Content screening for DementiaNet transcripts.

Screens videos to determine if the subject is actually present and speaking,
before running the blind speaker analysis (Agent 2).

Usage:
    # Single run:
    python run_content_screening.py --run-dir "runs/Dementia_raw_data/Glen Campbell/after_symptoms/..."

    # Batch all runs:
    python run_content_screening.py --batch

    # Dry run (no LLM calls):
    python run_content_screening.py --batch --dry-run

    # Override provider:
    python run_content_screening.py --batch --provider claude --model claude-sonnet-4-20250514

    # Force reprocess:
    python run_content_screening.py --run-dir ... --force
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from speaker_analysis.config import LLMProviderConfig
from speaker_analysis.discovery import discover_single_run
from content_screening.config import ContentScreeningConfig
from content_screening.runner import process_single_screening, process_batch_screening

# Provider defaults (same as speaker analysis)
PROVIDER_DEFAULTS = {
    "gemini": {"model_name": "gemini-3-flash-preview", "api_key_env": "GEMINI_API_KEY"},
    "claude": {"model_name": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
    "openai": {"model_name": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
}


def main():
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description="Agent 1: Content screening for DementiaNet transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=str, help="Path to a single run directory")
    mode.add_argument("--batch", action="store_true", help="Process all discovered runs")

    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude", "openai"])
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, test discovery only")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--v2-only", action="store_true", help="Only process v2 pipeline runs")
    parser.add_argument("--runs-root", default=None, help="Override runs root directory")
    parser.add_argument("--csv-dir", default=None, help="Override CSV sources directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("content_screening.log"),
        ],
    )

    # Paths
    base_dir = Path(__file__).resolve().parent
    runs_root = Path(args.runs_root) if args.runs_root else base_dir / "runs"
    csv_dir = Path(args.csv_dir) if args.csv_dir else base_dir / "csv_sources"

    # LLM config
    prov_defaults = PROVIDER_DEFAULTS.get(args.provider, {})
    llm_config = LLMProviderConfig(
        provider=args.provider,
        model_name=args.model or prov_defaults.get("model_name", "gemini-3-flash-preview"),
        api_key_env=prov_defaults.get("api_key_env", "GEMINI_API_KEY"),
    )

    config = ContentScreeningConfig(
        llm=llm_config,
        skip_existing=not args.force,
        dry_run=args.dry_run,
    )

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
        run_info = discover_single_run(run_dir, runs_root)
        if run_info is None:
            logging.error(f"No processable run found at {run_dir}")
            return
        result = process_single_screening(run_info, config, csv_dir)
        logging.info(result)
    else:
        process_batch_screening(runs_root, config, csv_dir, v2_only=args.v2_only)


if __name__ == "__main__":
    main()
