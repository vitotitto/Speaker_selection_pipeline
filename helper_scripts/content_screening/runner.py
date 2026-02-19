"""Batch orchestration for content screening."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from speaker_analysis.discovery import RunInfo, discover_runs, discover_single_run

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.export_metadata import write_json

from .config import ContentScreeningConfig
from .screener import screen_run

logger = logging.getLogger(__name__)


def process_single_screening(
    run_info: RunInfo,
    config: ContentScreeningConfig,
    csv_dir: Path,
) -> str:
    """Screen a single run. Returns a status string."""
    output_path = run_info.run_dir / "metadata" / "content_screening.json"

    if config.skip_existing and output_path.exists():
        return f"SKIPPED (exists): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}"

    result = screen_run(run_info, config, csv_dir)

    # Save
    write_json(str(output_path), result)

    verdict = result.get("content_type", "unknown")
    usable = result.get("usable_for_analysis", False)
    return (
        f"{'USABLE' if usable else 'NOT_USABLE'}: {run_info.person}/{run_info.timepoint}/{run_info.video_stem} "
        f"— type={verdict}, subject_speaking={result.get('subject_speaking', False)}"
    )


def process_batch_screening(
    runs_root: Path,
    config: ContentScreeningConfig,
    csv_dir: Path,
    v2_only: bool = False,
) -> None:
    """Discover all runs and screen each one."""
    runs = discover_runs(runs_root, v2_only=v2_only)
    logger.info(f"Discovered {len(runs)} runs for content screening")

    usable_count = 0
    not_usable_count = 0
    skipped_count = 0
    failed_count = 0

    for i, run_info in enumerate(runs):
        label = f"[{i + 1}/{len(runs)}]"
        try:
            status = process_single_screening(run_info, config, csv_dir)
            logger.info(f"{label} {status}")
            if status.startswith("USABLE"):
                usable_count += 1
            elif status.startswith("NOT_USABLE"):
                not_usable_count += 1
            elif status.startswith("SKIPPED"):
                skipped_count += 1
        except Exception as e:
            failed_count += 1
            logger.error(
                f"{label} FAILED {run_info.person}/{run_info.timepoint}/{run_info.video_stem}: {e}",
                exc_info=True,
            )

    logger.info(
        f"Content screening complete: {usable_count} usable, "
        f"{not_usable_count} not usable, {skipped_count} skipped, "
        f"{failed_count} failed (of {len(runs)} total)"
    )
