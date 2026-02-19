from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .classifier import classify_run
from .config import SpeakerAnalysisConfig
from .context import build_patient_context
from .discovery import RunInfo, discover_runs, discover_single_run

# Reuse existing write_json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.export_metadata import write_json

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthesize_asr_info(segments: list) -> dict:
    duration = max((s.get("end", 0) for s in segments), default=0.0)
    return {
        "language": "en",
        "language_probability": 0.0,
        "duration": duration,
        "duration_after_vad": duration,
    }


def process_single_run(
    run_info: RunInfo,
    config: SpeakerAnalysisConfig,
    csv_dir: Path,
) -> str:
    output_path = run_info.run_dir / "metadata" / "speaker_analysis.json"

    if config.skip_existing and output_path.exists():
        return f"SKIPPED (exists): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}"

    # Gate on content screening (unless explicitly overridden by config).
    screening_path = run_info.run_dir / "metadata" / "content_screening.json"
    if screening_path.exists():
        screening = _load_json(screening_path)
        if not config.ignore_content_screening and not screening.get("usable_for_analysis", True):
            content_type = screening.get("content_type", "unknown")
            return (
                f"SKIPPED (screening: {content_type}): "
                f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}"
            )

    # Load transcript
    transcript_data = _load_json(run_info.transcript_path)
    segments = transcript_data.get("segments", [])
    if not segments:
        return f"SKIPPED (empty transcript): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}"

    # Load ASR info
    if run_info.asr_info_path and run_info.asr_info_path.exists():
        asr_info = _load_json(run_info.asr_info_path)
    else:
        asr_info = _synthesize_asr_info(segments)

    # Build patient context
    patient_context = build_patient_context(
        run_info.person,
        run_info.source,
        run_info.timepoint,
        run_info.video_stem,
        csv_dir,
    )

    # Classify
    result = classify_run(segments, asr_info, patient_context, config)

    # Save
    write_json(str(output_path), result)

    stats = result.get("statistics", {})
    return (
        f"SUCCESS: {run_info.person}/{run_info.timepoint}/{run_info.video_stem} "
        f"— patient: {stats.get('patient_segments', '?')} segs "
        f"({stats.get('patient_fraction', 0):.0%}), "
        f"recommended: {len(result.get('recommended_segments', []))}"
    )


def process_batch(
    runs_root: Path,
    config: SpeakerAnalysisConfig,
    csv_dir: Path,
    v2_only: bool = False,
) -> None:
    runs = discover_runs(runs_root, v2_only=v2_only)
    logger.info(f"Discovered {len(runs)} processable runs")

    for i, run_info in enumerate(runs):
        label = f"[{i + 1}/{len(runs)}]"
        try:
            status = process_single_run(run_info, config, csv_dir)
            logger.info(f"{label} {status}")
        except Exception as e:
            logger.error(
                f"{label} FAILED {run_info.person}/{run_info.timepoint}/{run_info.video_stem}: {e}",
                exc_info=True,
            )
