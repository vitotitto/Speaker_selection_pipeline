"""Stage 3: Extract patient speech audio from classified pipeline runs.

Usage:
    # Single run:
    python run_audio_extraction.py --run-dir "runs/Dementia_raw_data/Bill Buckner/..."

    # Batch all runs:
    python run_audio_extraction.py --batch

    # Dry run (preview without writing audio):
    python run_audio_extraction.py --batch --dry-run

    # Custom quality threshold:
    python run_audio_extraction.py --batch --quality-threshold 0.8

    # Keep at most the best 5 minutes of speech:
    python run_audio_extraction.py --batch --max-total-minutes 5

    # Keep only segments at least 4 seconds long:
    python run_audio_extraction.py --batch --min-segment-duration-s 4

    # Optional cap for unusually long single ASR segments:
    python run_audio_extraction.py --batch --max-segment-duration-s 40

    # Group only near-consecutive segments (default max gap 0.75s):
    python run_audio_extraction.py --batch --max-gap-s 0.75

    # Keep every segment as a separate output clip:
    python run_audio_extraction.py --batch --split-segments

    # Standardize output sample rate (optional):
    python run_audio_extraction.py --batch --resample-to 16000

    # Force overwrite existing output:
    python run_audio_extraction.py --run-dir runs/... --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audio_extraction.extractor import (
    SELECTION_MODE_CONTINUITY_FIRST,
    SELECTION_MODE_SEGMENT_FIRST,
    extract_patient_audio,
)
from speaker_analysis.discovery import discover_runs, discover_single_run


def _build_output_dir(run_info, output_root: Path) -> Path:
    """Mirror the run's source/person/timepoint/video_stem under output_root."""
    return output_root / run_info.source / run_info.person / run_info.timepoint / run_info.video_stem


def _has_existing_output(output_dir: Path) -> bool:
    legacy_file = output_dir / "patient_speech.wav"
    has_legacy = legacy_file.exists()
    has_parts = any(output_dir.glob("patient_speech_part_*.wav"))
    has_manifest = (output_dir / "extraction_manifest.json").exists()
    return has_legacy or (has_manifest and has_parts)


def _configure_console_utf8() -> None:
    """Avoid Unicode logging failures on Windows consoles with legacy encodings."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class _ConsoleSafeFilter(logging.Filter):
    """Escape non-encodable console characters instead of crashing logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.getMessage().encode("cp1252")
        except UnicodeEncodeError:
            record.msg = record.getMessage().encode("ascii", "backslashreplace").decode("ascii")
            record.args = ()
        return True


def _process_single(
    run_dir: Path,
    runs_root: Path,
    output_root: Path,
    speaker_analysis_file: str,
    quality_threshold: float,
    top_quality_fraction: float | None,
    selection_mode: str,
    max_total_duration_s: float | None,
    min_segment_duration_s: float | None,
    max_segment_duration_s: float | None,
    output_sample_rate: int | None,
    group_consecutive: bool,
    max_group_gap_s: float,
    acoustic_post_filter: bool,
    acoustic_min_speech_band_ratio: float,
    acoustic_min_voiced_ratio: float,
    acoustic_max_music_score: float,
    acoustic_min_speaker_consistency: float,
    acoustic_min_subject_similarity: float,
    denoise_output: bool,
    denoise_strength: float,
    ignore_content_screening: bool,
    force: bool,
    dry_run: bool,
) -> str:
    run_info = discover_single_run(run_dir, runs_root)
    if run_info is None:
        return f"SKIPPED (not a valid run): {run_dir}"

    output_dir = _build_output_dir(run_info, output_root)

    # Skip if output exists (unless --force)
    if not force and not dry_run and _has_existing_output(output_dir):
        return (
            f"SKIPPED (exists): "
            f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}"
        )

    result = extract_patient_audio(
        run_dir=run_dir,
        output_dir=output_dir,
        quality_threshold=quality_threshold,
        top_quality_fraction=top_quality_fraction,
        selection_mode=selection_mode,
        max_total_duration_s=max_total_duration_s,
        min_segment_duration_s=min_segment_duration_s,
        max_segment_duration_s=max_segment_duration_s,
        output_sample_rate=output_sample_rate,
        group_consecutive=group_consecutive,
        max_group_gap_s=max_group_gap_s,
        acoustic_post_filter=acoustic_post_filter,
        acoustic_min_speech_band_ratio=acoustic_min_speech_band_ratio,
        acoustic_min_voiced_ratio=acoustic_min_voiced_ratio,
        acoustic_max_music_score=acoustic_max_music_score,
        acoustic_min_speaker_consistency=acoustic_min_speaker_consistency,
        acoustic_min_subject_similarity=acoustic_min_subject_similarity,
        denoise_output=denoise_output,
        denoise_strength=denoise_strength,
        speaker_analysis_filename=speaker_analysis_file,
        ignore_content_screening=ignore_content_screening,
        dry_run=dry_run,
    )

    if result is None:
        return (
            f"SKIPPED (no usable segments): "
            f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}"
        )

    action = "DRY RUN" if dry_run else "EXTRACTED"
    return (
        f"{action}: {run_info.person}/{run_info.timepoint}/{run_info.video_stem} "
        f"- {result['clips_generated']} clips, {result['segments_included']} segments, "
        f"{result['total_duration_s']:.1f}s"
    )


def main():
    _configure_console_utf8()

    parser = argparse.ArgumentParser(
        description="Stage 3: Extract patient speech audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=str, help="Path to a single run directory")
    mode.add_argument("--batch", action="store_true", help="Process all discovered runs")

    parser.add_argument(
        "--quality-threshold", type=float, default=0.7,
        help="Minimum quality_score for inclusion (default: 0.7)",
    )
    parser.add_argument(
        "--top-quality-fraction", type=float, default=None,
        help="Keep only top fraction of quality-ranked segments before duration cap (e.g. 0.30).",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default=SELECTION_MODE_CONTINUITY_FIRST,
        choices=[SELECTION_MODE_CONTINUITY_FIRST, SELECTION_MODE_SEGMENT_FIRST],
        help=(
            "Selection strategy: continuity_first (group-first, preferred) "
            "or segment_first (legacy)."
        ),
    )
    parser.add_argument(
        "--max-total-minutes", type=float, default=5.0,
        help="Maximum selected speech duration in minutes (default: 5.0).",
    )
    parser.add_argument(
        "--min-segment-duration-s", type=float, default=4.0,
        help="Exclude single segments shorter than this many seconds (default: 4.0).",
    )
    parser.add_argument(
        "--max-segment-duration-s", type=float, default=None,
        help="Optional: exclude single segments longer than this many seconds (default: disabled).",
    )
    parser.add_argument(
        "--resample-to", type=int, default=None,
        help="Optional output sample rate (e.g. 16000). Default preserves source rate.",
    )
    parser.add_argument(
        "--split-segments", action="store_true",
        help="Write one output file per segment instead of grouping consecutive segments.",
    )
    parser.add_argument(
        "--max-gap-s", type=float, default=0.75,
        help="Max silence gap (seconds) for grouping consecutive segments (default: 0.75).",
    )
    parser.add_argument(
        "--acoustic-post-filter", action="store_true",
        help="Enable lightweight acoustic post-filter to drop music/mixed-speaker segments.",
    )
    parser.add_argument(
        "--acoustic-min-speech-band-ratio", type=float, default=0.45,
        help="Minimum speech-band energy ratio to keep a segment (default: 0.45).",
    )
    parser.add_argument(
        "--acoustic-min-voiced-ratio", type=float, default=0.35,
        help="Minimum voiced-frame ratio to keep a segment (default: 0.35).",
    )
    parser.add_argument(
        "--acoustic-max-music-score", type=float, default=0.45,
        help="Maximum music-likeness score to keep a segment (default: 0.45).",
    )
    parser.add_argument(
        "--acoustic-min-speaker-consistency", type=float, default=0.68,
        help="Minimum speaker-consistency score to keep a segment (default: 0.68).",
    )
    parser.add_argument(
        "--acoustic-min-subject-similarity", type=float, default=0.52,
        help="Minimum similarity to run-level subject voice centroid (default: 0.52).",
    )
    parser.add_argument(
        "--denoise-output", action="store_true",
        help="Apply lightweight spectral denoise to extracted clips.",
    )
    parser.add_argument(
        "--denoise-strength", type=float, default=0.65,
        help="Denoise strength for --denoise-output (default: 0.65).",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output root directory (default: processed/)",
    )
    parser.add_argument(
        "--speaker-analysis-file",
        type=str,
        default="speaker_analysis.json",
        help="Metadata filename to use for speaker-analysis input (default: speaker_analysis.json).",
    )
    parser.add_argument("--runs-root", type=str, default=None, help="Override runs root directory")
    parser.add_argument(
        "--ignore-content-screening",
        action="store_true",
        help="Force extraction even when content_screening marks run unusable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing audio")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()
    if args.resample_to is not None and args.resample_to <= 0:
        parser.error("--resample-to must be a positive integer")
    if args.max_total_minutes is not None and args.max_total_minutes <= 0:
        parser.error("--max-total-minutes must be > 0")
    if args.top_quality_fraction is not None and not (0.0 < args.top_quality_fraction <= 1.0):
        parser.error("--top-quality-fraction must be in (0.0, 1.0]")
    if args.min_segment_duration_s is not None and args.min_segment_duration_s <= 0:
        parser.error("--min-segment-duration-s must be > 0")
    if args.max_segment_duration_s is not None and args.max_segment_duration_s <= 0:
        parser.error("--max-segment-duration-s must be > 0")
    if (
        args.min_segment_duration_s is not None
        and args.max_segment_duration_s is not None
        and args.min_segment_duration_s > args.max_segment_duration_s
    ):
        parser.error("--min-segment-duration-s cannot exceed --max-segment-duration-s")
    if args.max_gap_s < 0:
        parser.error("--max-gap-s must be >= 0")
    for name in (
        "acoustic_min_speech_band_ratio",
        "acoustic_min_voiced_ratio",
        "acoustic_max_music_score",
        "acoustic_min_speaker_consistency",
        "acoustic_min_subject_similarity",
    ):
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} must be in [0.0, 1.0]")
    if args.denoise_strength < 0:
        parser.error("--denoise-strength must be >= 0")

    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(_ConsoleSafeFilter())
    file_handler = logging.FileHandler("audio_extraction.log", encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logger = logging.getLogger(__name__)

    helper_dir = Path(__file__).resolve().parent
    project_root = helper_dir.parent
    runs_root = Path(args.runs_root) if args.runs_root else project_root / "runs"
    output_root = Path(args.output_root) if args.output_root else project_root / "processed"
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
    if not output_root.is_absolute():
        cwd_candidate = (Path.cwd() / output_root).resolve()
        project_candidate = (project_root / output_root).resolve()
        helper_candidate = (helper_dir / output_root).resolve()
        if cwd_candidate.exists():
            output_root = cwd_candidate
        elif project_candidate.exists():
            output_root = project_candidate
        else:
            output_root = helper_candidate
    max_total_duration_s = (
        args.max_total_minutes * 60.0 if args.max_total_minutes is not None else None
    )

    group_consecutive = not args.split_segments

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
        status = _process_single(
            run_dir=run_dir,
            runs_root=runs_root,
            output_root=output_root,
            speaker_analysis_file=args.speaker_analysis_file,
            quality_threshold=args.quality_threshold,
            top_quality_fraction=args.top_quality_fraction,
            selection_mode=args.selection_mode,
            max_total_duration_s=max_total_duration_s,
            min_segment_duration_s=args.min_segment_duration_s,
            max_segment_duration_s=args.max_segment_duration_s,
            output_sample_rate=args.resample_to,
            group_consecutive=group_consecutive,
            max_group_gap_s=args.max_gap_s,
            acoustic_post_filter=args.acoustic_post_filter,
            acoustic_min_speech_band_ratio=args.acoustic_min_speech_band_ratio,
            acoustic_min_voiced_ratio=args.acoustic_min_voiced_ratio,
            acoustic_max_music_score=args.acoustic_max_music_score,
            acoustic_min_speaker_consistency=args.acoustic_min_speaker_consistency,
            acoustic_min_subject_similarity=args.acoustic_min_subject_similarity,
            denoise_output=args.denoise_output,
            denoise_strength=args.denoise_strength,
            ignore_content_screening=args.ignore_content_screening,
            force=args.force,
            dry_run=args.dry_run,
        )
        logger.info(status)
    else:
        # Batch mode
        runs = discover_runs(runs_root)
        logger.info(f"Discovered {len(runs)} runs")

        # Filter to runs that have speaker_analysis.json
        eligible = [
            r for r in runs
            if (r.run_dir / "metadata" / args.speaker_analysis_file).exists()
        ]
        logger.info(f"{len(eligible)} runs have {args.speaker_analysis_file}")

        extracted = 0
        skipped = 0
        failed = 0

        for i, run_info in enumerate(eligible):
            label = f"[{i + 1}/{len(eligible)}]"
            try:
                output_dir = _build_output_dir(run_info, output_root)

                if not args.force and not args.dry_run and _has_existing_output(output_dir):
                    logger.info(
                        f"{label} SKIPPED (exists): "
                        f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}"
                    )
                    skipped += 1
                    continue

                result = extract_patient_audio(
                    run_dir=run_info.run_dir,
                    output_dir=output_dir,
                    quality_threshold=args.quality_threshold,
                    top_quality_fraction=args.top_quality_fraction,
                    selection_mode=args.selection_mode,
                    max_total_duration_s=max_total_duration_s,
                    min_segment_duration_s=args.min_segment_duration_s,
                    max_segment_duration_s=args.max_segment_duration_s,
                    output_sample_rate=args.resample_to,
                    group_consecutive=group_consecutive,
                    max_group_gap_s=args.max_gap_s,
                    acoustic_post_filter=args.acoustic_post_filter,
                    acoustic_min_speech_band_ratio=args.acoustic_min_speech_band_ratio,
                    acoustic_min_voiced_ratio=args.acoustic_min_voiced_ratio,
                    acoustic_max_music_score=args.acoustic_max_music_score,
                    acoustic_min_speaker_consistency=args.acoustic_min_speaker_consistency,
                    acoustic_min_subject_similarity=args.acoustic_min_subject_similarity,
                    denoise_output=args.denoise_output,
                    denoise_strength=args.denoise_strength,
                    speaker_analysis_filename=args.speaker_analysis_file,
                    ignore_content_screening=args.ignore_content_screening,
                    dry_run=args.dry_run,
                )

                if result is None:
                    logger.info(
                        f"{label} SKIPPED: "
                        f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}"
                    )
                    skipped += 1
                else:
                    action = "DRY RUN" if args.dry_run else "EXTRACTED"
                    logger.info(
                        f"{label} {action}: "
                        f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem} "
                        f"- {result['clips_generated']} clips, {result['segments_included']} segs, "
                        f"{result['total_duration_s']:.1f}s"
                    )
                    extracted += 1
            except Exception as e:
                logger.error(
                    f"{label} FAILED: "
                    f"{run_info.person}/{run_info.timepoint}/{run_info.video_stem}: {e}",
                    exc_info=True,
                )
                failed += 1

        logger.info(
            f"Batch complete: {extracted} extracted, {skipped} skipped, {failed} failed"
        )


if __name__ == "__main__":
    main()
