"""Stage 3b: Build per-person speech budgets from existing LLM + acoustic scores.

This script does NOT rerun any LLM stages. It reuses speaker_analysis outputs and
applies lightweight acoustic analysis to candidates, then selects up to N minutes
per person across all their runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from audio_extraction.extractor import extract_patient_audio
from speaker_analysis.discovery import discover_runs, discover_single_run


def _build_output_dir(run_info, output_root: Path) -> Path:
    return output_root / run_info.source / run_info.person / run_info.timepoint / run_info.video_stem


def _has_existing_output(output_dir: Path) -> bool:
    legacy_file = output_dir / "patient_speech.wav"
    has_legacy = legacy_file.exists()
    has_parts = any(output_dir.glob("patient_speech_part_*.wav"))
    has_manifest = (output_dir / "extraction_manifest.json").exists()
    return has_legacy or (has_manifest and has_parts)


def _configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class _ConsoleSafeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.getMessage().encode("cp1252")
        except UnicodeEncodeError:
            record.msg = record.getMessage().encode("ascii", "backslashreplace").decode("ascii")
            record.args = ()
        return True


def _segment_score(seg: Dict[str, Any], diag: Dict[str, Any]) -> Dict[str, float]:
    llm_quality = float(seg.get("quality_score", 0.0))
    subject_similarity = float(diag.get("subject_similarity", 1.0))
    metrics = diag.get("metrics", {}) if isinstance(diag, dict) else {}
    speech_band_ratio = float(metrics.get("speech_band_ratio", 1.0))
    voiced_ratio = float(metrics.get("voiced_ratio", 1.0))
    speaker_consistency = float(metrics.get("speaker_consistency", 1.0))
    music_score = float(metrics.get("music_score", 0.0))

    # Weighted blend: reward strong LLM + clean/acoustically speech-like segments.
    combined = (
        0.62 * llm_quality
        + 0.14 * subject_similarity
        + 0.09 * speech_band_ratio
        + 0.08 * voiced_ratio
        + 0.07 * speaker_consistency
        - 0.22 * music_score
    )
    return {
        "llm_quality": round(llm_quality, 6),
        "subject_similarity": round(subject_similarity, 6),
        "speech_band_ratio": round(speech_band_ratio, 6),
        "voiced_ratio": round(voiced_ratio, 6),
        "speaker_consistency": round(speaker_consistency, 6),
        "music_score": round(music_score, 6),
        "combined_score": round(combined, 6),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _configure_console_utf8()

    parser = argparse.ArgumentParser(
        description="Build per-person speech budgets from existing analysis outputs."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch", action="store_true", help="Process all runs")
    mode.add_argument("--run-dir", type=str, help="Process a single run")

    parser.add_argument("--runs-root", type=str, default=None, help="Runs root (default: runs/)")
    parser.add_argument("--output-root", type=str, default=None, help="Output root (default: processed_3/)")
    parser.add_argument("--target-minutes-per-person", type=float, default=5.0, help="Per-person target duration in minutes.")
    parser.add_argument("--quality-threshold", type=float, default=0.7, help="Minimum LLM quality score.")
    parser.add_argument("--top-quality-fraction", type=float, default=0.6, help="Keep top fraction of segments per run before person-budgeting.")
    parser.add_argument("--min-segment-duration-s", type=float, default=4.0)
    parser.add_argument("--max-segment-duration-s", type=float, default=None)
    parser.add_argument("--split-segments", action="store_true")
    parser.add_argument("--max-gap-s", type=float, default=0.75)
    parser.add_argument("--resample-to", type=int, default=None)

    parser.add_argument("--acoustic-post-filter", action="store_true", help="Apply acoustic filtering in candidate stage.")
    parser.add_argument("--acoustic-min-speech-band-ratio", type=float, default=0.38)
    parser.add_argument("--acoustic-min-voiced-ratio", type=float, default=0.28)
    parser.add_argument("--acoustic-max-music-score", type=float, default=0.60)
    parser.add_argument("--acoustic-min-speaker-consistency", type=float, default=0.60)
    parser.add_argument("--acoustic-min-subject-similarity", type=float, default=0.45)

    parser.add_argument("--denoise-output", action="store_true")
    parser.add_argument("--denoise-strength", type=float, default=0.65)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.target_minutes_per_person <= 0:
        parser.error("--target-minutes-per-person must be > 0")
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
    if args.resample_to is not None and args.resample_to <= 0:
        parser.error("--resample-to must be > 0")
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
    file_handler = logging.FileHandler("audio_extraction_budget.log", encoding="utf-8")
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
    output_root = Path(args.output_root) if args.output_root else project_root / "processed_3"
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
    target_seconds = args.target_minutes_per_person * 60.0
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
        one = discover_single_run(run_dir, runs_root)
        if one is None:
            logger.error(f"Invalid run dir: {run_dir}")
            return
        runs = [one]
    else:
        runs = discover_runs(runs_root)
    eligible = [r for r in runs if (r.run_dir / "metadata" / "speaker_analysis.json").exists()]
    logger.info(f"Discovered {len(runs)} runs, {len(eligible)} eligible (speaker_analysis.json).")

    # Pass 1: collect candidate segments from existing LLM + acoustic outputs.
    all_candidates: List[Dict[str, Any]] = []
    candidate_manifest_count = 0
    candidate_skipped = 0
    for i, run_info in enumerate(eligible, start=1):
        dummy_output = output_root / "_candidate_tmp" / run_info.source / run_info.person / run_info.timepoint / run_info.video_stem
        manifest = extract_patient_audio(
            run_dir=run_info.run_dir,
            output_dir=dummy_output,
            quality_threshold=args.quality_threshold,
            top_quality_fraction=args.top_quality_fraction,
            max_total_duration_s=None,
            min_segment_duration_s=args.min_segment_duration_s,
            max_segment_duration_s=args.max_segment_duration_s,
            output_sample_rate=args.resample_to,
            group_consecutive=group_consecutive,
            max_group_gap_s=args.max_gap_s,
            dry_run=True,
            acoustic_post_filter=args.acoustic_post_filter,
            acoustic_min_speech_band_ratio=args.acoustic_min_speech_band_ratio,
            acoustic_min_voiced_ratio=args.acoustic_min_voiced_ratio,
            acoustic_max_music_score=args.acoustic_max_music_score,
            acoustic_min_speaker_consistency=args.acoustic_min_speaker_consistency,
            acoustic_min_subject_similarity=args.acoustic_min_subject_similarity,
            denoise_output=False,
            denoise_strength=args.denoise_strength,
        )
        if manifest is None:
            candidate_skipped += 1
            logger.info(f"[{i}/{len(eligible)}] candidate-skip: {run_info.person}/{run_info.timepoint}/{run_info.video_stem}")
            continue

        candidate_manifest_count += 1
        diag_by_id = {
            int(d.get("segment_id", -1)): d
            for d in manifest.get("acoustic_diagnostics", [])
            if isinstance(d, dict)
        }
        person_key = f"{run_info.source}/{run_info.person}"
        for seg in manifest.get("segments", []):
            sid = int(seg.get("segment_id", -1))
            diag = diag_by_id.get(sid, {})
            score_parts = _segment_score(seg, diag)
            all_candidates.append({
                "person_key": person_key,
                "source": run_info.source,
                "person": run_info.person,
                "timepoint": run_info.timepoint,
                "video_stem": run_info.video_stem,
                "run_dir": str(run_info.run_dir),
                "segment_id": sid,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "duration": float(seg.get("duration", 0.0)),
                "text": str(seg.get("text", "")),
                **score_parts,
                "acoustic_reasons": ";".join(diag.get("reasons", [])) if isinstance(diag, dict) else "",
                "acoustic_passed": bool(diag.get("passed", True)) if isinstance(diag, dict) else True,
            })

    logger.info(
        f"Candidate pass complete: {candidate_manifest_count} runs yielded candidates, "
        f"{candidate_skipped} runs skipped, {len(all_candidates)} candidate segments."
    )

    # Pass 2: select up to target_seconds per person globally.
    by_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_candidates:
        by_person[row["person_key"]].append(row)

    selected_ids_by_run: Dict[str, set[int]] = defaultdict(set)
    decision_rows: List[Dict[str, Any]] = []
    person_summary: List[Dict[str, Any]] = []
    for person_key, rows in by_person.items():
        rows.sort(
            key=lambda r: (float(r["combined_score"]), float(r["llm_quality"]), float(r["duration"])),
            reverse=True,
        )
        used = 0.0
        selected_n = 0
        for r in rows:
            take = used + float(r["duration"]) <= target_seconds + 1e-9
            reason = "selected" if take else "budget_limit"
            if take:
                used += float(r["duration"])
                selected_n += 1
                selected_ids_by_run[r["run_dir"]].add(int(r["segment_id"]))
            out = dict(r)
            out["decision"] = reason
            out["person_cumulative_selected_s"] = round(used, 4)
            decision_rows.append(out)

        source, person = person_key.split("/", 1)
        person_summary.append({
            "person_key": person_key,
            "source": source,
            "person": person,
            "candidate_segments": len(rows),
            "selected_segments": selected_n,
            "selected_duration_s": round(used, 4),
            "target_duration_s": round(target_seconds, 4),
        })

    # Pass 3: export selected segments into processed_3.
    extracted = 0
    skipped = 0
    failed = 0
    for i, run_info in enumerate(eligible, start=1):
        selected_ids = sorted(selected_ids_by_run.get(str(run_info.run_dir), set()))
        label = f"[{i}/{len(eligible)}]"
        if not selected_ids:
            logger.info(f"{label} SKIPPED (no person-budget selection): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}")
            skipped += 1
            continue

        output_dir = _build_output_dir(run_info, output_root)
        if not args.force and not args.dry_run and _has_existing_output(output_dir):
            logger.info(f"{label} SKIPPED (exists): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}")
            skipped += 1
            continue

        try:
            result = extract_patient_audio(
                run_dir=run_info.run_dir,
                output_dir=output_dir,
                quality_threshold=args.quality_threshold,
                top_quality_fraction=args.top_quality_fraction,
                max_total_duration_s=None,
                min_segment_duration_s=args.min_segment_duration_s,
                max_segment_duration_s=args.max_segment_duration_s,
                output_sample_rate=args.resample_to,
                group_consecutive=group_consecutive,
                max_group_gap_s=args.max_gap_s,
                dry_run=args.dry_run,
                acoustic_post_filter=False,  # already accounted in candidate stage
                denoise_output=args.denoise_output,
                denoise_strength=args.denoise_strength,
                segment_id_whitelist=selected_ids,
            )
            if result is None:
                logger.info(f"{label} SKIPPED (no usable output): {run_info.person}/{run_info.timepoint}/{run_info.video_stem}")
                skipped += 1
            else:
                action = "DRY RUN" if args.dry_run else "EXTRACTED"
                logger.info(
                    f"{label} {action}: {run_info.person}/{run_info.timepoint}/{run_info.video_stem} - "
                    f"{result['clips_generated']} clips, {result['segments_included']} segs, {result['total_duration_s']:.1f}s"
                )
                extracted += 1
        except Exception as exc:
            logger.error(f"{label} FAILED: {run_info.person}/{run_info.timepoint}/{run_info.video_stem}: {exc}", exc_info=True)
            failed += 1

    # Save selection accounting reports.
    reports_dir = output_root / "_selection_reports"
    cfg = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "output_root": str(output_root),
        "target_minutes_per_person": args.target_minutes_per_person,
        "quality_threshold": args.quality_threshold,
        "top_quality_fraction": args.top_quality_fraction,
        "min_segment_duration_s": args.min_segment_duration_s,
        "max_segment_duration_s": args.max_segment_duration_s,
        "acoustic_post_filter": args.acoustic_post_filter,
        "acoustic_thresholds": {
            "min_speech_band_ratio": args.acoustic_min_speech_band_ratio,
            "min_voiced_ratio": args.acoustic_min_voiced_ratio,
            "max_music_score": args.acoustic_max_music_score,
            "min_speaker_consistency": args.acoustic_min_speaker_consistency,
            "min_subject_similarity": args.acoustic_min_subject_similarity,
        },
        "denoise_output": args.denoise_output,
        "denoise_strength": args.denoise_strength,
        "summary": {
            "eligible_runs": len(eligible),
            "candidate_runs_with_segments": candidate_manifest_count,
            "candidate_runs_skipped": candidate_skipped,
            "candidate_segments_total": len(all_candidates),
            "export_extracted": extracted,
            "export_skipped": skipped,
            "export_failed": failed,
        },
    }
    _write_json(reports_dir / "selection_config.json", cfg)
    _write_json(reports_dir / "person_budget_summary.json", sorted(person_summary, key=lambda x: x["person_key"]))
    run_selection_map = {
        run_dir: sorted(list(ids))
        for run_dir, ids in selected_ids_by_run.items()
        if ids
    }
    _write_json(reports_dir / "run_segment_selection_map.json", run_selection_map)
    _write_jsonl(reports_dir / "segment_decisions.jsonl", decision_rows)
    _write_csv(reports_dir / "segment_decisions.csv", decision_rows)

    logger.info(
        f"Budget extraction complete: extracted={extracted}, skipped={skipped}, failed={failed}. "
        f"Reports -> {reports_dir}"
    )


if __name__ == "__main__":
    main()
