"""Batch-build overlap-transferred speaker-analysis files for all runs.

For each run:
1) Read existing LLM output from metadata/speaker_analysis.json (+ segments_detailed.json)
2) Select newest diarization timeline:
   - metadata/segments_detailed_api.json (preferred)
   - pyannote_results_local/.../metadata/diarization_local.json (fallback)
3) Project subject windows by timestamp overlap and write:
   metadata/speaker_analysis_overlap_selected.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transfer_speaker_analysis_overlap import (
    _build_subject_windows,
    _transfer_recommended,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_runs_with_audio(runs_root: Path) -> List[Path]:
    runs: List[Path] = []
    for p in runs_root.rglob("*"):
        if p.is_dir() and (p / "audio" / "audio_16k.wav").exists():
            runs.append(p)
    return runs


def _segments_from_segments_detailed(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    segs = payload.get("segments", [])
    if not isinstance(segs, list):
        return []
    out: List[Dict[str, Any]] = []
    for seg in segs:
        try:
            out.append(
                {
                    "id": int(seg.get("id")),
                    "start": float(seg.get("start")),
                    "end": float(seg.get("end")),
                    "speaker": seg.get("speaker"),
                    "text": str(seg.get("text", "") or ""),
                    "words": seg.get("words") or [],
                }
            )
        except Exception:
            continue
    return out


def _segments_from_local_diarization(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    segs = payload.get("segments", [])
    if not isinstance(segs, list):
        return []
    parsed: List[Dict[str, Any]] = []
    for seg in segs:
        try:
            parsed.append(
                {
                    "start": float(seg.get("start")),
                    "end": float(seg.get("end")),
                    "speaker": seg.get("speaker"),
                }
            )
        except Exception:
            continue
    parsed.sort(key=lambda s: (s["start"], s["end"]))
    out: List[Dict[str, Any]] = []
    for idx, seg in enumerate(parsed, start=1):
        out.append(
            {
                "id": idx,
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker"),
                "text": "",
                "words": [],
            }
        )
    return out


def _choose_target_segments(
    run_dir: Path,
    runs_root: Path,
    local_root: Path | None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    meta = run_dir / "metadata"
    api_path = meta / "segments_detailed_api.json"
    if api_path.exists():
        api_segs = _segments_from_segments_detailed(api_path)
        if api_segs:
            return api_segs, "api", str(api_path)

    if local_root is not None:
        rel = run_dir.relative_to(runs_root)
        local_path = local_root / rel / "metadata" / "diarization_local.json"
        if local_path.exists():
            local_segs = _segments_from_local_diarization(local_path)
            if local_segs:
                return local_segs, "local", str(local_path)

    return [], "none", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch overlap-transfer speaker analysis onto newer diarization.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--local-root", default="pyannote_results_local")
    parser.add_argument("--source-speaker-analysis", default="speaker_analysis.json")
    parser.add_argument("--source-segments", default="segments_detailed.json")
    parser.add_argument("--output-speaker-analysis", default="speaker_analysis_overlap_selected.json")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.50)
    parser.add_argument("--min-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--default-quality", type=float, default=0.75)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output-speaker-analysis.")
    parser.add_argument(
        "--report-prefix",
        default="helper_scripts/overlap_transfer_batch",
        help="Prefix for JSON/CSV report files.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.min_overlap_ratio <= 1.0):
        raise ValueError("--min-overlap-ratio must be in [0.0, 1.0]")
    if args.min_overlap_seconds < 0:
        raise ValueError("--min-overlap-seconds must be >= 0")
    if not (0.0 <= args.default_quality <= 1.0):
        raise ValueError("--default-quality must be in [0.0, 1.0]")

    runs_root = Path(args.runs_root).resolve()
    local_root = Path(args.local_root).resolve() if args.local_root else None
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_prefix = Path(args.report_prefix)
    report_json = Path(f"{report_prefix}_{now}.json")
    report_csv = Path(f"{report_prefix}_{now}.csv")
    report_json.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    counts = {
        "runs_discovered": 0,
        "written": 0,
        "skipped_exists": 0,
        "skipped_missing_source": 0,
        "skipped_no_windows": 0,
        "skipped_no_target": 0,
        "skipped_no_recommended": 0,
        "errors": 0,
    }

    for run_dir in _iter_runs_with_audio(runs_root):
        counts["runs_discovered"] += 1
        rel = run_dir.relative_to(runs_root).as_posix()
        meta = run_dir / "metadata"
        src_sa_path = meta / args.source_speaker_analysis
        src_seg_path = meta / args.source_segments
        out_sa_path = meta / args.output_speaker_analysis

        row: Dict[str, Any] = {
            "run": rel,
            "status": "",
            "target_source": "",
            "target_path": "",
            "source_windows": 0,
            "target_segments": 0,
            "recommended_out": 0,
            "error": "",
        }
        try:
            if out_sa_path.exists() and not args.force:
                counts["skipped_exists"] += 1
                row["status"] = "skipped_exists"
                rows.append(row)
                continue

            if not src_sa_path.exists() or not src_seg_path.exists():
                counts["skipped_missing_source"] += 1
                row["status"] = "skipped_missing_source"
                rows.append(row)
                continue

            src_sa = _load_json(src_sa_path)
            src_segments = _segments_from_segments_detailed(src_seg_path)
            windows = _build_subject_windows(
                speaker_analysis=src_sa,
                source_segments=src_segments,
                default_quality=args.default_quality,
            )
            if not windows:
                counts["skipped_no_windows"] += 1
                row["status"] = "skipped_no_windows"
                rows.append(row)
                continue

            target_segments, target_source, target_path = _choose_target_segments(
                run_dir=run_dir,
                runs_root=runs_root,
                local_root=local_root,
            )
            if not target_segments:
                counts["skipped_no_target"] += 1
                row["status"] = "skipped_no_target"
                rows.append(row)
                continue

            recommended, diagnostics = _transfer_recommended(
                windows=windows,
                target_segments=target_segments,
                min_overlap_ratio=args.min_overlap_ratio,
                min_overlap_seconds=args.min_overlap_seconds,
                default_quality=args.default_quality,
            )
            if not recommended:
                counts["skipped_no_recommended"] += 1
                row["status"] = "skipped_no_recommended"
                row["target_source"] = target_source
                row["target_path"] = target_path
                row["source_windows"] = len(windows)
                row["target_segments"] = len(target_segments)
                rows.append(row)
                continue

            out_sa = dict(src_sa)
            out_sa["pipeline_stage"] = "speaker_classification_overlap_transfer"
            out_sa["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            out_sa["recommended_segments"] = recommended
            out_sa["overlap_transfer"] = {
                "source_speaker_analysis": str(src_sa_path),
                "source_segments": str(src_seg_path),
                "target_source": target_source,
                "target_path": target_path,
                "window_count": len(windows),
                "target_segment_count": len(target_segments),
                "recommended_count": len(recommended),
                "min_overlap_ratio": args.min_overlap_ratio,
                "min_overlap_seconds": args.min_overlap_seconds,
                "default_quality": args.default_quality,
                "diagnostics": diagnostics,
            }
            out_sa.setdefault("notes", {})
            if isinstance(out_sa["notes"], dict):
                out_sa["notes"]["overlap_transfer_applied"] = True
                out_sa["notes"]["overlap_target_source"] = target_source

            _write_json(out_sa_path, out_sa)
            counts["written"] += 1
            row["status"] = "written"
            row["target_source"] = target_source
            row["target_path"] = target_path
            row["source_windows"] = len(windows)
            row["target_segments"] = len(target_segments)
            row["recommended_out"] = len(recommended)
            rows.append(row)
        except Exception as exc:
            counts["errors"] += 1
            row["status"] = "error"
            row["error"] = str(exc)
            rows.append(row)

    # Write reports.
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "status",
                "target_source",
                "target_path",
                "source_windows",
                "target_segments",
                "recommended_out",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_root": str(runs_root),
        "local_root": str(local_root) if local_root else None,
        "output_speaker_analysis": args.output_speaker_analysis,
        "counts": counts,
        "report_csv": str(report_csv.resolve()),
        "rows": rows,
    }
    _write_json(report_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

