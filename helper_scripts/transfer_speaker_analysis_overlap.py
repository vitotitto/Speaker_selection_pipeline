"""Transfer existing LLM subject decisions onto a new diarization timeline.

This avoids re-running speaker-analysis LLM calls when new diarization segments
arrive from API/local models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _as_segments(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        segs = payload.get("segments", [])
        return segs if isinstance(segs, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _quality_from_confidence(conf: float) -> float:
    conf = max(0.0, min(1.0, float(conf)))
    # Map classification confidence to the quality-score range used downstream.
    return 0.55 + 0.40 * conf


def _build_subject_windows(
    speaker_analysis: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    default_quality: float,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    seg_by_id = {}
    for seg in source_segments:
        try:
            seg_by_id[int(seg.get("id"))] = seg
        except Exception:
            continue

    recommended = speaker_analysis.get("recommended_segments") or []
    for seg in recommended:
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        try:
            quality = float(seg.get("quality_score", default_quality))
        except Exception:
            quality = default_quality
        windows.append(
            {
                "start": start,
                "end": end,
                "quality_score": quality,
                "text": str(seg.get("text", "") or "").strip(),
                "segment_id": int(seg.get("segment_id", -1)),
                "source": "recommended_segments",
            }
        )

    if windows:
        return windows

    # Fallback path if recommended segments are missing:
    # infer subject windows from segment_classifications + source segments.
    for cls in speaker_analysis.get("segment_classifications") or []:
        if str(cls.get("speaker", "")).lower() != "subject":
            continue
        try:
            sid = int(cls.get("segment_id"))
        except Exception:
            continue
        src = seg_by_id.get(sid)
        if not src:
            continue
        try:
            start = float(src.get("start"))
            end = float(src.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        conf = float(cls.get("confidence", 0.8))
        windows.append(
            {
                "start": start,
                "end": end,
                "quality_score": _quality_from_confidence(conf),
                "text": str(src.get("text", "") or "").strip(),
                "segment_id": sid,
                "source": "segment_classifications",
            }
        )
    return windows


def _transfer_recommended(
    windows: List[Dict[str, Any]],
    target_segments: List[Dict[str, Any]],
    min_overlap_ratio: float,
    min_overlap_seconds: float,
    default_quality: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for t in target_segments:
        try:
            seg_id = int(t.get("id"))
            start = float(t.get("start"))
            end = float(t.get("end"))
        except Exception:
            continue
        duration = end - start
        if duration <= 0:
            continue

        hits: List[Tuple[Dict[str, Any], float]] = []
        overlap_total = 0.0
        for w in windows:
            ov = _overlap_s(start, end, float(w["start"]), float(w["end"]))
            if ov <= 0:
                continue
            hits.append((w, ov))
            overlap_total += ov

        overlap_ratio = overlap_total / duration if duration > 0 else 0.0
        passed = overlap_total >= min_overlap_seconds and overlap_ratio >= min_overlap_ratio

        diagnostics.append(
            {
                "segment_id": seg_id,
                "start": start,
                "end": end,
                "duration": duration,
                "overlap_seconds": round(overlap_total, 4),
                "overlap_ratio": round(overlap_ratio, 4),
                "matched_source_segments": sorted(
                    {int(w.get("segment_id", -1)) for (w, _) in hits if int(w.get("segment_id", -1)) >= 0}
                ),
                "passed": passed,
            }
        )

        if not passed:
            continue

        if hits:
            weighted_quality = sum(float(w["quality_score"]) * ov for (w, ov) in hits) / max(overlap_total, 1e-9)
            weighted_quality = max(0.0, min(1.0, weighted_quality))
        else:
            weighted_quality = default_quality

        tgt_text = str(t.get("text", "") or "").strip()
        if tgt_text:
            text = tgt_text
        else:
            snippets = []
            seen = set()
            for w, _ov in sorted(hits, key=lambda x: x[1], reverse=True):
                st = str(w.get("text", "") or "").strip()
                if st and st not in seen:
                    snippets.append(st)
                    seen.add(st)
            text = " ".join(snippets).strip()

        out.append(
            {
                "segment_id": seg_id,
                "start": round(start, 4),
                "end": round(end, 4),
                "duration": round(duration, 4),
                "text": text,
                "quality_score": round(weighted_quality, 4),
            }
        )

    # Keep deterministic ordering.
    out.sort(key=lambda s: (float(s.get("start", 0.0)), int(s.get("segment_id", 0))))
    return out, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project existing LLM speaker-analysis decisions onto a new segment timeline."
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory (contains metadata/)")
    parser.add_argument("--source-speaker-analysis", default="metadata/speaker_analysis.json")
    parser.add_argument("--source-segments", default="metadata/segments_detailed.json")
    parser.add_argument("--target-segments", default="metadata/segments_detailed_api.json")
    parser.add_argument("--output-speaker-analysis", default="metadata/speaker_analysis_overlap.json")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.50)
    parser.add_argument("--min-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--default-quality", type=float, default=0.75)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Replace metadata/speaker_analysis.json with overlap output (backup original first).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.min_overlap_ratio <= 1.0):
        raise ValueError("--min-overlap-ratio must be in [0.0, 1.0]")
    if args.min_overlap_seconds < 0:
        raise ValueError("--min-overlap-seconds must be >= 0")
    if not (0.0 <= args.default_quality <= 1.0):
        raise ValueError("--default-quality must be in [0.0, 1.0]")

    run_dir = Path(args.run_dir).resolve()
    source_sa_path = (run_dir / args.source_speaker_analysis).resolve()
    source_segments_path = (run_dir / args.source_segments).resolve()
    target_segments_path = (run_dir / args.target_segments).resolve()
    output_sa_path = (run_dir / args.output_speaker_analysis).resolve()

    if not source_sa_path.exists():
        raise FileNotFoundError(f"Missing speaker-analysis input: {source_sa_path}")
    if not source_segments_path.exists():
        raise FileNotFoundError(f"Missing source segments input: {source_segments_path}")
    if not target_segments_path.exists():
        raise FileNotFoundError(f"Missing target segments input: {target_segments_path}")

    source_sa = _load_json(source_sa_path)
    source_segments_payload = _load_json(source_segments_path)
    target_segments_payload = _load_json(target_segments_path)
    source_segments = _as_segments(source_segments_payload)
    target_segments = _as_segments(target_segments_payload)

    windows = _build_subject_windows(
        speaker_analysis=source_sa,
        source_segments=source_segments,
        default_quality=args.default_quality,
    )
    if not windows:
        raise RuntimeError("No subject windows found in source speaker-analysis.")

    recommended, diagnostics = _transfer_recommended(
        windows=windows,
        target_segments=target_segments,
        min_overlap_ratio=args.min_overlap_ratio,
        min_overlap_seconds=args.min_overlap_seconds,
        default_quality=args.default_quality,
    )

    out_sa = dict(source_sa)
    out_sa["pipeline_stage"] = "speaker_classification_overlap_transfer"
    out_sa["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out_sa["recommended_segments"] = recommended
    out_sa["overlap_transfer"] = {
        "source_speaker_analysis": str(source_sa_path),
        "source_segments": str(source_segments_path),
        "target_segments": str(target_segments_path),
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
    _write_json(output_sa_path, out_sa)

    activated = False
    backup_path = None
    if args.activate:
        live_sa_path = run_dir / "metadata" / "speaker_analysis.json"
        backup_path = run_dir / "metadata" / "speaker_analysis.pre_overlap_backup.json"
        if not backup_path.exists():
            backup_path.write_text(live_sa_path.read_text(encoding="utf-8"), encoding="utf-8")
        live_sa_path.write_text(output_sa_path.read_text(encoding="utf-8"), encoding="utf-8")
        activated = True

    summary = {
        "run_dir": str(run_dir),
        "source_speaker_analysis": str(source_sa_path),
        "source_segments": str(source_segments_path),
        "target_segments": str(target_segments_path),
        "output_speaker_analysis": str(output_sa_path),
        "windows_used": len(windows),
        "target_segments": len(target_segments),
        "recommended_segments_transferred": len(recommended),
        "activated": activated,
        "backup_path": str(backup_path) if backup_path else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

