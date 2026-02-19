"""Summarize pyannote API progress and timing from run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


TERMINAL = {"succeeded", "failed", "canceled"}


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_iso_z(s: str) -> float | None:
    if not s:
        return None
    try:
        # Keep dependency-free parser using datetime from stdlib.
        from datetime import datetime

        s2 = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s2).timestamp()
    except Exception:
        return None


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Report pyannote progress across runs/")
    parser.add_argument("--runs-root", default="runs", help="Root runs directory")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    meta_dirs = [p for p in runs_root.rglob("metadata") if p.is_dir()]

    total_runs = 0
    with_audio = 0
    status_counts: Dict[str, int] = {}
    durations_s: List[float] = []
    elapsed_succeeded: List[float] = []
    elapsed_terminal: List[float] = []

    for meta in meta_dirs:
        run_dir = meta.parent
        total_runs += 1

        asr_info = _load_json(meta / "asr_info.json")
        if asr_info and "duration" in asr_info:
            with_audio += 1
            durations_s.append(_safe_float(asr_info.get("duration")))

        state = _load_json(meta / "pyannote_job.json")
        if not state:
            continue

        status = str(state.get("status", "unknown")).lower()
        status_counts[status] = status_counts.get(status, 0) + 1

        submitted_ts = _parse_iso_z(str(state.get("submitted_at", "")))
        completed_ts = _parse_iso_z(str(state.get("completed_at", "")))
        updated_ts = _parse_iso_z(str(state.get("updated_at", "")))

        elapsed = None
        if submitted_ts and completed_ts and completed_ts >= submitted_ts:
            elapsed = completed_ts - submitted_ts
        elif submitted_ts and updated_ts and updated_ts >= submitted_ts and status in TERMINAL:
            elapsed = updated_ts - submitted_ts

        if elapsed is not None and elapsed >= 0:
            if status == "succeeded":
                elapsed_succeeded.append(elapsed)
            if status in TERMINAL:
                elapsed_terminal.append(elapsed)

    succeeded = status_counts.get("succeeded", 0)
    running = status_counts.get("running", 0) + status_counts.get("created", 0) + status_counts.get("pending", 0)
    failed = status_counts.get("failed", 0)
    canceled = status_counts.get("canceled", 0)

    discovered = with_audio
    completed = succeeded + failed + canceled
    remaining = max(0, discovered - completed)

    avg_elapsed_succeeded = sum(elapsed_succeeded) / len(elapsed_succeeded) if elapsed_succeeded else 0.0
    avg_elapsed_terminal = sum(elapsed_terminal) / len(elapsed_terminal) if elapsed_terminal else 0.0

    # Rough ETA in wall-clock minutes using average terminal time/run.
    eta_remaining_s = remaining * (avg_elapsed_terminal or avg_elapsed_succeeded or 0.0)

    total_audio_s = sum(durations_s)
    total_audio_h = total_audio_s / 3600.0 if total_audio_s else 0.0

    report = {
        "runs_root": str(runs_root),
        "counts": {
            "metadata_dirs_found": total_runs,
            "runs_with_audio": discovered,
            "completed_terminal": completed,
            "remaining": remaining,
            "succeeded": succeeded,
            "running_or_created_or_pending": running,
            "failed": failed,
            "canceled": canceled,
        },
        "status_counts": status_counts,
        "timing_seconds": {
            "avg_elapsed_succeeded": round(avg_elapsed_succeeded, 3),
            "avg_elapsed_terminal": round(avg_elapsed_terminal, 3),
            "eta_remaining_seconds_estimate": round(eta_remaining_s, 3),
        },
        "audio": {
            "total_audio_seconds": round(total_audio_s, 3),
            "total_audio_hours": round(total_audio_h, 3),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out_json:
        out = Path(args.out_json)
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

