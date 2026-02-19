from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunInfo:
    """A discovered pipeline run directory with parsed context."""
    run_dir: Path
    source: str          # "Dementia_raw_data" or "No_Dementia_raw_data"
    person: str          # e.g. "Alan Ramsey"
    timepoint: str       # e.g. "5_years"
    video_stem: str      # e.g. "Alan Ramsey speaks on election"
    transcript_path: Path
    words_path: Optional[Path]
    asr_info_path: Optional[Path]
    is_v2: bool


def _parse_folder_context(run_dir: Path, runs_root: Path) -> Optional[dict]:
    """Extract source/person/timepoint/video_stem from the folder hierarchy.

    Expected: runs_root / <source> / <person> / <timepoint> / <video_stem>
    """
    try:
        rel = run_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None

    # Optional grouping folder support, e.g.:
    # runs_root / No_Dementia_raw_data / missing_control / <person> / <timepoint> / <video_stem>
    if len(parts) >= 5 and parts[1] == "missing_control":
        return {
            "source": parts[0],
            "person": parts[2],
            "timepoint": parts[3],
            "video_stem": parts[4],
        }

    return {
        "source": parts[0],
        "person": parts[1],
        "timepoint": parts[2],
        "video_stem": parts[3],
    }


def _is_v2_run(meta_dir: Path) -> bool:
    """Check if a run was produced by the v2 pipeline."""
    asr_info = meta_dir / "asr_info.json"
    if asr_info.exists():
        return True
    run_json = meta_dir / "run.json"
    if run_json.exists():
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
            return data.get("pipeline_version", 1) >= 2
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: check if transcript segments have ASR confidence fields
    transcript = meta_dir / "transcript.json"
    if transcript.exists():
        try:
            data = json.loads(transcript.read_text(encoding="utf-8"))
            segs = data.get("segments", [])
            if segs and "avg_logprob" in segs[0]:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


def discover_single_run(run_dir: Path, runs_root: Path) -> Optional[RunInfo]:
    """Build RunInfo for a single run directory."""
    meta_dir = run_dir / "metadata"
    transcript = meta_dir / "transcript.json"
    if not transcript.exists():
        return None

    ctx = _parse_folder_context(run_dir, runs_root)
    if ctx is None:
        return None

    words = meta_dir / "words.json"
    asr_info = meta_dir / "asr_info.json"
    is_v2 = _is_v2_run(meta_dir)

    return RunInfo(
        run_dir=run_dir,
        source=ctx["source"],
        person=ctx["person"],
        timepoint=ctx["timepoint"],
        video_stem=ctx["video_stem"],
        transcript_path=transcript,
        words_path=words if words.exists() else None,
        asr_info_path=asr_info if asr_info.exists() else None,
        is_v2=is_v2,
    )


def discover_runs(runs_root: Path, v2_only: bool = False) -> List[RunInfo]:
    """Scan runs/ tree for processable run directories."""
    results: List[RunInfo] = []
    if not runs_root.exists():
        return results

    for transcript in runs_root.rglob("metadata/transcript.json"):
        run_dir = transcript.parent.parent
        info = discover_single_run(run_dir, runs_root)
        if info is None:
            continue
        if v2_only and not info.is_v2:
            continue
        results.append(info)

    results.sort(key=lambda r: (r.source, r.person, r.timepoint, r.video_stem))
    return results
