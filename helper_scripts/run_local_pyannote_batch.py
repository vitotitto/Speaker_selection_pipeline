"""Batch local pyannote diarization with live progress tracking."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import soundfile as sf

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ACTIVE_CLOUD_STATUSES = {"created", "running", "pending", "processing"}
TERMINAL_CLOUD_STATUSES = {"succeeded", "failed", "canceled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RunTarget:
    run_dir: Path
    rel_path: str
    audio_path: Path
    audio_duration_s: float
    metadata_dir: Path
    output_dir: Path


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_annotation(diarization_output: Any) -> Any:
    if hasattr(diarization_output, "itertracks"):
        return diarization_output
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    if isinstance(diarization_output, dict) and "speaker_diarization" in diarization_output:
        return diarization_output["speaker_diarization"]
    raise TypeError(f"Unsupported diarization output type: {type(diarization_output)!r}")


def _iter_turns(annotation: Any) -> Iterator[Tuple[Any, str]]:
    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            yield segment, str(speaker)
        return
    if hasattr(annotation, "__iter__"):
        for item in annotation:
            if not isinstance(item, tuple):
                continue
            if len(item) == 2:
                turn, speaker = item
                yield turn, str(speaker)
            elif len(item) == 3:
                turn, _, speaker = item
                yield turn, str(speaker)
        return
    raise TypeError(f"Unsupported diarization annotation type: {type(annotation)!r}")


def _collect_segments(diarization_output: Any) -> List[Dict[str, Any]]:
    annotation = _as_annotation(diarization_output)
    segments: List[Dict[str, Any]] = []
    for segment, speaker in _iter_turns(annotation):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end - segment.start),
                "speaker": str(speaker),
            }
        )
    segments.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return segments


def _pick_audio(run_dir: Path, preferred_audio_file: str) -> Optional[Path]:
    candidates = [preferred_audio_file, "audio/audio_16k.wav", "audio/audio_base.wav"]
    seen: Set[str] = set()
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        p = run_dir / rel
        if p.exists() and p.is_file():
            return p
    return None


def _run_rel_path(run_dir: Path, runs_root: Path) -> Optional[str]:
    try:
        rel = run_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return None
    if len(rel.parts) < 4:
        return None
    return rel.as_posix()


def _load_run_list(paths: List[str]) -> Set[str]:
    runs: Set[str] = set()
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                runs.add(str(Path(s).resolve()))
            except Exception:
                runs.add(s)
    return runs


def _discover_targets(
    runs_root: Path,
    output_root: Path,
    preferred_audio_file: str,
    skip_if_cloud_submitted: bool,
    skip_existing_local: bool,
    include_runs_abs: Set[str],
    excluded_runs_abs: Set[str],
) -> Tuple[List[RunTarget], Dict[str, int]]:
    stats = {
        "discovered_run_dirs": 0,
        "excluded_non_context": 0,
        "excluded_not_included": 0,
        "excluded_no_audio": 0,
        "excluded_by_list": 0,
        "excluded_cloud_submitted": 0,
        "excluded_local_exists": 0,
    }
    targets: List[RunTarget] = []
    seen: Set[Path] = set()

    for meta_dir in runs_root.rglob("metadata"):
        if not meta_dir.is_dir():
            continue
        run_dir = meta_dir.parent.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        stats["discovered_run_dirs"] += 1

        rel = _run_rel_path(run_dir, runs_root)
        if rel is None:
            stats["excluded_non_context"] += 1
            continue

        if include_runs_abs and str(run_dir) not in include_runs_abs:
            stats["excluded_not_included"] += 1
            continue

        if str(run_dir) in excluded_runs_abs:
            stats["excluded_by_list"] += 1
            continue

        if skip_if_cloud_submitted:
            state = _read_json(meta_dir / "pyannote_job.json") or {}
            cloud_status = str(state.get("status", "")).lower()
            if cloud_status in ACTIVE_CLOUD_STATUSES or cloud_status in TERMINAL_CLOUD_STATUSES:
                stats["excluded_cloud_submitted"] += 1
                continue

        audio_path = _pick_audio(run_dir, preferred_audio_file)
        if audio_path is None:
            stats["excluded_no_audio"] += 1
            continue

        out_dir = output_root / rel / "metadata"
        if skip_existing_local:
            if (out_dir / "diarization_local.json").exists() and (out_dir / "timing_local.json").exists():
                stats["excluded_local_exists"] += 1
                continue

        try:
            info = sf.info(str(audio_path))
            duration_s = float(info.frames) / float(info.samplerate)
        except Exception:
            stats["excluded_no_audio"] += 1
            continue

        targets.append(
            RunTarget(
                run_dir=run_dir,
                rel_path=rel,
                audio_path=audio_path,
                audio_duration_s=duration_s,
                metadata_dir=meta_dir,
                output_dir=out_dir,
            )
        )

    targets.sort(key=lambda t: t.rel_path.lower())
    return targets, stats


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "timestamp",
        "run_rel",
        "audio_path",
        "audio_duration_s",
        "status",
        "inference_seconds",
        "load_seconds",
        "wall_seconds",
        "real_time_factor",
        "speed_x_realtime",
        "num_segments",
        "error",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _progress_payload(
    *,
    started_at: float,
    targets: List[RunTarget],
    done: int,
    succeeded: int,
    failed: int,
    skipped: int,
    infer_seconds_sum: float,
    audio_processed_s: float,
    current_run: Optional[str],
    discovery_stats: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    elapsed = max(0.0, time.time() - started_at)
    remaining = max(0, len(targets) - done)
    speed_audio_per_wall = (audio_processed_s / elapsed) if elapsed > 0 else 0.0
    eta_seconds = (sum(t.audio_duration_s for t in targets[done:]) / speed_audio_per_wall) if speed_audio_per_wall > 0 else None
    avg_rtf = (infer_seconds_sum / audio_processed_s) if audio_processed_s > 0 else None

    payload: Dict[str, Any] = {
        "updated_at": utc_now(),
        "summary": {
            "total_selected_runs": len(targets),
            "done_runs": done,
            "remaining_runs": remaining,
            "succeeded_runs": succeeded,
            "failed_runs": failed,
            "skipped_runs": skipped,
            "elapsed_seconds": round(elapsed, 3),
            "processed_audio_hours": round(audio_processed_s / 3600.0, 3),
            "remaining_audio_hours_estimate": round(sum(t.audio_duration_s for t in targets[done:]) / 3600.0, 3),
            "avg_real_time_factor": round(avg_rtf, 6) if avg_rtf is not None else None,
            "throughput_x_realtime_end_to_end": round(speed_audio_per_wall, 6),
            "eta_seconds_estimate": round(eta_seconds, 1) if eta_seconds is not None else None,
        },
        "current_run": current_run,
        "discovery_stats": discovery_stats,
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch local pyannote diarization with progress tracking.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="pyannote_results_local")
    parser.add_argument("--audio-file", default="audio/audio_16k.wav")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-audio-hours", type=float, default=None)
    parser.add_argument("--skip-if-cloud-submitted", action="store_true", default=True)
    parser.add_argument("--no-skip-if-cloud-submitted", dest="skip_if_cloud_submitted", action="store_false")
    parser.add_argument("--skip-existing-local", action="store_true", default=True)
    parser.add_argument("--no-skip-existing-local", dest="skip_existing_local", action="store_false")
    parser.add_argument("--include-run-list-file", action="append", default=[])
    parser.add_argument("--exclude-run-list-file", action="append", default=[])
    parser.add_argument("--progress-json", default="helper_scripts/local_pyannote_progress_live.json")
    parser.add_argument("--progress-csv", default="helper_scripts/local_pyannote_progress.csv")
    parser.add_argument("--selection-report", default="helper_scripts/local_pyannote_selection.json")
    parser.add_argument("--log-file", default="local_pyannote_batch.log")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    token = os.getenv(args.hf_token_env) or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            f"No HuggingFace token found. Set {args.hf_token_env} or HUGGINGFACE_TOKEN."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("local_pyannote")

    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    progress_json = Path(args.progress_json).resolve()
    progress_csv = Path(args.progress_csv).resolve()
    selection_report = Path(args.selection_report).resolve()

    include_runs_abs = _load_run_list(args.include_run_list_file)
    excluded_runs_abs = _load_run_list(args.exclude_run_list_file)
    targets, discovery_stats = _discover_targets(
        runs_root=runs_root,
        output_root=output_root,
        preferred_audio_file=args.audio_file,
        skip_if_cloud_submitted=args.skip_if_cloud_submitted,
        skip_existing_local=args.skip_existing_local,
        include_runs_abs=include_runs_abs,
        excluded_runs_abs=excluded_runs_abs,
    )

    if args.max_audio_hours is not None and args.max_audio_hours > 0:
        selected: List[RunTarget] = []
        acc = 0.0
        for t in targets:
            if selected and (acc + t.audio_duration_s) > args.max_audio_hours * 3600.0:
                break
            selected.append(t)
            acc += t.audio_duration_s
        targets = selected

    if args.max_runs is not None and args.max_runs >= 0:
        targets = targets[: args.max_runs]

    total_audio_h = sum(t.audio_duration_s for t in targets) / 3600.0
    logger.info("Discovered %s run dirs. Selected %s runs (%.3f hours).", discovery_stats["discovered_run_dirs"], len(targets), total_audio_h)
    logger.info("Discovery exclusions: %s", discovery_stats)

    selection_report.parent.mkdir(parents=True, exist_ok=True)
    selection_report.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "runs_root": str(runs_root),
                "output_root": str(output_root),
                "selected_runs": len(targets),
                "selected_audio_hours": round(total_audio_h, 3),
                "discovery_stats": discovery_stats,
                "excluded_runs_from_list_count": len(excluded_runs_abs),
                "targets": [
                    {
                        "run_rel": t.rel_path,
                        "run_dir": str(t.run_dir),
                        "audio_path": str(t.audio_path),
                        "audio_duration_s": round(t.audio_duration_s, 3),
                        "output_dir": str(t.output_dir),
                    }
                    for t in targets
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not targets:
        payload = _progress_payload(
            started_at=time.time(),
            targets=[],
            done=0,
            succeeded=0,
            failed=0,
            skipped=0,
            infer_seconds_sum=0.0,
            audio_processed_s=0.0,
            current_run=None,
            discovery_stats=discovery_stats,
            extra={"status": "nothing_to_process"},
        )
        _write_json(progress_json, payload)
        logger.info("Nothing to process.")
        return

    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyannote.audio + torch stack is not available in this environment.") from exc

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _: None  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model %s on device=%s", args.model, device)
    t_load0 = time.time()
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if device == "cuda":
        pipeline.to(torch.device("cuda"))
    model_load_seconds = time.time() - t_load0
    logger.info("Model loaded in %.2fs", model_load_seconds)

    diarize_kwargs: Dict[str, Any] = {}
    if args.min_speakers is not None:
        diarize_kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diarize_kwargs["max_speakers"] = args.max_speakers

    started_at = time.time()
    done = 0
    succeeded = 0
    failed = 0
    skipped = 0
    infer_seconds_sum = 0.0
    audio_processed_s = 0.0

    _write_json(
        progress_json,
        _progress_payload(
            started_at=started_at,
            targets=targets,
            done=done,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            infer_seconds_sum=infer_seconds_sum,
            audio_processed_s=audio_processed_s,
            current_run=None,
            discovery_stats=discovery_stats,
            extra={"status": "running", "model": args.model, "device": device, "model_load_seconds": round(model_load_seconds, 3)},
        ),
    )

    for idx, target in enumerate(targets, start=1):
        logger.info("[%s/%s] START %s", idx, len(targets), target.rel_path)
        current_run = target.rel_path
        _write_json(
            progress_json,
            _progress_payload(
                started_at=started_at,
                targets=targets,
                done=done,
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                infer_seconds_sum=infer_seconds_sum,
                audio_processed_s=audio_processed_s,
                current_run=current_run,
                discovery_stats=discovery_stats,
                extra={"status": "running"},
            ),
        )
        try:
            target.output_dir.mkdir(parents=True, exist_ok=True)
            diar_json = target.output_dir / "diarization_local.json"
            timing_json = target.output_dir / "timing_local.json"
            rttm_path = target.output_dir / "diarization_local.rttm"

            t0 = time.time()
            diarization = pipeline(str(target.audio_path), **diarize_kwargs)
            t1 = time.time()

            segments = _collect_segments(diarization)
            annotation = _as_annotation(diarization)
            with rttm_path.open("w", encoding="utf-8") as f:
                if hasattr(annotation, "write_rttm"):
                    annotation.write_rttm(f)

            infer_seconds = t1 - t0
            wall_seconds = infer_seconds
            rtf = infer_seconds / target.audio_duration_s if target.audio_duration_s > 0 else 0.0
            speed_x = (1.0 / rtf) if rtf > 0 else 0.0

            diar_payload = {
                "model": args.model,
                "run_rel": target.rel_path,
                "audio_path": str(target.audio_path),
                "audio_duration_s": target.audio_duration_s,
                "device": device,
                "segments": segments,
            }
            timing_payload = {
                "model": args.model,
                "run_rel": target.rel_path,
                "audio_path": str(target.audio_path),
                "output_dir": str(target.output_dir),
                "audio_duration_s": target.audio_duration_s,
                "load_seconds": 0.0,
                "inference_seconds": infer_seconds,
                "wall_seconds": wall_seconds,
                "real_time_factor": rtf,
                "speed_x_realtime": speed_x,
                "num_segments": len(segments),
                "device": device,
            }
            diar_json.write_text(json.dumps(diar_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            timing_json.write_text(json.dumps(timing_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            _append_csv_row(
                progress_csv,
                {
                    "timestamp": utc_now(),
                    "run_rel": target.rel_path,
                    "audio_path": str(target.audio_path),
                    "audio_duration_s": round(target.audio_duration_s, 3),
                    "status": "succeeded",
                    "inference_seconds": round(infer_seconds, 3),
                    "load_seconds": 0.0,
                    "wall_seconds": round(wall_seconds, 3),
                    "real_time_factor": round(rtf, 6),
                    "speed_x_realtime": round(speed_x, 6),
                    "num_segments": len(segments),
                    "error": "",
                },
            )
            succeeded += 1
            infer_seconds_sum += infer_seconds
            audio_processed_s += target.audio_duration_s
            logger.info(
                "[%s/%s] OK %s | %.1fs audio in %.1fs (%.2fx realtime), %s segments",
                idx,
                len(targets),
                target.rel_path,
                target.audio_duration_s,
                infer_seconds,
                speed_x,
                len(segments),
            )
        except Exception as exc:
            failed += 1
            logger.exception("[%s/%s] FAILED %s: %s", idx, len(targets), target.rel_path, exc)
            _append_csv_row(
                progress_csv,
                {
                    "timestamp": utc_now(),
                    "run_rel": target.rel_path,
                    "audio_path": str(target.audio_path),
                    "audio_duration_s": round(target.audio_duration_s, 3),
                    "status": "failed",
                    "inference_seconds": "",
                    "load_seconds": "",
                    "wall_seconds": "",
                    "real_time_factor": "",
                    "speed_x_realtime": "",
                    "num_segments": "",
                    "error": str(exc).replace("\n", " ")[:2000],
                },
            )
            if args.fail_fast:
                done = idx
                _write_json(
                    progress_json,
                    _progress_payload(
                        started_at=started_at,
                        targets=targets,
                        done=done,
                        succeeded=succeeded,
                        failed=failed,
                        skipped=skipped,
                        infer_seconds_sum=infer_seconds_sum,
                        audio_processed_s=audio_processed_s,
                        current_run=current_run,
                        discovery_stats=discovery_stats,
                        extra={"status": "failed_fail_fast"},
                    ),
                )
                raise
        finally:
            done = idx
            _write_json(
                progress_json,
                _progress_payload(
                    started_at=started_at,
                    targets=targets,
                    done=done,
                    succeeded=succeeded,
                    failed=failed,
                    skipped=skipped,
                    infer_seconds_sum=infer_seconds_sum,
                    audio_processed_s=audio_processed_s,
                    current_run=current_run if done < len(targets) else None,
                    discovery_stats=discovery_stats,
                    extra={"status": "running"},
                ),
            )

    final_payload = _progress_payload(
        started_at=started_at,
        targets=targets,
        done=done,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        infer_seconds_sum=infer_seconds_sum,
        audio_processed_s=audio_processed_s,
        current_run=None,
        discovery_stats=discovery_stats,
        extra={"status": "completed"},
    )
    _write_json(progress_json, final_payload)
    logger.info(
        "Completed local batch. selected=%s, succeeded=%s, failed=%s, processed_hours=%.3f",
        len(targets),
        succeeded,
        failed,
        audio_processed_s / 3600.0,
    )


if __name__ == "__main__":
    main()
