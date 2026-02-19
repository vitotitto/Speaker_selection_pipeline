from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from pipeline.asr_faster_whisper import load_model
from pipeline.config import PipelineConfig
from pipeline.run import run_pipeline


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _is_v2_complete(meta_dir: Path) -> bool:
    run_json = meta_dir / "run.json"
    if not run_json.exists():
        return False
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if int(payload.get("pipeline_version", 0)) < 2:
        return False
    status = payload.get("status")
    if status == "success":
        return True
    if status == "failed":
        return False

    required = (
        meta_dir / "asr_info.json",
        meta_dir / "transcript.json",
        meta_dir / "segments_detailed.json",
        meta_dir / "words.json",
    )
    return all(p.exists() for p in required)


def _discover_media_files(
    data_root: Path,
    input_sources: Iterable[str],
    allowed_extensions: Iterable[str],
) -> List[Tuple[Path, Path, str]]:
    allowed = {ext.lower() for ext in allowed_extensions}
    videos: List[Tuple[Path, Path, str]] = []
    for source in input_sources:
        source_root = data_root / source
        if not source_root.exists():
            logging.warning("Input source does not exist: %s", source_root)
            continue
        for p in source_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                videos.append((p, source_root, source))
    videos.sort(key=lambda item: str(item[0]).lower())
    return videos


def _build_stage1_config(stage_cfg: Dict[str, Any]) -> PipelineConfig:
    asr_cfg = stage_cfg.get("asr", {}) or {}
    audio_cfg = stage_cfg.get("audio", {}) or {}
    config = PipelineConfig()

    config.audio.pcm_codec = str(audio_cfg.get("pcm_codec", config.audio.pcm_codec))
    config.audio.model_sample_rate = _as_int(
        audio_cfg.get("model_sample_rate"),
        config.audio.model_sample_rate,
    )
    config.audio.model_channels = _as_int(
        audio_cfg.get("model_channels"),
        config.audio.model_channels,
    )

    config.asr.backend = str(asr_cfg.get("backend", config.asr.backend))
    config.asr.model_name = str(asr_cfg.get("model_name", config.asr.model_name))
    config.asr.language = str(asr_cfg.get("language", config.asr.language))
    config.asr.device = str(asr_cfg.get("device", config.asr.device))
    config.asr.compute_type = str(asr_cfg.get("compute_type", config.asr.compute_type))
    config.asr.beam_size = _as_int(asr_cfg.get("beam_size"), config.asr.beam_size)
    config.asr.batch_size = _as_int(asr_cfg.get("batch_size"), config.asr.batch_size)
    config.asr.vad_filter = _as_bool(asr_cfg.get("vad_filter"), config.asr.vad_filter)
    config.asr.skip = _as_bool(asr_cfg.get("skip"), config.asr.skip)
    return config


def _run_stage1_asr(
    *,
    data_root: Path,
    runs_root: Path,
    stage_cfg: Dict[str, Any],
    continue_on_error: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    defaults = {
        "input_sources": ["Dementia_raw_data", "No_Dementia_raw_data"],
        "allowed_extensions": [".mkv", ".mp4", ".avi", ".mov", ".webm", ".ts", ".m4v", ".m4a"],
        "force": False,
        "max_files": None,
    }
    cfg = dict(defaults)
    cfg.update(stage_cfg or {})

    videos = _discover_media_files(
        data_root=data_root,
        input_sources=cfg["input_sources"],
        allowed_extensions=cfg["allowed_extensions"],
    )
    force = _as_bool(cfg.get("force"), False)
    max_files = cfg.get("max_files")
    if max_files is not None:
        max_files = int(max_files)
        if max_files >= 0:
            videos = videos[:max_files]

    to_process: List[Tuple[Path, Path, str, Path]] = []
    skipped_existing = 0
    for video_path, source_root, source_name in videos:
        rel = video_path.relative_to(source_root)
        out_dir = runs_root / source_name / rel.parent / video_path.stem
        meta_dir = out_dir / "metadata"
        if not force and _is_v2_complete(meta_dir):
            skipped_existing += 1
            continue
        to_process.append((video_path, source_root, source_name, out_dir))

    stage_summary: Dict[str, Any] = {
        "discovered_media_files": len(videos),
        "to_process": len(to_process),
        "skipped_existing": skipped_existing,
        "succeeded": 0,
        "failed": 0,
        "dry_run": dry_run,
    }
    logging.info(
        "Stage 1 ASR: discovered=%s, to_process=%s, skipped_existing=%s",
        len(videos),
        len(to_process),
        skipped_existing,
    )

    if dry_run or not to_process:
        return stage_summary

    pipeline_cfg = _build_stage1_config(cfg)
    asr_model = None
    if pipeline_cfg.asr.skip:
        logging.info("ASR skip enabled — no Whisper model will be loaded.")
    else:
        backend = str(getattr(pipeline_cfg.asr, "backend", "faster-whisper")).strip().lower()
        if backend in {"faster-whisper", "faster_whisper"}:
            logging.info(
                "Loading faster-whisper model once: model=%s device=%s compute_type=%s",
                pipeline_cfg.asr.model_name,
                pipeline_cfg.asr.device,
                pipeline_cfg.asr.compute_type,
            )
            t0 = time.perf_counter()
            asr_model = load_model(
                model_name=pipeline_cfg.asr.model_name,
                device=pipeline_cfg.asr.device,
                compute_type=pipeline_cfg.asr.compute_type,
            )
            logging.info("Model loaded in %.1fs", time.perf_counter() - t0)
        else:
            logging.info("Using ASR backend '%s' (per-file model load expected).", backend)

    for idx, (video_path, _source_root, _source_name, out_dir) in enumerate(to_process, start=1):
        label = f"[{idx}/{len(to_process)}]"
        logging.info("%s START %s", label, video_path)
        try:
            run_pipeline(str(video_path), str(out_dir), pipeline_cfg, asr_model=asr_model)
            stage_summary["succeeded"] += 1
            logging.info("%s SUCCESS %s", label, video_path.name)
        except Exception as exc:
            stage_summary["failed"] += 1
            logging.exception("%s FAILED %s: %s", label, video_path.name, exc)
            if not continue_on_error:
                raise
    return stage_summary


def _run_subprocess_stage(
    *,
    name: str,
    python_exe: Path,
    script_path: Path,
    args: List[str],
    cwd: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> Dict[str, Any]:
    cmd = [str(python_exe), str(script_path), *args]
    logging.info("[%s] Command: %s", name, " ".join(cmd))
    if dry_run:
        return {"status": "dry_run", "cmd": cmd, "returncode": 0}

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "").strip()
    root_str = str(cwd)
    env["PYTHONPATH"] = f"{root_str};{existing}" if existing else root_str

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    elapsed = round(time.perf_counter() - started, 2)
    result = {
        "status": "success" if proc.returncode == 0 else "failed",
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
    }
    if proc.returncode != 0 and not continue_on_error:
        raise RuntimeError(f"Stage {name} failed with exit code {proc.returncode}")
    return result


def _build_pyannote_api_args(cfg: Dict[str, Any], runs_root: Path) -> List[str]:
    args: List[str] = []
    if _as_bool(cfg.get("batch"), True):
        args.append("--batch")
    run_dir = cfg.get("run_dir")
    if run_dir:
        args.extend(["--run-dir", str(run_dir)])
    run_list_file = cfg.get("run_list_file")
    if run_list_file:
        args.extend(["--run-list-file", str(run_list_file)])

    args.extend(["--runs-root", str(runs_root)])
    args.extend(["--audio-file", str(cfg.get("audio_file", "audio_base.wav"))])
    args.extend(["--api-key-env", str(cfg.get("api_key_env", "PYANNOTE_API"))])
    args.extend(["--api-base-url", str(cfg.get("api_base_url", "https://api.pyannote.ai/v1"))])
    args.extend(["--model", str(cfg.get("model", "precision-2"))])
    args.extend(["--transcription-model", str(cfg.get("transcription_model", "parakeet-tdt-0.6b-v3"))])

    args.append("--transcription" if _as_bool(cfg.get("transcription"), True) else "--no-transcription")
    args.append("--exclusive" if _as_bool(cfg.get("exclusive"), True) else "--no-exclusive")
    args.append("--confidence" if _as_bool(cfg.get("confidence"), True) else "--no-confidence")
    args.append(
        "--turn-level-confidence"
        if _as_bool(cfg.get("turn_level_confidence"), True)
        else "--no-turn-level-confidence"
    )

    if _as_bool(cfg.get("write_standard"), False):
        args.append("--write-standard")
    if _as_bool(cfg.get("prefer_exclusive"), True):
        args.append("--prefer-exclusive")
    if _as_bool(cfg.get("submit_only"), False):
        args.append("--submit-only")
    if _as_bool(cfg.get("poll_only"), False):
        args.append("--poll-only")
    if _as_bool(cfg.get("force"), False):
        args.append("--force")
    if _as_bool(cfg.get("retry_failed"), False):
        args.append("--retry-failed")

    for key, flag in (
        ("num_speakers", "--num-speakers"),
        ("min_speakers", "--min-speakers"),
        ("max_speakers", "--max-speakers"),
        ("poll_interval_s", "--poll-interval-s"),
        ("timeout_s", "--timeout-s"),
        ("timeout_http_s", "--timeout-http-s"),
        ("max_runs", "--max-runs"),
    ):
        value = cfg.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    webhook = cfg.get("webhook")
    if webhook:
        args.extend(["--webhook", str(webhook)])

    report_file = cfg.get("report_file")
    if report_file:
        args.extend(["--report-file", str(report_file)])
    args.extend(["--log-file", str(cfg.get("log_file", "pyannote_api.log"))])
    return args


def _build_pyannote_local_args(
    cfg: Dict[str, Any],
    runs_root: Path,
    local_root: Path,
) -> List[str]:
    args: List[str] = [
        "--runs-root",
        str(runs_root),
        "--output-root",
        str(local_root),
        "--audio-file",
        str(cfg.get("audio_file", "audio/audio_16k.wav")),
        "--model",
        str(cfg.get("model", "pyannote/speaker-diarization-community-1")),
        "--hf-token-env",
        str(cfg.get("hf_token_env", "HF_TOKEN")),
        "--device",
        str(cfg.get("device", "auto")),
        "--progress-json",
        str(cfg.get("progress_json", "helper_scripts/local_pyannote_progress_live.json")),
        "--progress-csv",
        str(cfg.get("progress_csv", "helper_scripts/local_pyannote_progress.csv")),
        "--selection-report",
        str(cfg.get("selection_report", "helper_scripts/local_pyannote_selection.json")),
        "--log-file",
        str(cfg.get("log_file", "local_pyannote_batch.log")),
    ]
    for key, flag in (
        ("min_speakers", "--min-speakers"),
        ("max_speakers", "--max-speakers"),
        ("max_runs", "--max-runs"),
        ("max_audio_hours", "--max-audio-hours"),
    ):
        value = cfg.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    if _as_bool(cfg.get("skip_if_cloud_submitted"), True):
        args.append("--skip-if-cloud-submitted")
    else:
        args.append("--no-skip-if-cloud-submitted")

    if _as_bool(cfg.get("skip_existing_local"), True):
        args.append("--skip-existing-local")
    else:
        args.append("--no-skip-existing-local")

    if _as_bool(cfg.get("fail_fast"), False):
        args.append("--fail-fast")

    for item in cfg.get("exclude_run_list_file", []) or []:
        args.extend(["--exclude-run-list-file", str(item)])
    return args


def _build_overlap_args(
    cfg: Dict[str, Any],
    runs_root: Path,
    local_root: Path,
) -> List[str]:
    args = [
        "--runs-root",
        str(runs_root),
        "--local-root",
        str(local_root),
        "--source-speaker-analysis",
        str(cfg.get("source_speaker_analysis", "speaker_analysis.json")),
        "--source-segments",
        str(cfg.get("source_segments", "segments_detailed.json")),
        "--output-speaker-analysis",
        str(cfg.get("output_speaker_analysis", "speaker_analysis_overlap_selected.json")),
        "--min-overlap-ratio",
        str(_as_float(cfg.get("min_overlap_ratio"), 0.5)),
        "--min-overlap-seconds",
        str(_as_float(cfg.get("min_overlap_seconds"), 1.0)),
        "--default-quality",
        str(_as_float(cfg.get("default_quality"), 0.75)),
        "--report-prefix",
        str(cfg.get("report_prefix", "helper_scripts/overlap_transfer_batch")),
    ]
    if _as_bool(cfg.get("force"), False):
        args.append("--force")
    return args


def _build_speaker_analysis_args(
    cfg: Dict[str, Any],
    runs_root: Path,
    csv_dir: Path,
) -> List[str]:
    args = [
        "--batch",
        "--runs-root",
        str(runs_root),
        "--csv-dir",
        str(csv_dir),
        "--provider",
        str(cfg.get("provider", "gemini")),
    ]
    model = cfg.get("model")
    if model:
        args.extend(["--model", str(model)])
    if _as_bool(cfg.get("dry_run"), False):
        args.append("--dry-run")
    if _as_bool(cfg.get("force"), False):
        args.append("--force")
    if _as_bool(cfg.get("v2_only"), False):
        args.append("--v2-only")
    if _as_bool(cfg.get("disable_named_turn_guard"), False):
        args.append("--disable-named-turn-guard")
    if _as_bool(cfg.get("disable_subject_anchor"), False):
        args.append("--disable-subject-anchor")
    return args


def _build_audit_args(
    cfg: Dict[str, Any],
    runs_root: Path,
    csv_dir: Path,
) -> List[str]:
    args = [
        "--batch",
        "--runs-root",
        str(runs_root),
        "--csv-dir",
        str(csv_dir),
    ]
    if _as_bool(cfg.get("force"), False):
        args.append("--force")
    return args


def _build_extraction_args(
    cfg: Dict[str, Any],
    runs_root: Path,
    output_root: Path,
) -> List[str]:
    args = [
        "--batch",
        "--runs-root",
        str(runs_root),
        "--output-root",
        str(output_root),
        "--speaker-analysis-file",
        str(cfg.get("speaker_analysis_file", "speaker_analysis.json")),
        "--selection-mode",
        str(cfg.get("selection_mode", "continuity_first")),
        "--quality-threshold",
        str(_as_float(cfg.get("quality_threshold"), 0.7)),
        "--max-total-minutes",
        str(_as_float(cfg.get("max_total_minutes"), 5.0)),
        "--min-segment-duration-s",
        str(_as_float(cfg.get("min_segment_duration_s"), 4.0)),
        "--max-gap-s",
        str(_as_float(cfg.get("max_gap_s"), 0.75)),
        "--acoustic-min-speech-band-ratio",
        str(_as_float(cfg.get("acoustic_min_speech_band_ratio"), 0.38)),
        "--acoustic-min-voiced-ratio",
        str(_as_float(cfg.get("acoustic_min_voiced_ratio"), 0.28)),
        "--acoustic-max-music-score",
        str(_as_float(cfg.get("acoustic_max_music_score"), 0.60)),
        "--acoustic-min-speaker-consistency",
        str(_as_float(cfg.get("acoustic_min_speaker_consistency"), 0.60)),
        "--acoustic-min-subject-similarity",
        str(_as_float(cfg.get("acoustic_min_subject_similarity"), 0.45)),
        "--denoise-strength",
        str(_as_float(cfg.get("denoise_strength"), 0.65)),
    ]
    if cfg.get("top_quality_fraction") is not None:
        args.extend(["--top-quality-fraction", str(cfg.get("top_quality_fraction"))])
    if cfg.get("max_segment_duration_s") is not None:
        args.extend(["--max-segment-duration-s", str(cfg.get("max_segment_duration_s"))])
    if cfg.get("resample_to") is not None:
        args.extend(["--resample-to", str(cfg.get("resample_to"))])

    if _as_bool(cfg.get("force"), True):
        args.append("--force")
    if _as_bool(cfg.get("split_segments"), False):
        args.append("--split-segments")
    if _as_bool(cfg.get("acoustic_post_filter"), True):
        args.append("--acoustic-post-filter")
    if _as_bool(cfg.get("denoise_output"), True):
        args.append("--denoise-output")
    if _as_bool(cfg.get("dry_run"), False):
        args.append("--dry-run")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full speaker-selection pipeline using YAML config.")
    parser.add_argument(
        "--config",
        default="configs/full_pipeline_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and planned work without running.")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    config_path = _resolve_path(args.config, script_root)
    config = _load_config(config_path)

    project_cfg = config.get("project", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    stages_cfg = config.get("stages", {}) or {}

    data_root = _resolve_path(project_cfg.get("data_root", script_root), script_root)
    runs_root = _resolve_path(project_cfg.get("runs_root", data_root / "runs"), data_root)
    csv_dir = _resolve_path(project_cfg.get("csv_dir", data_root / "csv_sources"), data_root)
    pyannote_local_root = _resolve_path(
        project_cfg.get("pyannote_local_root", data_root / "pyannote_results_local"),
        data_root,
    )
    extraction_output_root = _resolve_path(
        project_cfg.get("extraction_output_root", data_root / "processed_final"),
        data_root,
    )

    python_exe = _resolve_path(runtime_cfg.get("python_exe", sys.executable), script_root)
    continue_on_error = _as_bool(runtime_cfg.get("continue_on_error"), True)
    log_file = _resolve_path(runtime_cfg.get("log_file", "logs/full_pipeline.log"), script_root)
    summary_file = _resolve_path(runtime_cfg.get("summary_file", "logs/full_pipeline_summary.json"), script_root)
    _setup_logging(log_file)

    helper_dir = script_root / "helper_scripts"

    logging.info("Config loaded: %s", config_path)
    logging.info("data_root=%s", data_root)
    logging.info("runs_root=%s", runs_root)
    logging.info("python_exe=%s", python_exe)
    logging.info("dry_run=%s", args.dry_run)

    stage_results: Dict[str, Any] = {}

    # Stage 1: video -> audio files (+ ASR when asr.skip is false)
    stage1_cfg = stages_cfg.get("stage_1_asr_prepare", {}) or {}
    if _as_bool(stage1_cfg.get("enabled"), True):
        logging.info("==== Stage 1: ASR prepare ====")
        stage_results["stage_1_asr_prepare"] = _run_stage1_asr(
            data_root=data_root,
            runs_root=runs_root,
            stage_cfg=stage1_cfg,
            continue_on_error=continue_on_error,
            dry_run=args.dry_run,
        )

    # Stage 2: pyannote API
    stage2_cfg = stages_cfg.get("stage_2_pyannote_api", {}) or {}
    if _as_bool(stage2_cfg.get("enabled"), False):
        logging.info("==== Stage 2: pyannote API ====")
        stage_results["stage_2_pyannote_api"] = _run_subprocess_stage(
            name="stage_2_pyannote_api",
            python_exe=python_exe,
            script_path=helper_dir / "run_pyannote_api.py",
            args=_build_pyannote_api_args(stage2_cfg, runs_root),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Stage 3: local pyannote
    stage3_cfg = stages_cfg.get("stage_3_pyannote_local", {}) or {}
    if _as_bool(stage3_cfg.get("enabled"), False):
        logging.info("==== Stage 3: pyannote local ====")
        stage_results["stage_3_pyannote_local"] = _run_subprocess_stage(
            name="stage_3_pyannote_local",
            python_exe=python_exe,
            script_path=helper_dir / "run_local_pyannote_batch.py",
            args=_build_pyannote_local_args(stage3_cfg, runs_root, pyannote_local_root),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Stage 4: audit
    stage4_cfg = stages_cfg.get("stage_4_audit", stages_cfg.get("stage_6_audit", {})) or {}
    if _as_bool(stage4_cfg.get("enabled"), False):
        logging.info("==== Stage 4: audit ====")
        stage_results["stage_4_audit"] = _run_subprocess_stage(
            name="stage_4_audit",
            python_exe=python_exe,
            script_path=helper_dir / "run_audit.py",
            args=_build_audit_args(stage4_cfg, runs_root, csv_dir),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Stage 5: speaker analysis (LLM)
    stage5_cfg = stages_cfg.get("stage_5_speaker_analysis", {}) or {}
    if _as_bool(stage5_cfg.get("enabled"), False):
        logging.info("==== Stage 5: speaker analysis ====")
        stage_results["stage_5_speaker_analysis"] = _run_subprocess_stage(
            name="stage_5_speaker_analysis",
            python_exe=python_exe,
            script_path=helper_dir / "run_speaker_analysis.py",
            args=_build_speaker_analysis_args(stage5_cfg, runs_root, csv_dir),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Stage 6: overlap transfer
    stage6_cfg = stages_cfg.get(
        "stage_6_overlap_transfer",
        stages_cfg.get("stage_4_overlap_transfer", {}),
    ) or {}
    if _as_bool(stage6_cfg.get("enabled"), True):
        logging.info("==== Stage 6: overlap transfer ====")
        stage_results["stage_6_overlap_transfer"] = _run_subprocess_stage(
            name="stage_6_overlap_transfer",
            python_exe=python_exe,
            script_path=helper_dir / "build_overlap_speaker_analysis_batch.py",
            args=_build_overlap_args(stage6_cfg, runs_root, pyannote_local_root),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Stage 7: extraction
    stage7_cfg = stages_cfg.get("stage_7_extraction", {}) or {}
    if _as_bool(stage7_cfg.get("enabled"), True):
        logging.info("==== Stage 7: extraction ====")
        stage7_effective = dict(stage7_cfg)
        speaker_analysis_file = str(stage7_effective.get("speaker_analysis_file", "auto")).strip()
        if not speaker_analysis_file or speaker_analysis_file.lower() == "auto":
            if _as_bool(stage6_cfg.get("enabled"), False):
                speaker_analysis_file = str(
                    stage6_cfg.get("output_speaker_analysis", "speaker_analysis_overlap_selected.json")
                )
            else:
                speaker_analysis_file = "speaker_analysis.json"
        stage7_effective["speaker_analysis_file"] = speaker_analysis_file

        out_root = _resolve_path(stage7_cfg.get("output_root", extraction_output_root), data_root)
        logging.info("Stage 7 speaker analysis source: %s", speaker_analysis_file)
        stage_results["stage_7_extraction"] = _run_subprocess_stage(
            name="stage_7_extraction",
            python_exe=python_exe,
            script_path=helper_dir / "run_audio_extraction.py",
            args=_build_extraction_args(stage7_effective, runs_root, out_root),
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    # Optional Stage 8: per-person budget extraction
    stage8_cfg = stages_cfg.get("stage_8_extraction_budget", {}) or {}
    if _as_bool(stage8_cfg.get("enabled"), False):
        logging.info("==== Stage 8: extraction budget ====")
        budget_output = _resolve_path(
            stage8_cfg.get("output_root", data_root / "processed_budget"),
            data_root,
        )
        stage8_effective = dict(stage8_cfg)
        stage8_sa_file = str(stage8_effective.get("speaker_analysis_file", "auto")).strip()
        if not stage8_sa_file or stage8_sa_file.lower() == "auto":
            if _as_bool(stage6_cfg.get("enabled"), False):
                stage8_sa_file = str(
                    stage6_cfg.get("output_speaker_analysis", "speaker_analysis_overlap_selected.json")
                )
            else:
                stage8_sa_file = "speaker_analysis.json"

        budget_args = [
            "--batch",
            "--runs-root",
            str(runs_root),
            "--output-root",
            str(budget_output),
            "--speaker-analysis-file",
            stage8_sa_file,
            "--minutes-per-person",
            str(_as_float(stage8_cfg.get("minutes_per_person"), 5.0)),
            "--min-segment-duration-s",
            str(_as_float(stage8_cfg.get("min_segment_duration_s"), 4.0)),
            "--quality-threshold",
            str(_as_float(stage8_cfg.get("quality_threshold"), 0.7)),
            "--alpha-quality",
            str(_as_float(stage8_cfg.get("alpha_quality"), 0.7)),
            "--beta-acoustic",
            str(_as_float(stage8_cfg.get("beta_acoustic"), 0.3)),
        ]
        if stage8_cfg.get("top_quality_fraction") is not None:
            budget_args.extend(["--top-quality-fraction", str(stage8_cfg.get("top_quality_fraction"))])
        if _as_bool(stage8_cfg.get("acoustic_post_filter"), True):
            budget_args.append("--acoustic-post-filter")
        if _as_bool(stage8_cfg.get("denoise_output"), True):
            budget_args.append("--denoise-output")
        if _as_bool(stage8_cfg.get("force"), True):
            budget_args.append("--force")

        stage_results["stage_8_extraction_budget"] = _run_subprocess_stage(
            name="stage_8_extraction_budget",
            python_exe=python_exe,
            script_path=helper_dir / "run_audio_extraction_budget.py",
            args=budget_args,
            cwd=script_root,
            dry_run=args.dry_run,
            continue_on_error=continue_on_error,
        )

    summary = {
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "data_root": str(data_root),
        "runs_root": str(runs_root),
        "python_exe": str(python_exe),
        "dry_run": args.dry_run,
        "stage_results": stage_results,
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Summary written: %s", summary_file)


if __name__ == "__main__":
    main()
