from __future__ import annotations

import json
import logging
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .asr_faster_whisper import transcribe_with_faster_whisper
from .asr_whisperx import transcribe_with_whisperx
from .audio_utils import (
    ensure_dir,
    extract_audio_wav,
    ffprobe_audio_info,
    resample_audio_wav,
)
from .config import PipelineConfig
from .export_metadata import write_json

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_words(words: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for w in words or []:
        if not isinstance(w, dict):
            continue
        start = _safe_float(w.get("start"), 0.0)
        end = _safe_float(w.get("end"), start)
        if end < start:
            continue
        probability = _safe_float(w.get("probability", w.get("score", 1.0)), 1.0)
        normalized.append(
            {
                "word": str(w.get("word", "")).strip(),
                "start": round(start, 3),
                "end": round(end, 3),
                "probability": round(max(0.0, min(1.0, probability)), 4),
            }
        )
    return normalized


def _normalize_segments(segments: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for idx, s in enumerate(segments or [], start=1):
        if not isinstance(s, dict):
            continue
        start = _safe_float(s.get("start"), 0.0)
        end = _safe_float(s.get("end"), start)
        if end < start:
            continue

        seg_words = _normalize_words(s.get("words") or [])
        seg_id = s.get("id", idx)
        try:
            seg_id = int(seg_id)
        except (TypeError, ValueError):
            seg_id = idx

        normalized.append(
            {
                "id": seg_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(s.get("text", "")).strip(),
                "avg_logprob": round(_safe_float(s.get("avg_logprob"), 0.0), 4),
                "no_speech_prob": round(_safe_float(s.get("no_speech_prob"), 0.0), 4),
                "compression_ratio": round(_safe_float(s.get("compression_ratio"), 0.0), 4),
                "words": seg_words,
            }
        )
    return normalized


def run_pipeline(video_path: str, output_dir: str, config: PipelineConfig, asr_model=None) -> None:
    out_dir = Path(output_dir)
    audio_dir = out_dir / "audio"
    meta_dir = out_dir / "metadata"
    ensure_dir(str(audio_dir))
    ensure_dir(str(meta_dir))

    timings: Dict[str, float] = {}
    logger.info(f"Starting pipeline for {video_path}")

    run_info: Dict[str, Any] = {
        "video_path": video_path,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": 2,
        "config": json.loads(json.dumps(config, default=lambda o: o.__dict__)),
    }

    try:
        # ---- Step 1: Extract base audio ----
        t0 = time.perf_counter()
        logger.info("Extracting base audio...")
        base_audio = str(audio_dir / "audio_base.wav")
        audio_info = ffprobe_audio_info(video_path)
        run_info["ffprobe"] = audio_info

        extract_audio_wav(
            video_path,
            base_audio,
            pcm_codec=config.audio.pcm_codec,
        )
        timings["extract_audio"] = round(time.perf_counter() - t0, 2)

        # ---- Step 2: Resample to 16 kHz mono for ASR ----
        t0 = time.perf_counter()
        logger.info("Resampling audio to 16kHz mono...")
        model_audio = str(audio_dir / "audio_16k.wav")
        resample_audio_wav(
            base_audio,
            model_audio,
            sample_rate=config.audio.model_sample_rate,
            channels=config.audio.model_channels,
        )
        timings["resample"] = round(time.perf_counter() - t0, 2)

        # ---- Step 3: ASR with configured backend ----
        if config.asr.skip:
            logger.info("ASR skipped (asr.skip=true): only extracting audio")
            timings["asr"] = 0.0

            # ---- Finalize (audio-only) ----
            timings["total"] = round(sum(timings.values()), 2)
            run_info["timings_seconds"] = timings
            run_info["asr_skipped"] = True
            run_info["status"] = "success"
            run_info["outputs"] = {
                "audio_base": base_audio,
                "audio_model": model_audio,
            }
        else:
            t0 = time.perf_counter()
            asr_backend = str(getattr(config.asr, "backend", "faster-whisper")).strip().lower()
            logger.info(f"Running ASR ({asr_backend} {config.asr.model_name})...")

            if asr_backend in {"faster-whisper", "faster_whisper"}:
                asr_result = transcribe_with_faster_whisper(
                    model_audio,
                    model_name=config.asr.model_name,
                    device=config.asr.device,
                    compute_type=config.asr.compute_type,
                    language=config.asr.language,
                    beam_size=config.asr.beam_size,
                    vad_filter=config.asr.vad_filter,
                    batch_size=config.asr.batch_size,
                    model=asr_model,
                )
                asr_info = asr_result["info"]
                segments_raw = asr_result["segments"]
                words_raw = asr_result["words"]
            elif asr_backend in {"whisperx", "whisper-x"}:
                asr_result = transcribe_with_whisperx(
                    model_audio,
                    model_name=config.asr.model_name,
                    device=config.asr.device,
                    compute_type=config.asr.compute_type,
                    language=config.asr.language,
                    batch_size=config.asr.batch_size,
                    vad_filter=config.asr.vad_filter,
                )
                duration = _safe_float(
                    (audio_info.get("format") or {}).get("duration"),
                    0.0,
                )
                asr_info = {
                    "language": str(asr_result.get("language") or config.asr.language),
                    "language_probability": 1.0,
                    "duration": round(duration, 3),
                    "duration_after_vad": round(duration, 3),
                }
                segments_raw = asr_result.get("segments", [])
                words_raw = asr_result.get("words", [])
            else:
                raise ValueError(
                    f"Unsupported ASR backend '{asr_backend}'. Use 'faster-whisper' or 'whisperx'."
                )
            timings["asr"] = round(time.perf_counter() - t0, 2)

            segments = _normalize_segments(segments_raw)
            words = _normalize_words(words_raw)

            logger.info(
                f"ASR done: {len(segments)} segments, "
                f"{len(words)} words, "
                f"language={asr_info.get('language')} "
                f"({float(asr_info.get('language_probability', 0.0)):.2f}), "
                f"duration={float(asr_info.get('duration', 0.0)):.1f}s "
                f"(after VAD: {float(asr_info.get('duration_after_vad', 0.0)):.1f}s)"
            )

            # ---- Save outputs ----
            write_json(str(meta_dir / "asr_info.json"), asr_info)

            write_json(
                str(meta_dir / "transcript.json"),
                {
                    "language": asr_info["language"],
                    "language_probability": asr_info["language_probability"],
                    "segments": [
                        {
                            "id": s.get("id"),
                            "start": s.get("start"),
                            "end": s.get("end"),
                            "text": s.get("text", ""),
                            "avg_logprob": s.get("avg_logprob", 0.0),
                            "no_speech_prob": s.get("no_speech_prob", 0.0),
                            "compression_ratio": s.get("compression_ratio", 0.0),
                        }
                        for s in segments
                    ],
                },
            )

            write_json(
                str(meta_dir / "segments_detailed.json"),
                {"segments": segments},
            )

            write_json(
                str(meta_dir / "words.json"),
                {"words": words},
            )

            # ---- Finalize ----
            timings["total"] = round(sum(timings.values()), 2)
            run_info["timings_seconds"] = timings
            run_info["asr_info"] = asr_info
            run_info["status"] = "success"
            run_info["outputs"] = {
                "audio_base": base_audio,
                "audio_model": model_audio,
                "metadata": {
                    "asr_info": str(meta_dir / "asr_info.json"),
                    "transcript": str(meta_dir / "transcript.json"),
                    "segments_detailed": str(meta_dir / "segments_detailed.json"),
                    "words": str(meta_dir / "words.json"),
                },
            }

        write_json(str(meta_dir / "run.json"), run_info)

        logger.info(f"Pipeline finished for {video_path}")
        logger.info(f"Timings: {json.dumps(timings)}")

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg/ffprobe failed for {video_path}: {e}")
        run_info["status"] = "failed"
        run_info["error"] = {
            "type": "subprocess",
            "message": str(e),
            "stderr": getattr(e, "stderr", None),
            "traceback": traceback.format_exc(),
        }
        run_info["timings_seconds"] = timings
        write_json(str(meta_dir / "run.json"), run_info)
        raise

    except (MemoryError, OSError) as e:
        logger.error(f"Model load / I/O failed for {video_path}: {e}")
        run_info["status"] = "failed"
        run_info["error"] = {
            "type": "model_or_io",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        run_info["timings_seconds"] = timings
        write_json(str(meta_dir / "run.json"), run_info)
        raise

    except Exception as e:
        logger.error(f"Pipeline failed for {video_path}: {e}", exc_info=True)
        run_info["status"] = "failed"
        run_info["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        run_info["timings_seconds"] = timings
        write_json(str(meta_dir / "run.json"), run_info)
        raise
