from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Dict, List, Optional


def load_model(
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
):
    """Load a WhisperModel once for reuse across multiple transcriptions."""
    from faster_whisper import WhisperModel

    try:
        # Prefer local cache to avoid network/proxy failures on offline workers.
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
    except Exception:
        # Fall back to default behavior (may download) when cache is missing.
        return WhisperModel(model_name, device=device, compute_type=compute_type)


def _stderr_is_usable() -> bool:
    if sys.stderr is None:
        return False
    try:
        sys.stderr.flush()
    except OSError:
        return False
    return True


def transcribe_with_faster_whisper(
    audio_path: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = "en",
    beam_size: int = 5,
    vad_filter: bool = True,
    batch_size: int = 16,
    model: Any = None,
) -> Dict[str, Any]:
    """Run faster-whisper with word-level timestamps.

    Args:
        model: Pre-loaded WhisperModel. If None, a new model is loaded.

    Returns dict with keys:
        info: transcription metadata (language, duration, etc.)
        segments: list of segment dicts with confidence metrics
        words: flat list of word dicts with per-word probability
    """
    if model is None:
        model = load_model(model_name, device, compute_type)

    def _run_transcribe() -> Dict[str, Any]:
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=True,
            # tqdm can touch stderr even when disabled.
            log_progress=False,
        )

        segments: List[Dict[str, Any]] = []
        all_words: List[Dict[str, Any]] = []

        for seg in segments_iter:
            seg_words: List[Dict[str, Any]] = []
            if seg.words:
                for w in seg.words:
                    word_entry = {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    seg_words.append(word_entry)
                    all_words.append(word_entry)

            segments.append(
                {
                    "id": seg.id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "compression_ratio": round(seg.compression_ratio, 4),
                    "words": seg_words,
                }
            )

        info_dict = {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "duration_after_vad": round(info.duration_after_vad, 3),
        }

        return {
            "info": info_dict,
            "segments": segments,
            "words": all_words,
        }

    if _stderr_is_usable():
        return _run_transcribe()

    # Detached Windows sessions can expose an invalid stderr handle.
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
        return _run_transcribe()
