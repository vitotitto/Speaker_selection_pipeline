from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Dict, List


def _ensure_ffmpeg_on_path() -> None:
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if not ffmpeg_path:
        return
    ffmpeg_dir = Path(ffmpeg_path).parent
    current = os.environ.get("PATH", "")
    if str(ffmpeg_dir) not in current.split(os.pathsep):
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + current


def _allow_omegaconf_globals() -> None:
    try:
        import torch
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(
                [DictConfig, ListConfig, ContainerMetadata, typing.Any]
            )
    except Exception:
        # If this fails, let torch raise a clearer error on load.
        return


def transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    batch_size: int,
    vad_filter: bool,
) -> Dict[str, Any]:
    try:
        _ensure_ffmpeg_on_path()
        _allow_omegaconf_globals()
        import whisperx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "whisperx is required. Install with: pip install whisperx"
        ) from exc

    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    print("DEBUG: calling model.transcribe (without vad_filter)")
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        # vad_filter=vad_filter, # Removed due to TypeError in current whisperx version
    )

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words: List[Dict[str, Any]] = []
    for segment in aligned.get("segments", []):
        for word in segment.get("words", []):
            if word.get("start") is None or word.get("end") is None:
                continue
            words.append(
                {
                    "word": word.get("word", "").strip(),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                }
            )

    return {
        "language": aligned.get("language", language),
        "segments": aligned.get("segments", []),
        "words": words,
    }
