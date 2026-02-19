from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def diarize_audio(
    audio_path: str,
    hf_token_env: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyannote.audio is required. Install with: pip install pyannote.audio"
        ) from exc

    hf_token = os.getenv(hf_token_env)
    if not hf_token:
        raise RuntimeError(
            f"Hugging Face token not found in env var {hf_token_env}"
        )
    print(f"DEBUG: Found HF token: {hf_token[:4]}...{hf_token[-4:]} (Length: {len(hf_token)})")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments: List[Dict[str, Any]] = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(label),
            }
        )

    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments
