from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def export_speaker_audio(
    cleaned_audio_path: str,
    segments: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, str]]:
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required. Install with: pip install soundfile"
        ) from exc

    info = sf.info(cleaned_audio_path)
    subtype = info.subtype
    dtype = "float32"
    if subtype == "PCM_16":
        dtype = "int16"
    elif subtype in ("PCM_24", "PCM_32"):
        dtype = "int32"

    audio, sr = sf.read(cleaned_audio_path, dtype=dtype, always_2d=True)
    channels = audio.shape[1]

    speaker_segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        speaker_segments[seg["speaker"]].append(seg)

    speaker_map: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    speaker_files: Dict[str, str] = {}

    for speaker, segs in speaker_segments.items():
        segs = sorted(segs, key=lambda s: (s["start"], s["end"]))
        output_path = f"{output_dir}/{speaker}.wav"
        speaker_files[speaker] = output_path

        offset = 0.0
        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=sr,
            channels=channels,
            subtype=subtype,
        ) as f:
            for seg in segs:
                start = max(0, int(round(seg["start"] * sr)))
                end = min(len(audio), int(round(seg["end"] * sr)))
                if end <= start:
                    continue
                f.write(audio[start:end])
                duration = (end - start) / sr
                speaker_map[speaker].append(
                    {
                        "segment_id": seg["id"],
                        "out_start": offset,
                        "out_end": offset + duration,
                    }
                )
                offset += duration

    return speaker_map, speaker_files
