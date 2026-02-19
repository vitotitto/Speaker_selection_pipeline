from __future__ import annotations

from typing import Any, Dict, Iterable, List


def assign_speakers_to_words(
    words: List[Dict[str, Any]],
    diarization: List[Dict[str, Any]],
    default_speaker: str = "speaker_00",
) -> List[Dict[str, Any]]:
    if not diarization:
        for w in words:
            w["speaker"] = default_speaker
        return words

    diarization = sorted(diarization, key=lambda s: (s["start"], s["end"]))
    idx = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        while idx < len(diarization) and diarization[idx]["end"] <= mid:
            idx += 1
        if idx < len(diarization) and diarization[idx]["start"] <= mid < diarization[idx]["end"]:
            w["speaker"] = diarization[idx]["speaker"]
        else:
            w["speaker"] = default_speaker
    return words


def _join_words(words: Iterable[str]) -> str:
    text = ""
    for word in words:
        w = word.strip()
        if not w:
            continue
        if w.startswith((",", ".", "?", "!", ";", ":", "'")):
            text += w
        else:
            if text:
                text += " "
            text += w
    return text


def build_segments_from_words(
    words: List[Dict[str, Any]],
    max_gap_s: float,
) -> List[Dict[str, Any]]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    segments: List[Dict[str, Any]] = []
    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "words": [words[0]],
    }

    for w in words[1:]:
        gap = w["start"] - current["end"]
        if w["speaker"] == current["speaker"] and gap <= max_gap_s:
            current["end"] = w["end"]
            current["words"].append(w)
        else:
            segment_id = len(segments)
            segments.append(
                {
                    "id": segment_id,
                    "speaker": current["speaker"],
                    "start": current["start"],
                    "end": current["end"],
                    "text": _join_words([x["word"] for x in current["words"]]),
                    "words": current["words"],
                }
            )
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w],
            }

    segment_id = len(segments)
    segments.append(
        {
            "id": segment_id,
            "speaker": current["speaker"],
            "start": current["start"],
            "end": current["end"],
            "text": _join_words([x["word"] for x in current["words"]]),
            "words": current["words"],
        }
    )
    return segments
