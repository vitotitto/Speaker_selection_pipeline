from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from .config import PreFilterConfig


def pre_filter_segments(
    segments: List[Dict[str, Any]],
    config: PreFilterConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply pre-LLM quality filters to transcript segments.

    Each segment gets annotated with:
        pre_filter_status:  "pass" | "noise" | "flagged_hallucination" | "flagged_low_confidence"
        pre_filter_reasons: list of reason strings

    Returns (annotated_segments, summary_dict).
    """
    repeated_ids = _detect_repeated_phrases(segments, config)

    counts = {"total": len(segments), "pass": 0, "noise": 0,
              "flagged_hallucination": 0, "flagged_low_confidence": 0}

    annotated: List[Dict[str, Any]] = []
    for seg in segments:
        reasons: List[str] = []
        has_metrics = "no_speech_prob" in seg

        if has_metrics:
            reasons.extend(_check_noise(seg, config))
            reasons.extend(_check_hallucination(seg, config))
            reasons.extend(_check_low_confidence(seg, config))

        if seg.get("id") in repeated_ids:
            reasons.append("repeated_phrase")

        # Determine status (priority: noise > hallucination > low_confidence > pass)
        if any(r.startswith("no_speech_prob") or r.startswith("very_short") for r in reasons):
            status = "noise"
        elif any(r.startswith("compression_ratio") or r == "repeated_phrase"
                 or r.startswith("hallucination_phrase") for r in reasons):
            status = "flagged_hallucination"
        elif any(r.startswith("avg_logprob") for r in reasons):
            status = "flagged_low_confidence"
        else:
            status = "pass"

        counts[status] = counts.get(status, 0) + 1

        entry = dict(seg)
        entry["pre_filter_status"] = status
        entry["pre_filter_reasons"] = reasons
        annotated.append(entry)

    has_metrics = len(segments) > 0 and "no_speech_prob" in segments[0]
    summary = {
        "total_segments": counts["total"],
        "passed": counts["pass"],
        "noise": counts["noise"],
        "flagged_hallucination": counts["flagged_hallucination"],
        "flagged_low_confidence": counts["flagged_low_confidence"],
        "segments_sent_to_llm": counts["total"],  # all go to LLM, just annotated
        "pre_filter_skipped": not has_metrics,
    }
    return annotated, summary


def _check_noise(seg: Dict[str, Any], config: PreFilterConfig) -> List[str]:
    reasons: List[str] = []
    nsp = seg.get("no_speech_prob", 0.0)
    if nsp > config.no_speech_prob_threshold:
        reasons.append(f"no_speech_prob={nsp:.3f}")

    duration = seg.get("end", 0) - seg.get("start", 0)
    word_count = len(seg.get("text", "").split())
    if duration < config.min_segment_duration_s or word_count <= 1:
        reasons.append(f"very_short={duration:.2f}s/{word_count}w")
    return reasons


def _check_hallucination(seg: Dict[str, Any], config: PreFilterConfig) -> List[str]:
    reasons: List[str] = []
    cr = seg.get("compression_ratio", 0.0)
    if cr > config.compression_ratio_threshold:
        reasons.append(f"compression_ratio={cr:.2f}")

    text_lower = seg.get("text", "").lower().strip()
    for phrase in config.hallucination_phrases:
        if phrase in text_lower:
            reasons.append(f"hallucination_phrase='{phrase}'")
            break  # one match is enough
    return reasons


def _check_low_confidence(seg: Dict[str, Any], config: PreFilterConfig) -> List[str]:
    reasons: List[str] = []
    alp = seg.get("avg_logprob", 0.0)
    if alp < config.avg_logprob_threshold:
        reasons.append(f"avg_logprob={alp:.4f}")
    return reasons


def _detect_repeated_phrases(
    segments: List[Dict[str, Any]],
    config: PreFilterConfig,
) -> set:
    """Return segment IDs whose text appears suspiciously often (Whisper hallucination)."""
    text_counter: Counter = Counter()
    text_to_ids: Dict[str, List[int]] = {}

    for seg in segments:
        text = seg.get("text", "").strip().lower()
        if len(text) < 10:  # skip very short texts
            continue
        text_counter[text] += 1
        text_to_ids.setdefault(text, []).append(seg.get("id", -1))

    flagged: set = set()
    for text, count in text_counter.items():
        if count >= config.max_phrase_repeat_count:
            for sid in text_to_ids[text]:
                flagged.add(sid)
    return flagged
