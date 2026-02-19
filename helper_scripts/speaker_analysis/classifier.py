from __future__ import annotations

import difflib
import json
import logging
import math
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import SpeakerAnalysisConfig
from .context import PatientContext
from .llm_providers import create_provider
from .llm_providers.base import LLMProvider
from .post_validate import validate_llm_output
from .pre_filter import pre_filter_segments
from .prompts import build_classification_prompt

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_TITLE_TOKENS = {"mr", "mrs", "ms", "miss", "sir", "lord", "lady", "dr", "prof"}
_NON_NAME_TOKENS = {
    "a", "an", "and", "again", "as", "at", "back", "because", "but", "by",
    "for", "from", "here", "how", "i", "if", "in", "into", "is", "it", "its",
    "my", "of", "on", "or", "our", "so", "that", "the", "their", "there", "they",
    "this", "to", "we", "well", "what", "when", "where", "who", "why", "with",
    "yet", "you", "your",
}
_ADDRESS_PATTERNS = [
    re.compile(
        r"(?:^|[.!?]\s*)(?:well\s+|but\s+|and\s+|so\s+)?"
        r"(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2}),\s*"
        r"(?:you\b|as\b|do\b|did\b|are\b|is\b|can\b|could\b|will\b|would\b|thank\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.!?]\s*)(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){1,2})\s+do\s+you\b",
        re.IGNORECASE,
    ),
    re.compile(
        r",\s*(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2}),\s*that\b",
        re.IGNORECASE,
    ),
    # Intro pattern: "Alison Ankle, an activist...", "Robin Oakley there, ..."
    re.compile(
        r"(?:^|[.!?]\s*)(?P<name>[A-Z][a-z'\-]*(?:\s+[A-Z][a-z'\-]*){0,2})\s*,\s*"
        r"(?:an\b|a\b|the\b|who\b|there\b|from\b|on\b|as\b|one\b|thank\b|you're\b|you\b)",
    ),
]


def classify_run(
    segments: List[Dict[str, Any]],
    asr_info: Dict[str, Any],
    patient_context: PatientContext,
    config: SpeakerAnalysisConfig,
) -> Dict[str, Any]:
    """Full classification pipeline for a single run."""
    audio_duration = asr_info.get("duration", 0.0)

    # ---- Pre-filter ----
    filtered, pre_summary = pre_filter_segments(segments, config.pre_filter)
    logger.info(
        f"Pre-filter: {pre_summary['passed']} pass, "
        f"{pre_summary['noise']} noise, "
        f"{pre_summary['flagged_hallucination']} hallucination, "
        f"{pre_summary['flagged_low_confidence']} low-conf"
    )

    if config.dry_run:
        return _build_dry_run_output(filtered, pre_summary, patient_context, asr_info, config)

    # ---- LLM classification ----
    # ---- LLM classification (Batched) ----
    import copy
    
    # Relax validation for individual chunks (subject might be silent in a chunk)
    chunk_config = copy.deepcopy(config)
    chunk_config.post_validation.min_subject_speech_fraction = 0.0
    chunk_config.post_validation.max_subject_speech_fraction = 1.0

    provider = create_provider(config.llm)
    
    all_classifications: List[Dict[str, Any]] = []
    all_recommended_ids: List[int] = []
    chunk_notes: List[str] = []

    # Create chunks
    batch_size = config.max_segments_per_llm_call
    chunks = [filtered[i : i + batch_size] for i in range(0, len(filtered), batch_size)]
    logger.info(f"Processing {len(filtered)} segments in {len(chunks)} batches (max {batch_size})...")

    # Combine validation results just for logging last one or aggregate? 
    # We'll just keep the last one for the return record, checking logic doesn't use it elsewhere.
    last_validation = {} 

    for i, chunk_segs in enumerate(chunks):
        if not chunk_segs:
            continue
            
        chunk_start = chunk_segs[0]["start"]
        chunk_end = chunk_segs[-1]["end"]
        chunk_dur = chunk_end - chunk_start
        
        logger.info(f"Batch {i+1}/{len(chunks)}: {len(chunk_segs)} segs ({chunk_start:.1f}-{chunk_end:.1f}s)")
        
        prompt = build_classification_prompt(patient_context, chunk_segs, chunk_dur)
        
        batch_result, batch_validation = _call_llm_with_retry(
            provider, prompt, chunk_segs, chunk_dur, chunk_config
        )
        last_validation = batch_validation
        
        all_classifications.extend(batch_result.get("segment_classifications", []))
        all_recommended_ids.extend(batch_result.get("recommended_segments", []))
        if batch_result.get("notes"):
            chunk_notes.append(f"[B{i+1}]: {batch_result.get('notes')}")

    classifications = all_classifications
    recommended_ids = all_recommended_ids
    notes = " | ".join(chunk_notes)

    # ---- Normalize labels (e.g. "Antony Flew" → "subject") ----
    _normalize_labels(classifications, patient_context.name)

    # ---- Deterministic guard for panel cross-talk ----
    guard_summary = _apply_named_turn_guard(
        classifications=classifications,
        segments=filtered,
        subject_name=patient_context.name,
        max_gap_segments=config.post_validation.named_turn_guard_max_gap_segments,
        demoted_confidence=config.post_validation.named_turn_guard_demoted_confidence,
        enforce_subject_anchor=config.post_validation.enforce_subject_anchor_when_cued,
        subject_anchor_max_gap_segments=config.post_validation.subject_anchor_max_gap_segments,
        subject_anchor_max_non_subject_streak=config.post_validation.subject_anchor_max_non_subject_streak,
        enabled=config.post_validation.enable_named_turn_guard,
    )
    if guard_summary.get("demoted_subject_segments", 0) > 0:
        logger.info(
            "Named-turn guard demoted "
            f"{guard_summary['demoted_subject_segments']} subject segments "
            f"(non-subject cues: {guard_summary['non_subject_cues_detected']})"
        )

    # ---- Whole-run post-validation (real thresholds) ----
    merged_output = {
        "segment_classifications": classifications,
        "recommended_segments": recommended_ids,
    }
    validation = validate_llm_output(
        merged_output, filtered, audio_duration, config.post_validation
    )
    if not validation["is_valid"]:
        logger.warning(f"Whole-run validation warnings: {validation['warnings']}")

    # ---- Build recommended segments with quality scores ----
    seg_by_id = {s["id"]: s for s in filtered}
    class_by_id = {c["segment_id"]: c for c in classifications}

    recommended = []
    for rid in recommended_ids:
        sid = rid if isinstance(rid, int) else rid.get("segment_id", rid)
        if sid not in seg_by_id:
            continue
        seg = seg_by_id[sid]
        cls = class_by_id.get(sid, {})
        speaker = str(cls.get("speaker", "")).strip().lower()
        if speaker not in ("subject", "patient"):
            continue
        try:
            cls_confidence = float(cls.get("confidence", 0.0))
        except (TypeError, ValueError):
            cls_confidence = 0.0
        if not math.isfinite(cls_confidence) or cls_confidence < 0.7:
            continue
        qs = _quality_score(seg, cls)
        recommended.append({
            "segment_id": sid,
            "start": seg["start"],
            "end": seg["end"],
            "duration": round(seg["end"] - seg["start"], 3),
            "text": seg.get("text", ""),
            "quality_score": qs,
        })
    recommended.sort(key=lambda r: r["quality_score"], reverse=True)

    # ---- Statistics ----
    stats = _compute_statistics(classifications, filtered, audio_duration)

    return {
        "schema_version": "1.0",
        "pipeline_stage": "speaker_classification",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_used": json.loads(json.dumps(config, default=lambda o: o.__dict__)),
        "patient_name": patient_context.name,
        "group": patient_context.group,
        "timepoint": patient_context.timepoint,
        "video_stem": patient_context.video_stem,
        "pre_filter_summary": pre_summary,
        "llm_provider": provider.provider_name,
        "llm_model": provider.model_name,
        "segment_classifications": classifications,
        "isolation_guard": guard_summary,
        "validation": validation,
        "recommended_segments": recommended,
        "statistics": stats,
        "notes": notes,
    }


def _call_llm_with_retry(
    provider: LLMProvider,
    prompt: str,
    segments: List[Dict[str, Any]],
    audio_duration: float,
    config: SpeakerAnalysisConfig,
) -> tuple:
    max_retries = config.llm.max_retries
    base_delay = config.llm.base_delay_s
    max_delay = config.llm.max_delay_s

    last_error = None
    total_retries = 0

    for attempt in range(max_retries + 1):
        try:
            raw = provider.classify_segments(prompt)
            parsed = _parse_text_response(raw)
            validation = validate_llm_output(
                parsed, segments, audio_duration, config.post_validation
            )
            validation["retry_count"] = total_retries
            if validation["is_valid"]:
                return parsed, validation
            # Soft failure — validation issues but JSON parsed
            logger.warning(f"Validation warnings: {validation['warnings']}")
            if attempt == max_retries:
                # Accept with warnings on last attempt
                return parsed, validation
            raise ValueError(f"Validation failed: {validation['warnings']}")

        except Exception as e:
            last_error = e
            total_retries += 1
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    raise RuntimeError(f"LLM failed after {max_retries + 1} attempts: {last_error}")


def _parse_text_response(text: str) -> Dict[str, Any]:
    """Parse pipe-delimited text response from LLM into classification dict."""
    classifications: List[Dict[str, Any]] = []
    recommended: List[int] = []
    notes = ""

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.upper().startswith("SEGMENT_ID"):
            continue

        if line.upper().startswith("RECOMMENDED:"):
            ids_str = line.split(":", 1)[1].strip()
            recommended = [
                int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()
            ]
            continue

        if line.upper().startswith("NOTES:"):
            notes = line.split(":", 1)[1].strip()
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            try:
                seg_id = int(parts[0])
                speaker = parts[1].lower()
                confidence = float(parts[2])
                if not math.isfinite(confidence):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                reasoning = parts[3] if len(parts) >= 4 else ""
                classifications.append({
                    "segment_id": seg_id,
                    "speaker": speaker,
                    "confidence": confidence,
                    "reasoning": reasoning,
                })
            except (ValueError, IndexError):
                continue  # skip unparseable lines

    return {
        "segment_classifications": classifications,
        "recommended_segments": recommended,
        "notes": notes,
    }


def _normalize_labels(
    classifications: List[Dict[str, Any]],
    subject_name: str,
) -> None:
    """Normalize speaker labels in-place.

    If the LLM returned the subject's actual name (e.g. "Antony Flew") instead
    of "subject", fix it. Also accept "patient" as a synonym for "subject".
    """
    name_lower = subject_name.strip().lower()
    # Also match on last name alone
    name_parts = name_lower.split()
    last_name = name_parts[-1] if name_parts else ""

    normalized_count = 0
    for c in classifications:
        speaker = c.get("speaker", "")
        speaker_lower = speaker.strip().lower()
        # Already a valid canonical label
        if speaker_lower in ("subject", "interviewer", "narrator", "other", "noise"):
            continue
        # Check if the label is the subject's name or "patient"
        if (speaker_lower == name_lower
                or (last_name and speaker_lower == last_name)
                or speaker_lower == "patient"):
            c["speaker"] = "subject"
            normalized_count += 1

    if normalized_count:
        logger.info(f"Normalized {normalized_count} speaker labels to 'subject'")


def _safe_segment_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_name_tokens(value: str) -> List[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(value or "")]
    while tokens and tokens[0] in _TITLE_TOKENS:
        tokens = tokens[1:]
    return tokens


def _is_plausible_name(raw_name: str, tokens: List[str]) -> bool:
    if not tokens:
        return False
    if any(t in _NON_NAME_TOKENS for t in tokens):
        return False
    # Lowercase single-word vocatives are too noisy (e.g., "again", "yet").
    if len(tokens) == 1 and raw_name == raw_name.lower():
        return False
    return True


def _extract_addressed_names(text: str) -> List[Tuple[str, List[str]]]:
    if not text:
        return []

    extracted: List[Tuple[str, List[str]]] = []
    seen: set[str] = set()
    for pattern in _ADDRESS_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("name").strip(" ,.-")
            tokens = _normalize_name_tokens(raw)
            if not _is_plausible_name(raw, tokens):
                continue
            key = " ".join(tokens)
            if key in seen:
                continue
            seen.add(key)
            display = " ".join(part.title() for part in tokens)
            extracted.append((display, tokens))
    return extracted


def _similar_token(a: str, b: str) -> bool:
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82


def _matches_subject_name(candidate_tokens: List[str], subject_tokens: List[str]) -> bool:
    if not candidate_tokens or not subject_tokens:
        return False

    if (
        len(candidate_tokens) >= 2
        and len(subject_tokens) >= 2
        and _similar_token(candidate_tokens[0], subject_tokens[0])
        and _similar_token(candidate_tokens[-1], subject_tokens[-1])
    ):
        return True

    anchors = [subject_tokens[0], subject_tokens[-1]]
    return any(_similar_token(token, anchor) for token in candidate_tokens for anchor in anchors)


def _contains_subject_reference(text: str, subject_tokens: List[str]) -> bool:
    if not text or not subject_tokens:
        return False
    text_tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if not text_tokens:
        return False
    first_anchor = subject_tokens[0]
    last_anchor = subject_tokens[-1]
    first_hit = any(_similar_token(token, first_anchor) for token in text_tokens)
    last_hit = any(_similar_token(token, last_anchor) for token in text_tokens)
    if len(subject_tokens) >= 2:
        return first_hit and last_hit
    return first_hit or last_hit


def _apply_named_turn_guard(
    classifications: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    subject_name: str,
    max_gap_segments: int,
    demoted_confidence: float,
    enforce_subject_anchor: bool,
    subject_anchor_max_gap_segments: int,
    subject_anchor_max_non_subject_streak: int,
    enabled: bool,
) -> Dict[str, Any]:
    summary = {
        "enabled": bool(enabled),
        "subject_cues_detected": 0,
        "non_subject_cues_detected": 0,
        "demoted_subject_segments": 0,
        "anchor_demoted_subject_segments": 0,
        "demoted_segment_ids_sample": [],
    }
    if not enabled:
        return summary
    if max_gap_segments <= 0:
        return summary

    subject_tokens = _normalize_name_tokens(subject_name)
    if not subject_tokens:
        return summary

    class_by_id: Dict[int, Dict[str, Any]] = {}
    for classification in classifications:
        sid = _safe_segment_id(classification.get("segment_id"))
        if sid is None:
            continue
        class_by_id[sid] = classification

    ordered_segments = sorted(
        segments,
        key=lambda s: (_safe_segment_id(s.get("id")) is None, _safe_segment_id(s.get("id")) or 0),
    )

    latest_subject_cue_id: Optional[int] = None
    latest_other_cue_id: Optional[int] = None
    latest_other_cue_name = ""
    active_subject_anchor_id: Optional[int] = None
    non_subject_streak_after_anchor = 0
    demoted_segment_ids: List[int] = []
    clamped_demoted_conf = max(0.0, min(1.0, demoted_confidence))

    for segment in ordered_segments:
        sid = _safe_segment_id(segment.get("id"))
        if sid is None:
            continue
        classification = class_by_id.get(sid)
        if classification is None:
            continue

        speaker = str(classification.get("speaker", "")).strip().lower()
        text = str(segment.get("text", ""))

        # Detect explicit named cues in non-subject turns.
        if speaker in ("interviewer", "narrator", "other"):
            names = _extract_addressed_names(text)
            has_subject_cue = _contains_subject_reference(text, subject_tokens)
            has_other_cue = False
            current_other_name = ""

            for display, tokens in names:
                if _matches_subject_name(tokens, subject_tokens):
                    has_subject_cue = True
                else:
                    has_other_cue = True
                    if not current_other_name:
                        current_other_name = display

            if has_subject_cue:
                latest_subject_cue_id = sid
                active_subject_anchor_id = sid
                non_subject_streak_after_anchor = 0
                summary["subject_cues_detected"] += 1

            if has_other_cue:
                latest_other_cue_id = sid
                latest_other_cue_name = current_other_name
                summary["non_subject_cues_detected"] += 1
                # A clear switch to another named person closes subject anchoring.
                active_subject_anchor_id = None
                non_subject_streak_after_anchor = 0
            elif active_subject_anchor_id is not None and not has_subject_cue:
                non_subject_streak_after_anchor += 1

        if speaker not in ("subject", "patient"):
            continue

        demote_reason = ""

        if latest_other_cue_id is not None:
            gap_from_other = sid - latest_other_cue_id
            if 0 < gap_from_other <= max_gap_segments:
                # Keep only if the latest explicit cue points to subject more recently.
                if latest_subject_cue_id is None or latest_subject_cue_id <= latest_other_cue_id:
                    demote_reason = "recent cue points to different named speaker"

        cues_present = (summary["subject_cues_detected"] + summary["non_subject_cues_detected"]) > 0
        if (
            not demote_reason
            and enforce_subject_anchor
            and cues_present
            and subject_anchor_max_gap_segments > 0
        ):
            if active_subject_anchor_id is None:
                demote_reason = "no active subject anchor"
            else:
                anchor_gap = sid - active_subject_anchor_id
                if anchor_gap <= 0 or anchor_gap > subject_anchor_max_gap_segments:
                    demote_reason = "subject anchor is too far back"
                elif (
                    subject_anchor_max_non_subject_streak >= 0
                    and non_subject_streak_after_anchor > subject_anchor_max_non_subject_streak
                ):
                    demote_reason = "too many non-subject turns since last subject anchor"

        if not demote_reason:
            if active_subject_anchor_id is not None:
                active_subject_anchor_id = sid
                non_subject_streak_after_anchor = 0
            continue

        confidence = classification.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        if not math.isfinite(confidence_value):
            confidence_value = 0.0

        classification["speaker"] = "other"
        classification["confidence"] = min(confidence_value, clamped_demoted_conf)
        guard_reason = f"demoted by named-turn guard ({demote_reason})"
        if latest_other_cue_name:
            guard_reason += f" (recent addressee: {latest_other_cue_name})"

        existing_reason = str(classification.get("reasoning", "")).strip()
        if existing_reason:
            if guard_reason.lower() not in existing_reason.lower():
                classification["reasoning"] = f"{existing_reason}; {guard_reason}"
        else:
            classification["reasoning"] = guard_reason

        if (
            "anchor" in demote_reason
            or "non-subject turns" in demote_reason
        ):
            summary["anchor_demoted_subject_segments"] += 1
        demoted_segment_ids.append(sid)

    summary["demoted_subject_segments"] = len(demoted_segment_ids)
    summary["demoted_segment_ids_sample"] = demoted_segment_ids[:50]
    return summary


def _quality_score(seg: Dict[str, Any], classification: Dict[str, Any]) -> float:
    llm_conf = classification.get("confidence", 0.5)
    logprob = seg.get("avg_logprob", -0.5)
    # Normalize avg_logprob from [-1, 0] to [0, 1]
    norm_logprob = min(1.0, max(0.0, (logprob + 1.0)))
    filter_bonus = 1.0 if seg.get("pre_filter_status") == "pass" else 0.3
    return round(0.5 * llm_conf + 0.3 * norm_logprob + 0.2 * filter_bonus, 4)


def _compute_statistics(
    classifications: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    audio_duration: float,
) -> Dict[str, Any]:
    seg_by_id = {s["id"]: s for s in segments}
    stats: Dict[str, Any] = {
        "patient_segments": 0,
        "patient_duration_s": 0.0,
        "patient_fraction": 0.0,
        "interviewer_segments": 0,
        "interviewer_duration_s": 0.0,
        "narrator_segments": 0,
        "noise_segments": 0,
        "other_segments": 0,
        "total_duration_s": audio_duration,
        "mean_patient_confidence": 0.0,
    }

    patient_confs: List[float] = []
    for c in classifications:
        speaker = c.get("speaker", "other")
        sid = c.get("segment_id")
        dur = 0.0
        if sid in seg_by_id:
            s = seg_by_id[sid]
            dur = s.get("end", 0) - s.get("start", 0)

        if speaker == "patient" or speaker == "subject":
            stats["patient_segments"] += 1
            stats["patient_duration_s"] += dur
            patient_confs.append(c.get("confidence", 0.0))
        elif speaker == "interviewer":
            stats["interviewer_segments"] += 1
            stats["interviewer_duration_s"] += dur
        elif speaker == "narrator":
            stats["narrator_segments"] += 1
        elif speaker == "noise":
            stats["noise_segments"] += 1
        else:
            stats["other_segments"] += 1

    if audio_duration > 0:
        stats["patient_fraction"] = round(stats["patient_duration_s"] / audio_duration, 4)
    if patient_confs:
        stats["mean_patient_confidence"] = round(sum(patient_confs) / len(patient_confs), 4)

    stats["patient_duration_s"] = round(stats["patient_duration_s"], 2)
    stats["interviewer_duration_s"] = round(stats["interviewer_duration_s"], 2)
    return stats


def _build_dry_run_output(
    filtered: List[Dict[str, Any]],
    pre_summary: Dict[str, Any],
    patient_context: PatientContext,
    asr_info: Dict[str, Any],
    config: SpeakerAnalysisConfig,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline_stage": "speaker_classification",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_used": json.loads(json.dumps(config, default=lambda o: o.__dict__)),
        "patient_name": patient_context.name,
        "group": patient_context.group,
        "timepoint": patient_context.timepoint,
        "video_stem": patient_context.video_stem,
        "pre_filter_summary": pre_summary,
        "llm_provider": "dry_run",
        "llm_model": "none",
        "segment_classifications": [],
        "validation": {"is_valid": False, "checks": {}, "warnings": ["dry_run"], "retry_count": 0},
        "recommended_segments": [],
        "statistics": {"total_duration_s": asr_info.get("duration", 0.0)},
        "notes": "Dry run — pre-filter only, no LLM call.",
    }
