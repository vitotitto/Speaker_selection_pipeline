from __future__ import annotations

from typing import Any, Dict, List, Set

from .config import PostValidationConfig

VALID_SPEAKERS = {"subject", "interviewer", "narrator", "other", "noise"}


def _normalize_speaker(value: Any) -> str:
    return str(value).strip().lower()


def validate_llm_output(
    llm_output: Dict[str, Any],
    segments: List[Dict[str, Any]],
    audio_duration: float,
    config: PostValidationConfig,
) -> Dict[str, Any]:
    """Validate LLM classification output. Returns a ValidationResult dict."""
    checks: Dict[str, bool] = {}
    warnings: List[str] = []

    # 1. Has segment_classifications key
    classifications = llm_output.get("segment_classifications", [])
    checks["has_segment_classifications"] = isinstance(classifications, list) and len(classifications) > 0
    if not checks["has_segment_classifications"]:
        warnings.append("No segment_classifications in LLM output")
        return {"is_valid": False, "checks": checks, "warnings": warnings, "retry_count": 0}

    valid_ids: Set[int] = {s["id"] for s in segments}

    # 2. All IDs valid
    returned_ids = [c.get("segment_id") for c in classifications]
    invalid_ids = [i for i in returned_ids if i not in valid_ids]
    checks["all_ids_valid"] = len(invalid_ids) == 0
    if invalid_ids:
        warnings.append(f"Invalid segment IDs: {invalid_ids[:10]}")

    # 3. No duplicate IDs
    seen: Set[int] = set()
    dupes: List[int] = []
    for sid in returned_ids:
        if sid in seen:
            dupes.append(sid)
        seen.add(sid)
    checks["no_duplicate_ids"] = len(dupes) == 0
    if dupes:
        warnings.append(f"Duplicate segment IDs: {dupes[:10]}")

    # 4. Speaker labels valid
    bad_labels = [
        c.get("speaker") for c in classifications
        if _normalize_speaker(c.get("speaker", "")) not in VALID_SPEAKERS
    ]
    checks["speaker_labels_valid"] = len(bad_labels) == 0
    if bad_labels:
        warnings.append(f"Invalid speaker labels: {set(bad_labels)}")

    # 5. Has subject segments
    subject_segs = [
        c for c in classifications
        if _normalize_speaker(c.get("speaker", "")) == "subject"
    ]
    checks["has_subject_segments"] = len(subject_segs) > 0
    if not subject_segs:
        warnings.append("No segments classified as 'subject'")

    # 6. Subject fraction in bounds
    seg_by_id = {s["id"]: s for s in segments}
    subject_duration = 0.0
    for c in subject_segs:
        sid = c.get("segment_id")
        if sid in seg_by_id:
            s = seg_by_id[sid]
            subject_duration += s.get("end", 0) - s.get("start", 0)

    if audio_duration > 0:
        fraction = subject_duration / audio_duration
    else:
        fraction = 0.0

    in_bounds = (
        config.min_subject_speech_fraction <= fraction <= config.max_subject_speech_fraction
    )
    checks["subject_fraction_in_bounds"] = in_bounds
    if not in_bounds:
        warnings.append(
            f"Subject fraction {fraction:.2f} outside "
            f"[{config.min_subject_speech_fraction}, {config.max_subject_speech_fraction}]"
        )

    # 7. Coverage check — did the LLM classify most segments?
    coverage = len(returned_ids) / max(len(segments), 1)
    checks["good_coverage"] = coverage >= 0.8
    if coverage < 0.8:
        warnings.append(f"LLM only classified {len(returned_ids)}/{len(segments)} segments ({coverage:.0%})")

    is_valid = all(checks.values())
    return {
        "is_valid": is_valid,
        "checks": checks,
        "warnings": warnings,
        "retry_count": 0,
    }
