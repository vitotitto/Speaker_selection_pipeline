"""Core screening logic: load transcript sample, call LLM, parse result."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from speaker_analysis.config import LLMProviderConfig
from speaker_analysis.context import PatientContext, build_patient_context
from speaker_analysis.discovery import RunInfo
from speaker_analysis.llm_providers import create_provider
from speaker_analysis.utils import extract_json

from .config import ContentScreeningConfig, VALID_CONTENT_TYPES, USABLE_CONTENT_TYPES
from .prompts import build_screening_prompt

logger = logging.getLogger(__name__)


def _sample_segments(
    segments: List[Dict[str, Any]],
    head: int,
    tail: int,
) -> List[Dict[str, Any]]:
    """Take first `head` + last `tail` segments, avoiding overlap."""
    if len(segments) <= head + tail:
        return segments
    return segments[:head] + segments[-tail:]


def _build_file_path_context(run_info: RunInfo) -> str:
    lines = [
        f"Source folder: {run_info.source}",
        f"Person folder: {run_info.person}",
        f"Timepoint folder: {run_info.timepoint}",
        f"Video stem: {run_info.video_stem}",
    ]
    return "\n".join(lines)


def screen_run(
    run_info: RunInfo,
    config: ContentScreeningConfig,
    csv_dir: Path,
) -> Dict[str, Any]:
    """Screen a single run: load transcript, call LLM, return screening result."""
    # Load transcript
    transcript_data = json.loads(
        run_info.transcript_path.read_text(encoding="utf-8")
    )
    segments = transcript_data.get("segments", [])
    if not segments:
        return _build_result(
            run_info, config, None,
            content_type="unknown",
            subject_present=False,
            subject_speaking=False,
            estimated_subject_fraction=0.0,
            flags=["empty_transcript"],
            reasoning="Transcript has no segments.",
        )

    # Build full patient context (omniscient)
    patient_context = build_patient_context(
        run_info.person,
        run_info.source,
        run_info.timepoint,
        run_info.video_stem,
        csv_dir,
    )

    if config.dry_run:
        return _build_result(
            run_info, config, None,
            content_type="unknown",
            subject_present=False,
            subject_speaking=False,
            estimated_subject_fraction=0.0,
            flags=["dry_run"],
            reasoning="Dry run — no LLM call made.",
        )

    # Sample segments
    sample = _sample_segments(segments, config.sample_head, config.sample_tail)
    file_context = _build_file_path_context(run_info)

    # Build prompt
    prompt = build_screening_prompt(
        patient_context, sample, len(segments), file_context,
    )

    # Call LLM with retry
    provider = create_provider(config.llm)
    parsed = _call_llm_with_retry(provider, prompt, config.llm)

    # Validate and normalize the response
    content_type = parsed.get("content_type", "unknown")
    if content_type not in VALID_CONTENT_TYPES:
        content_type = "unknown"

    subject_present = bool(parsed.get("subject_present", False))
    subject_speaking = bool(parsed.get("subject_speaking", False))
    estimated_fraction = float(parsed.get("estimated_subject_fraction", 0.0))
    flags = parsed.get("flags", [])
    reasoning = parsed.get("reasoning", "")

    return _build_result(
        run_info, config, provider,
        content_type=content_type,
        subject_present=subject_present,
        subject_speaking=subject_speaking,
        estimated_subject_fraction=estimated_fraction,
        flags=flags,
        reasoning=reasoning,
    )


def _call_llm_with_retry(provider, prompt: str, llm_config: LLMProviderConfig) -> Dict[str, Any]:
    """Call the LLM and parse JSON response, with retries."""
    max_retries = llm_config.max_retries
    base_delay = llm_config.base_delay_s
    max_delay = llm_config.max_delay_s

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = provider.classify_segments(prompt)
            return extract_json(raw)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    raise RuntimeError(f"LLM failed after {max_retries + 1} attempts: {last_error}")


def _build_result(
    run_info: RunInfo,
    config: ContentScreeningConfig,
    provider,
    *,
    content_type: str,
    subject_present: bool,
    subject_speaking: bool,
    estimated_subject_fraction: float,
    flags: List[str],
    reasoning: str,
) -> Dict[str, Any]:
    usable = content_type in USABLE_CONTENT_TYPES
    return {
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "subject_name": run_info.person,
        "group": "Dementia" if "dementia" in run_info.source.lower() and "no_dementia" not in run_info.source.lower() else "No Dementia",
        "timepoint": run_info.timepoint,
        "video_stem": run_info.video_stem,
        "content_type": content_type,
        "subject_present": subject_present,
        "subject_speaking": subject_speaking,
        "estimated_subject_fraction": estimated_subject_fraction,
        "usable_for_analysis": usable,
        "flags": flags,
        "reasoning": reasoning,
        "llm_provider": provider.provider_name if provider else "none",
        "llm_model": provider.model_name if provider else "none",
    }
