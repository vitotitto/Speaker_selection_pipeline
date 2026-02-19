from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from speaker_analysis.context import build_patient_context, PatientContext
from speaker_analysis.discovery import RunInfo, discover_runs
from speaker_analysis.llm_providers import create_provider

from .config import SpeakerAuditConfig
from .prompts import build_audit_prompt

logger = logging.getLogger(__name__)


def _parse_audit_json(response_text: str) -> Dict[str, Any]:
    """Parse JSON payload from an LLM response with tolerant extraction."""
    text = response_text.strip()
    if not text:
        raise ValueError("Empty response from audit model")

    # 1) Fast path: response is pure JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Audit response JSON is not an object")
    except json.JSONDecodeError:
        pass

    # 2) Common case: fenced json block
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            raise ValueError("Audit fenced JSON is not an object")
        except json.JSONDecodeError:
            pass

    # 3) Robust extraction: decode first valid object found in text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in audit response")

def run_audit(
    run_info: RunInfo,
    config: SpeakerAuditConfig,
    csv_dir: Path,
) -> Dict[str, Any]:
    # Skip existing output unless explicitly forced.
    audit_output_path = run_info.run_dir / "metadata" / config.output_filename
    if config.skip_existing and audit_output_path.exists():
        return {"status": "skipped", "reason": "exists"}

    # 1. Load Speaker Analysis Output
    analysis_path = run_info.run_dir / "metadata" / "speaker_analysis.json"
    if not analysis_path.exists():
        logger.warning(f"No speaker analysis found for {run_info.video_stem}")
        return {"status": "skipped", "reason": "no_analysis"}
    
    analysis_data = json.loads(analysis_path.read_text(encoding="utf-8"))
    
    # 2. Extract Patient Segments
    patient_segments = []
    for cls in analysis_data.get("segment_classifications", []):
        speaker = cls.get("speaker", "").lower()
        if speaker in ("patient", "subject"):
            # We need the text! The classification object MIGHT not have text if it was separate.
            # But wait, looking at speaker_analysis/classifier.py, the classification result 
            # only has {segment_id, speaker, confidence, reasoning}. 
            # We need to load the transcript to get the text.
            patient_segments.append(cls)
            
    if not patient_segments:
        logger.info(f"No patient segments found for {run_info.video_stem}")
        return {"status": "skipped", "reason": "no_patient_segments"}

    # 3. Load Transcript to get text
    transcript_path = run_info.transcript_path
    if not transcript_path.exists():
        return {"status": "error", "reason": "no_transcript"}
        
    transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    str_seg_map = {str(s["id"]): s for s in transcript_data.get("segments", [])}
    int_seg_map = {s["id"]: s for s in transcript_data.get("segments", [])}
    
    candidate_segments = []
    for p_seg in patient_segments:
        sid = p_seg["segment_id"]
        # Handle potential string/int mismatch in IDs
        full_seg = int_seg_map.get(sid) or str_seg_map.get(str(sid))
        
        if full_seg:
            candidate_segments.append({
                "id": sid,
                "text": full_seg.get("text", ""),
                "start": full_seg.get("start"),
                "end": full_seg.get("end"),
                "duration": full_seg.get("end", 0) - full_seg.get("start", 0)
            })
            
    # 4. Build Omniscient Context
    patient_context = build_patient_context(
        run_info.person,
        run_info.source,
        run_info.timepoint,
        run_info.video_stem,
        csv_dir,
    )
    
    # 5. Build Prompt
    prompt = build_audit_prompt(patient_context, candidate_segments)
    
    # 6. Call LLM
    provider = create_provider(config.llm)
    try:
        response_text = provider.classify_segments(prompt)
        audit_result = _parse_audit_json(response_text)
            
        # 7. Save Audit Report
        audit_output_path = run_info.run_dir / "metadata" / config.output_filename
        with open(audit_output_path, "w", encoding="utf-8") as f:
            json.dump(audit_result, f, indent=2)
            
        return {"status": "success", "audit": audit_result}
        
    except Exception as e:
        logger.error(f"Audit failed for {run_info.video_stem}: {e}")
        return {"status": "error", "reason": str(e)}

