from __future__ import annotations

from typing import Any, Dict, List
from speaker_analysis.context import PatientContext

SYSTEM_AUDIT_CONTEXT = """\
You are a strict Data Integrity Auditor for a medical research database.
Your task is to verify that speech segments attributed to a specific patient are valid.

You have access to the FULL PATIENT PROFILE (Name, Age, Condition, etc.).
You must check if the text contents of the segments plausibly come from this person AND match the file/folder context.

CRITICAL: You must detect IMPOSSIBLE mismatches or if the person is merely being DISCUSSED.

FLAG any segment that:
1. WRONG SPEAKER / IMPOSSIBLE COGNITION: 
   - Patient is a CHILD (<10yo) but text is adult complex speech.
   - Patient has SEVERE APHASIA / NON-VERBAL but text is fluent.
   - Patient is ELDERLY (90+) but text implies a young career Professional.
2. ABSENT TARGET / THIRD PERSON:
   - The text is ABOUT the patient, but not SPOKEN BY the patient (e.g. "He was a great man", "She is suffering from...").
   - The speaker is clearly an interviewer, host, or narrator talking to the audience about the patient.
3. CONTEXT MISMATCH:
   - The file name implies a specific show/event (e.g. "Radio Aircheck", "Interview 2010") but the text is completely unrelated (e.g. Cooking Show, Audiobook).
4. HALLUCINATION: Nonsense text, repeated phrases, or "Thanks for watching" debris.
5. BACKGROUND: TV, Radio, or other noise classified as speech.

If the patient is marked "Non-verbal" or "Severe Aphasia", ANY fluent speech MUST be flagged as WRONG SPEAKER or BACKGROUND.

Output a rigorous audit report.
"""

AUDIT_OUTPUT_SCHEMA = """\
Return a single JSON object:
{
  "audit_results": [
    {
      "segment_id": <int>,
      "verdict": "<PASS|FAIL|FLAG>",
      "confidence": <float 0.0-1.0>,
      "flag_reason": "<string or null if PASS>"
    }
  ],
  "overall_assessment": "<brief summary of data quality for this file>"
}
"""

def build_audit_prompt(
    patient_context: PatientContext,
    candidate_segments: List[Dict[str, Any]],
) -> str:
    path_context = f"{patient_context.group} / {patient_context.name} / {patient_context.timepoint} / {patient_context.video_stem}"
    
    parts = [
        SYSTEM_AUDIT_CONTEXT,
        "",
        "=== PATIENT PROFILE (OMNISCIENT) ===",
        f"Name: {patient_context.name}",
        f"Group: {patient_context.group}",
        f"Gender: {patient_context.gender or 'Unknown'}",
        f"Year of Birth: {patient_context.birth_year or 'Unknown'}",
        f"Condition: {patient_context.dementia_type or 'None'}",
        f"Language: {patient_context.language or 'en'}",
        "",
        "=== FILE CONTEXT ===",
        f"Full Path Structure: {path_context}",
        "Requirement: The text MUST align with this video identifier.",
        "",
        "=== CANDIDATE SEGMENTS (Attributed to Patient) ===",
        _format_segments_for_audit(candidate_segments),
        "",
        "=== OUTPUT FORMAT ===",
        AUDIT_OUTPUT_SCHEMA,
    ]
    return "\n".join(parts)

def _format_segments_for_audit(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg['id']}] ({seg['duration']:.1f}s): {seg['text']}")
    return "\n".join(lines)
