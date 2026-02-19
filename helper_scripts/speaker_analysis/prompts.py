from __future__ import annotations

from typing import Any, Dict, List

from .context import PatientContext

SYSTEM_CONTEXT = """\
You are a speech research assistant analyzing interview transcripts.
Your task is to classify which person is speaking in each segment.

The transcript comes from a video interview, talk show, or public appearance.
It features a subject of interest (the person being interviewed) and one or
more interviewers, hosts, or other speakers. Some segments may be narration,
introductions, or noise.

RULES:
1. Classify EVERY segment provided. Do not skip any.
2. Base classification ONLY on conversational structure (who asks vs. answers,
   who is introduced, turn-taking patterns, pronoun usage).
3. Do not invent information not in the transcript.
4. Questions directed at the subject are typically from interviewers; personal
   answers, anecdotes, and opinions are typically from the subject.
5. Narrators typically appear at the start, providing biographical context in third person.
6. Short affirmations ("Yes", "Right") should follow conversational flow.
7. If a segment is marked [AUTO-NOISE] or [FLAGGED], still classify it but factor
   the quality warning into your confidence score.
8. Classify speakers based on WHO is talking, not WHAT they are talking about.
   The content or topic of speech must not influence speaker identification.
9. Do not analyze the mental state, health, or cognitive ability of the speakers.
   Focus strictly on the structural role.
10. VERIFY IDENTITY: You must verify that [Subject Name] is present in the file.
    - If the person is NOT present (only discussed), you must NOT attribute speech to them.
    - You must explicitly extract segments for [Subject Name] if they are speaking.
    - If you are unsure if the speaker is [Subject Name], be conservative.
    - DATA INTEGRITY CHECK: If [Subject Name] is not found speaking, explicitly note this in the 'notes' field.
"""

OUTPUT_FORMAT = """\
Return PLAIN TEXT, one line per segment, using pipe-delimited fields.
Do NOT return JSON. Do NOT use code fences or brackets.

Format:
SEGMENT_ID | SPEAKER | CONFIDENCE | REASONING

Where:
- SEGMENT_ID: the integer segment ID from the transcript
- SPEAKER: one of subject, interviewer, narrator, other, noise
- CONFIDENCE: a float from 0.0 to 1.0
- REASONING: brief explanation, max 20 words

After all segment lines, add:
RECOMMENDED: <comma-separated list of segment IDs for best subject speech>
NOTES: <optional overall observations, max 50 words>

For RECOMMENDED: include only segment IDs classified as "subject"
with confidence >= 0.7, ordered best-first.

Example output:
1 | narrator | 0.95 | Third-person introduction of the subject
2 | interviewer | 0.90 | Asks direct question about career
3 | subject | 0.85 | Responds with personal anecdote
4 | noise | 0.30 | Unintelligible audio
RECOMMENDED: 3
NOTES: Subject is clearly present and actively engaged.
"""


def build_classification_prompt(
    patient_context: PatientContext,
    filtered_segments: List[Dict[str, Any]],
    audio_duration: float,
) -> str:
    parts = [
        SYSTEM_CONTEXT,
        "",
        "=== SUBJECT INFO ===",
        _format_patient_block(patient_context),
        "",
        f"Audio duration: {audio_duration:.1f} seconds",
        "",
        "=== TRANSCRIPT ===",
        _format_segments(filtered_segments),
        "",
        "=== OUTPUT FORMAT ===",
        OUTPUT_FORMAT,
    ]
    return "\n".join(parts)


def _format_patient_block(ctx: PatientContext) -> str:
    """Format subject info for the LLM prompt.

    Only includes information useful for speaker identification (name, gender).
    Excludes medical/diagnostic info to avoid classification bias.
    """
    lines = [
        f"Subject name: {ctx.name}",
    ]
    if ctx.gender:
        lines.append(f"Gender: {ctx.gender}")
    if ctx.birth_year:
        lived = str(ctx.birth_year)
        if ctx.death_year:
            lived += f" - {ctx.death_year}"
        lines.append(f"Lived: {lived}")
    return "\n".join(lines)


def _format_segments(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        marker = ""
        status = seg.get("pre_filter_status", "pass")
        if status == "noise":
            marker = " [AUTO-NOISE]"
        elif status.startswith("flagged"):
            reasons = seg.get("pre_filter_reasons", [])
            marker = f" [FLAGGED: {', '.join(reasons)}]"

        lines.append(
            f"[{seg['id']}] ({seg['start']:.1f}-{seg['end']:.1f}s){marker}: "
            f"{seg.get('text', '')}"
        )
    return "\n".join(lines)
