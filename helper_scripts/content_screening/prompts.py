from __future__ import annotations

from typing import Any, Dict, List

from speaker_analysis.context import PatientContext


SCREENING_SYSTEM = """\
You are a content classification assistant for a speech research project.
Your task is to determine whether a video actually contains the expected subject
speaking, or whether it is a different type of content (tribute, obituary,
wrong video, etc.).

You have FULL context about the subject and the file structure. Use this to
make an informed judgement about the video content.

RULES:
1. Focus on whether the SUBJECT is present and speaking, not just mentioned.
2. Third-person references ("He was a great man") indicate the subject is being
   DISCUSSED, not that they are speaking.
3. First-person speech, direct answers to questions, and conversational
   turn-taking indicate the subject IS speaking.
4. Video titles can be misleading — a "Memorial" video might actually contain
   archival interview footage of the subject.
5. If the person folder name does not match the actual content, flag it as
   wrong_content.
6. Music performances with minimal speech should be flagged as music_performance.
7. News segments reporting on someone's death/condition are news_obituary.
8. If the subject appears only in embedded archival clips within a larger
   news/documentary piece, classify as archival_mixed.
"""

SCREENING_OUTPUT_SCHEMA = """\
Return a single JSON object with this exact structure.
Do not include any text before or after the JSON.
Do not output markdown code fences.
{
  "content_type": "<interview_with_subject|panel_with_subject|tribute_about_subject|news_obituary|archival_mixed|wrong_content|music_performance|unknown>",
  "subject_present": <true|false>,
  "subject_speaking": <true|false>,
  "estimated_subject_fraction": <float 0.0-1.0>,
  "flags": [<list of string warnings, if any>],
  "reasoning": "<2-3 sentence explanation>"
}

content_type definitions:
- interview_with_subject: Subject is present and interviewed (best case for analysis)
- panel_with_subject: Subject participates in a multi-person discussion/panel
- tribute_about_subject: Others talking about the subject (memorial, retrospective, tribute)
- news_obituary: News report about the subject's death or condition
- archival_mixed: News/documentary with embedded clips of the subject speaking
- wrong_content: Video has nothing to do with the named subject (wrong link, unrelated person)
- music_performance: Primarily music/singing with minimal speech
- unknown: Cannot determine content type from available information
"""


def build_screening_prompt(
    patient_context: PatientContext,
    segments_sample: List[Dict[str, Any]],
    total_segment_count: int,
    file_path_context: str,
) -> str:
    parts = [
        SCREENING_SYSTEM,
        "",
        "=== SUBJECT PROFILE ===",
        _format_full_profile(patient_context),
        "",
        "=== FILE CONTEXT ===",
        file_path_context,
        "",
        f"=== TRANSCRIPT SAMPLE ({len(segments_sample)} of {total_segment_count} segments) ===",
        _format_segments_sample(segments_sample),
        "",
        "=== OUTPUT FORMAT ===",
        SCREENING_OUTPUT_SCHEMA,
    ]
    return "\n".join(parts)


def _format_full_profile(ctx: PatientContext) -> str:
    """Full omniscient profile — screening agent sees everything."""
    lines = [
        f"Name: {ctx.name}",
        f"Group: {ctx.group}",
    ]
    if ctx.dementia_type:
        lines.append(f"Dementia type: {ctx.dementia_type}")
    if ctx.gender:
        lines.append(f"Gender: {ctx.gender}")
    if ctx.birth_year:
        lived = str(ctx.birth_year)
        if ctx.death_year:
            lived += f" - {ctx.death_year}"
        lines.append(f"Lived: {lived}")
    if ctx.first_symptoms_year:
        lines.append(f"First symptoms year: {ctx.first_symptoms_year}")
    if ctx.timepoint:
        lines.append(f"Timepoint: {ctx.timepoint}")
    if ctx.video_stem:
        lines.append(f"Video title: {ctx.video_stem}")
    if ctx.language:
        lines.append(f"Language: {ctx.language}")
    return "\n".join(lines)


def _format_segments_sample(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        lines.append(
            f"[{seg.get('id', '?')}] ({seg.get('start', 0):.1f}-{seg.get('end', 0):.1f}s): "
            f"{seg.get('text', '')}"
        )
    return "\n".join(lines)
