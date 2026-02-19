from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object found in a text blob.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    return json.loads(text[start : end + 1])


def label_speakers_with_gemini(
    segments: List[Dict[str, Any]],
    speaker_ids: List[str],
    model_name: str,
    api_key_env: str,
) -> List[Dict[str, Any]]:
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-generativeai is required. Install with: pip install google-generativeai"
        ) from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Gemini API key not found in env var {api_key_env}")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Build a compact transcript for role inference.
    snippet_lines = []
    for seg in segments[:200]:
        speaker = seg["speaker"]
        text = seg.get("text", "").strip()
        if not text:
            continue
        snippet_lines.append(f"{speaker}: {text}")

    prompt = (
        "You are labeling speaker roles in an interview transcript.\n"
        "Return JSON with a top-level key 'speakers' as a list.\n"
        "Each item must include: id, role, confidence (0-1), notes.\n"
        "Roles should be concise (e.g., interviewer, interviewee, host, narrator).\n"
        "Only use the provided speaker IDs.\n\n"
        f"Speaker IDs: {', '.join(speaker_ids)}\n\n"
        "Transcript:\n"
        + "\n".join(snippet_lines)
    )

    response = model.generate_content(prompt)
    data = _extract_json(response.text)
    speakers = data.get("speakers", [])

    # Ensure all speakers are present.
    by_id = {s.get("id"): s for s in speakers if isinstance(s, dict)}
    output = []
    for sid in speaker_ids:
        if sid in by_id:
            output.append(by_id[sid])
        else:
            output.append(
                {
                    "id": sid,
                    "role": "unknown",
                    "confidence": 0.0,
                    "notes": "LLM did not return a label for this speaker.",
                }
            )
    return output
