"""Shared utilities for speaker_analysis and content_screening."""
from __future__ import annotations

import json
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling various wrapper formats."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences
    if "```" in text:
        for fence in ("```json", "```"):
            start = text.find(fence)
            if start != -1:
                start += len(fence)
                end = text.find("```", start)
                if end != -1:
                    try:
                        return json.loads(text[start:end].strip())
                    except json.JSONDecodeError:
                        pass

    # Last resort: find outermost braces
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response ({len(text)} chars)")
