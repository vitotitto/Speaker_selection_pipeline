from __future__ import annotations

from dataclasses import dataclass, field

from speaker_analysis.config import LLMProviderConfig


VALID_CONTENT_TYPES = {
    "interview_with_subject",
    "panel_with_subject",
    "tribute_about_subject",
    "news_obituary",
    "archival_mixed",
    "wrong_content",
    "music_performance",
    "unknown",
}

USABLE_CONTENT_TYPES = {
    "interview_with_subject",
    "panel_with_subject",
    "archival_mixed",
}


@dataclass
class ContentScreeningConfig:
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    sample_head: int = 30
    sample_tail: int = 10
    skip_existing: bool = True
    dry_run: bool = False
