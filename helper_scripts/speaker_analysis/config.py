from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMProviderConfig:
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    timeout_s: float = 120.0
    temperature: float = 0.1
    max_output_tokens: int = 8192


@dataclass
class PreFilterConfig:
    no_speech_prob_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    avg_logprob_threshold: float = -1.0
    min_segment_duration_s: float = 0.5
    hallucination_phrases: List[str] = field(default_factory=lambda: [
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "subtitles by",
        "captions by",
        "translated by",
        "transcribed by",
    ])
    max_phrase_repeat_count: int = 3


@dataclass
class PostValidationConfig:
    min_subject_speech_fraction: float = 0.05
    max_subject_speech_fraction: float = 0.90
    enable_named_turn_guard: bool = True
    named_turn_guard_max_gap_segments: int = 35
    named_turn_guard_demoted_confidence: float = 0.35
    enforce_subject_anchor_when_cued: bool = True
    subject_anchor_max_gap_segments: int = 30
    subject_anchor_max_non_subject_streak: int = 6


@dataclass
class SpeakerAnalysisConfig:
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    pre_filter: PreFilterConfig = field(default_factory=PreFilterConfig)
    post_validation: PostValidationConfig = field(default_factory=PostValidationConfig)
    max_segments_per_llm_call: int = 50
    skip_existing: bool = True
    dry_run: bool = False
    ignore_content_screening: bool = False
