from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Reuse LLM config from speaker_analysis to keep it consistent
from speaker_analysis.config import LLMProviderConfig

@dataclass
class AuditThresholds:
    min_confidence: float = 0.7
    require_reasoning: bool = True

@dataclass
class SpeakerAuditConfig:
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    thresholds: AuditThresholds = field(default_factory=AuditThresholds)
    output_filename: str = "speaker_audit.json"
    skip_existing: bool = True
