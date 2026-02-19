from __future__ import annotations

import os
from abc import ABC, abstractmethod

from ..config import LLMProviderConfig


class LLMProvider(ABC):
    """Abstract interface for an LLM provider."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        key = os.getenv(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.provider_name} API key not found in env var "
                f"{self.config.api_key_env}"
            )

    @abstractmethod
    def classify_segments(self, prompt: str) -> str:
        """Send prompt to LLM and return raw response text."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        return self.config.model_name
