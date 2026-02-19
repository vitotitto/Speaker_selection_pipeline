from __future__ import annotations

from ..config import LLMProviderConfig
from .base import LLMProvider


def create_provider(config: LLMProviderConfig) -> LLMProvider:
    """Instantiate the configured LLM provider."""
    name = config.provider.lower()
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(config)
    elif name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(config)
    elif name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider!r}")
