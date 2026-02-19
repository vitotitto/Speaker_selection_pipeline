from __future__ import annotations

import os

from .base import LLMProvider


class ClaudeProvider(LLMProvider):

    @property
    def provider_name(self) -> str:
        return "claude"

    def classify_segments(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv(self.config.api_key_env))
        message = client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
