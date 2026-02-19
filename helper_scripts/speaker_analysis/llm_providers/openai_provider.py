from __future__ import annotations

import os

from .base import LLMProvider


class OpenAIProvider(LLMProvider):

    @property
    def provider_name(self) -> str:
        return "openai"

    def classify_segments(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv(self.config.api_key_env))
        response = client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
