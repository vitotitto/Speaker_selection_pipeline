from __future__ import annotations

import json
import os

from .base import LLMProvider


class GeminiProvider(LLMProvider):

    @property
    def provider_name(self) -> str:
        return "gemini"

    def classify_segments(self, prompt: str) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv(self.config.api_key_env))
            model = genai.GenerativeModel(
                self.config.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            return self._classify_via_rest(prompt)

    def _classify_via_rest(self, prompt: str) -> str:
        """Fallback using REST API when google-generativeai SDK is not installed."""
        import requests

        api_key = os.getenv(self.config.api_key_env)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model_name}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
