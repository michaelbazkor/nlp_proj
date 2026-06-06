"""Anthropic LLM client stub."""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

from ssrisk.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class AnthropicClient(LLMClient):
    """Anthropic Messages API with JSON extraction."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.temperature = temperature
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Install anthropic package to use Anthropic provider: pip install anthropic"
            ) from exc
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        user_context: str,
        response_schema: Type[T],
    ) -> T:
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        prompt = (
            f"{user_context}\n\n"
            f"Respond with valid JSON matching this schema:\n{schema_json}"
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise RuntimeError("Anthropic response did not contain JSON object.")
        data = json.loads(text[start:end])
        return response_schema.model_validate(data)
