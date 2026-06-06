"""OpenAI structured-output LLM client."""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

from ssrisk.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(LLMClient):
    """OpenAI chat completions with JSON schema structured outputs."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature

    def complete(
        self,
        system_prompt: str,
        user_context: str,
        response_schema: Type[T],
    ) -> T:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            response_format=response_schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned empty parsed response.")
        return parsed
