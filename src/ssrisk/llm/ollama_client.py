"""Ollama LLM client stub."""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

import urllib.request
from pydantic import BaseModel

from ssrisk.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):
    """Local Ollama chat API with JSON extraction."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature

    def complete(
        self,
        system_prompt: str,
        user_context: str,
        response_schema: Type[T],
    ) -> T:
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_context}\n\n"
                        f"Respond ONLY with JSON matching:\n{schema_json}"
                    ),
                },
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        text = body["message"]["content"]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise RuntimeError("Ollama response did not contain JSON object.")
        data = json.loads(text[start:end])
        return response_schema.model_validate(data)
