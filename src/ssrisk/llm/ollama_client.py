"""Ollama LLM client."""

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
        payload = {
            "model": self.model,
            "stream": False,
            "format": response_schema.model_json_schema(),
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
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
        data = json.loads(text)
        return response_schema.model_validate(data)
