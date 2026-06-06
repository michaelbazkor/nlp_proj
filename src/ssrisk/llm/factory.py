"""LLM client factory."""

from __future__ import annotations

import os
from typing import Any

from ssrisk.llm.base import LLMClient
from ssrisk.llm.mock import MockLLMClient


def create_llm_client(config: dict[str, Any] | None = None) -> LLMClient:
    """Instantiate an LLM client from config and environment."""
    cfg = config or {}
    llm_cfg = cfg.get("llm", {})
    provider = (os.getenv("LLM_PROVIDER") or llm_cfg.get("provider", "mock")).lower()
    model = llm_cfg.get("model")
    temperature = float(llm_cfg.get("temperature", 0.2))

    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        from ssrisk.llm.openai_client import OpenAIClient

        return OpenAIClient(model=model, temperature=temperature)
    if provider == "anthropic":
        from ssrisk.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model, temperature=temperature)
    if provider == "ollama":
        from ssrisk.llm.ollama_client import OllamaClient

        return OllamaClient(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM provider: {provider}. Use mock|openai|anthropic|ollama.")
