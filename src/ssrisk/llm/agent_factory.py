"""Per-agent LLM client factory."""

from __future__ import annotations

import os
from typing import Any

from ssrisk.llm.base import LLMClient
from ssrisk.llm.factory import create_llm_client

AGENT_NAMES = (
    "image_caption",
    "motivation",
    "personality",
    "psychosocial",
    "clinical",
    "risk",
)


def create_agent_clients(config: dict[str, Any] | None = None) -> dict[str, LLMClient]:
    """Create one LLM client per agent, with per-agent model/temperature overrides."""
    cfg = config or {}
    llm_cfg = cfg.get("llm", {})
    agents_cfg = cfg.get("agents", {})
    provider = (os.getenv("LLM_PROVIDER") or llm_cfg.get("provider", "mock")).lower()
    default_model = llm_cfg.get("model")
    default_temperature = float(llm_cfg.get("temperature", 0.2))

    clients: dict[str, LLMClient] = {}
    for name in AGENT_NAMES:
        agent_cfg = agents_cfg.get(name, {}) or {}
        model = agent_cfg.get("model") or default_model
        temperature = float(agent_cfg.get("temperature", default_temperature))

        if provider == "mock":
            clients[name] = create_llm_client({"llm": {"provider": "mock"}})
            continue

        merged = {
            "llm": {
                "provider": provider,
                "model": model,
                "temperature": temperature,
            }
        }
        clients[name] = create_llm_client(merged)

    return clients
