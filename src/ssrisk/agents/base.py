"""Base agent class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from ssrisk.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """Agent that calls an LLM with system prompt + user context."""

    name: str = "base"
    response_schema: Type[BaseModel]

    def __init__(self, client: LLMClient, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt

    def run(self, user_context: str, few_shot: str = "") -> BaseModel:
        system = self.system_prompt
        if few_shot:
            system = f"{system}\n{few_shot}"
        return self.client.complete(system, user_context, self.response_schema)

    @abstractmethod
    def build_few_shot(self, dev_examples: list) -> str:
        ...
