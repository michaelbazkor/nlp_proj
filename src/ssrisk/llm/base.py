"""LLM client abstract base class."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract interface for structured LLM completions."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_context: str,
        response_schema: Type[T],
    ) -> T:
        """Return a parsed Pydantic object matching response_schema."""
        ...
