"""Personality agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.few_shot import format_few_shot_examples, personality_labels
from ssrisk.prompts.system import PERSONALITY_SYSTEM
from ssrisk.schemas import PersonalityOutput


class PersonalityAgent(BaseAgent):
    name = "personality"
    response_schema = PersonalityOutput

    def __init__(self, client):
        super().__init__(client, PERSONALITY_SYSTEM)

    def build_few_shot(self, dev_examples: list) -> str:
        return format_few_shot_examples(
            dev_examples, self.name, personality_labels, max_examples=3
        )
