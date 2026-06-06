"""Motivation agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.few_shot import format_few_shot_examples, motivation_labels
from ssrisk.prompts.system import MOTIVATION_SYSTEM
from ssrisk.schemas import MotivationOutput


class MotivationAgent(BaseAgent):
    name = "motivation"
    response_schema = MotivationOutput

    def __init__(self, client):
        super().__init__(client, MOTIVATION_SYSTEM)

    def build_few_shot(self, dev_examples: list) -> str:
        return format_few_shot_examples(
            dev_examples, self.name, motivation_labels, max_examples=3
        )
