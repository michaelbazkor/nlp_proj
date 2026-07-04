"""Psychosocial distress agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.few_shot import format_few_shot_examples, psychosocial_labels
from ssrisk.prompts.system import PSYCHOSOCIAL_SYSTEM
from ssrisk.schemas import PsychosocialOutput


class PsychosocialAgent(BaseAgent):
    name = "psychosocial"
    response_schema = PsychosocialOutput

    def __init__(self, client, context_budget=None):
        super().__init__(client, PSYCHOSOCIAL_SYSTEM, context_budget=context_budget)

    def build_few_shot(self, dev_examples: list) -> str:
        return format_few_shot_examples(
            dev_examples, self.name, psychosocial_labels, max_examples=3
        )
