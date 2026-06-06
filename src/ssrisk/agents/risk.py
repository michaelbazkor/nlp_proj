"""Risk assessment agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.few_shot import format_few_shot_examples, risk_labels
from ssrisk.prompts.system import RISK_SYSTEM
from ssrisk.schemas import RiskOutput


class RiskAgent(BaseAgent):
    name = "risk"
    response_schema = RiskOutput

    def __init__(self, client):
        super().__init__(client, RISK_SYSTEM)

    def build_few_shot(self, dev_examples: list) -> str:
        return format_few_shot_examples(
            dev_examples, self.name, risk_labels, max_examples=3
        )
