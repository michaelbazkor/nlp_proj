"""Clinical agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.few_shot import clinical_labels, format_few_shot_examples
from ssrisk.prompts.system import CLINICAL_SYSTEM
from ssrisk.schemas import ClinicalOutput


class ClinicalAgent(BaseAgent):
    name = "clinical"
    response_schema = ClinicalOutput

    def __init__(self, client):
        super().__init__(client, CLINICAL_SYSTEM)

    def build_few_shot(self, dev_examples: list) -> str:
        return format_few_shot_examples(
            dev_examples, self.name, clinical_labels, max_examples=3
        )
