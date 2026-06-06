"""Image caption agent."""

from __future__ import annotations

from ssrisk.agents.base import BaseAgent
from ssrisk.prompts.system import IMAGE_CAPTION_SYSTEM
from ssrisk.schemas import ImageCaptionBatchOutput


class ImageCaptionAgent(BaseAgent):
    name = "image_caption"
    response_schema = ImageCaptionBatchOutput

    def __init__(self, client):
        super().__init__(client, IMAGE_CAPTION_SYSTEM)

    def build_few_shot(self, dev_examples: list) -> str:
        return ""
