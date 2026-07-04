"""Agent chain result container."""

from __future__ import annotations

from dataclasses import dataclass

from ssrisk.schemas import (
    ClinicalOutput,
    ImageCaptionOutput,
    MotivationOutput,
    PersonalityOutput,
    PsychosocialOutput,
    RiskOutput,
)


@dataclass
class AgentChainResult:
    """Outputs from agents 0-5 for one user."""

    captions: list[ImageCaptionOutput]
    motivation: MotivationOutput
    personality: PersonalityOutput
    psychosocial: PsychosocialOutput
    clinical: ClinicalOutput
    risk: RiskOutput | None = None
