"""Mock LLM client with deterministic heuristic outputs."""

from __future__ import annotations

import hashlib
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from ssrisk.llm.base import LLMClient
from ssrisk.schemas import (
    ClinicalOutput,
    ImageCaptionBatchOutput,
    ImageCaptionOutput,
    MotivationCategory,
    MotivationOutput,
    PersonalityOutput,
    RiskOutput,
)

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(LLMClient):
    """Offline client that infers structured outputs from text heuristics."""

    def complete(
        self,
        system_prompt: str,
        user_context: str,
        response_schema: Type[T],
    ) -> T:
        seed = int(hashlib.md5(user_context.encode()).hexdigest()[:8], 16)
        distress = self._distress_score(user_context)

        if response_schema is ImageCaptionBatchOutput:
            return self._image_captions(user_context, seed)  # type: ignore[return-value]
        if response_schema is MotivationOutput:
            return self._motivation(distress, seed)  # type: ignore[return-value]
        if response_schema is PersonalityOutput:
            return self._personality(distress, seed)  # type: ignore[return-value]
        if response_schema is ClinicalOutput:
            return self._clinical(distress, seed)  # type: ignore[return-value]
        if response_schema is RiskOutput:
            return self._risk(distress, seed)  # type: ignore[return-value]

        raise ValueError(f"MockLLMClient does not support schema: {response_schema.__name__}")

    @staticmethod
    def _distress_score(text: str) -> float:
        negative = len(
            re.findall(
                r"\b(sad|alone|lonely|depressed|hopeless|tired|pain|hurt|cry|"
                r"worthless|anxious|worried|empty|numb|suicide|die|kill)\b",
                text.lower(),
            )
        )
        positive = len(
            re.findall(
                r"\b(happy|grateful|blessed|love|friends|family|church|"
                r"celebration|wedding|great|wonderful|excited)\b",
                text.lower(),
            )
        )
        raw = (negative - positive * 0.5) / max(len(text.split()) / 50, 1)
        return max(0.0, min(1.0, 0.15 + raw * 0.35))

    @staticmethod
    def _scale(value: float, lo: int, hi: int, seed: int, jitter: int = 3) -> int:
        mid = lo + (hi - lo) * value
        offset = (seed % (2 * jitter + 1)) - jitter
        return int(max(lo, min(hi, round(mid + offset))))

    def _image_captions(self, context: str, seed: int) -> ImageCaptionBatchOutput:
        ids = re.findall(r"\[Image ([^\]]+)\]", context)
        if not ids:
            ids = ["1"]
        distress = self._distress_score(context)
        captions = []
        tones = ["neutral", "positive", "melancholic", "anxious", "social"]
        for i, img_id in enumerate(ids[:5]):
            tone_idx = int((distress * 4 + seed + i) % 5)
            tone = tones[tone_idx]
            if tone in ("melancholic", "anxious"):
                caption = f"A dimly lit photo with subdued colors suggesting {tone} mood."
            elif tone == "positive":
                caption = "A bright photo showing people smiling at a social gathering."
            elif tone == "social":
                caption = "A casual group photo with friends outdoors."
            else:
                caption = "An everyday snapshot with no strong emotional cues."
            captions.append(
                ImageCaptionOutput(image_id=img_id, caption=caption, emotional_tone=tone)
            )
        return ImageCaptionBatchOutput(captions=captions)

    def _motivation(self, distress: float, seed: int) -> MotivationOutput:
        if distress > 0.55:
            primary = MotivationCategory.SELF_EXPRESSION_VENTING
            secondary = [MotivationCategory.VALIDATION_SEEKING, MotivationCategory.SOCIAL_CONNECTION]
        elif distress > 0.35:
            primary = MotivationCategory.SOCIAL_CONNECTION
            secondary = [MotivationCategory.FOMO_DRIVEN]
        else:
            primary = MotivationCategory.ENTERTAINMENT
            secondary = [MotivationCategory.INFORMATION_SHARING]

        fomo = self._scale(distress, 10, 40, seed)
        return MotivationOutput(
            primary_motivation=primary,
            secondary_motivations=secondary[:2],
            pred_FOMO=fomo,
            motivation_analysis=(
                f"User appears motivated primarily by {primary.value.replace('_', ' ')}. "
                f"Posting patterns suggest {'elevated' if distress > 0.4 else 'moderate'} "
                "engagement with social validation and connection themes."
            ),
        )

    def _personality(self, distress: float, seed: int) -> PersonalityOutput:
        neuroticism = self._scale(distress, 1, 10, seed)
        extraversion = self._scale(1 - distress * 0.8, 1, 10, seed + 1)
        return PersonalityOutput(
            pred_BFI_N=neuroticism,
            pred_BFI_E=extraversion,
            pred_BFI_O=self._scale(0.5, 1, 10, seed + 2),
            pred_BFI_A=self._scale(0.55, 1, 10, seed + 3),
            pred_BFI_C=self._scale(0.5, 1, 10, seed + 4),
            pred_Lonely=self._scale(distress, 10, 40, seed + 5),
            pred_Brooding=self._scale(distress, 5, 20, seed + 6),
            pred_Worry=self._scale(distress, 16, 80, seed + 7),
            pred_SWL=self._scale(1 - distress, 5, 35, seed + 8),
            personality_analysis=(
                "Elevated neuroticism and loneliness indicators align with distress-laden language. "
                "Life satisfaction appears reduced relative to baseline norms."
                if distress > 0.4
                else "Personality profile suggests moderate emotional stability with typical social engagement."
            ),
        )

    def _clinical(self, distress: float, seed: int) -> ClinicalOutput:
        items = [self._scale(distress, 0, 3, seed + i, jitter=1) for i in range(9)]
        phq9_total = sum(items)
        mdd = 1 if phq9_total >= 10 or items[1] >= 2 else 0
        return ClinicalOutput(
            pred_PHQ9_1=items[0],
            pred_PHQ9_2=items[1],
            pred_PHQ9_3=items[2],
            pred_PHQ9_4=items[3],
            pred_PHQ9_5=items[4],
            pred_PHQ9_6=items[5],
            pred_PHQ9_7=items[6],
            pred_PHQ9_8=items[7],
            pred_PHQ9_9=items[8],
            pred_MDD=mdd,
            clinical_analysis=(
                f"PHQ-9 pattern suggests {'probable MDD' if mdd else 'subthreshold depressive symptoms'}. "
                f"Item 9 (suicidal ideation) scored {items[8]}/3."
            ),
        )

    def _risk(self, distress: float, seed: int) -> RiskOutput:
        sd = self._scale(distress, 0, 6, seed, jitter=1)
        if sd == 0 and distress > 0.25:
            sd = 1
        return RiskOutput(
            pred_SD=sd,
            risk_analysis=(
                f"CSSRS severity estimate: {sd}/6. "
                + (
                    "Passive ideation or elevated distress markers present in language."
                    if sd >= 1
                    else "No clear suicidal ideation detected from accumulated profile."
                )
            ),
        )
