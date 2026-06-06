"""Pydantic schemas for structured LLM agent outputs."""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class MotivationCategory(str, Enum):
    VALIDATION_SEEKING = "validation_seeking"
    SOCIAL_CONNECTION = "social_connection"
    SELF_EXPRESSION_VENTING = "self_expression_venting"
    INFORMATION_SHARING = "information_sharing"
    ENTERTAINMENT = "entertainment"
    FOMO_DRIVEN = "fomo_driven"
    PROFESSIONAL_SELF_PROMOTION = "professional_self_promotion"


class ImageCaptionOutput(BaseModel):
    image_id: str = Field(description="Identifier of the image being described")
    caption: str = Field(description="Natural-language description of the image content and mood")
    emotional_tone: str = Field(description="Dominant emotional tone inferred from the image")


class ImageCaptionBatchOutput(BaseModel):
    captions: List[ImageCaptionOutput] = Field(description="Captions for all user images")


class MotivationOutput(BaseModel):
    primary_motivation: MotivationCategory = Field(
        description="Primary motivation category for social media use"
    )
    secondary_motivations: List[MotivationCategory] = Field(
        default_factory=list,
        description="Additional motivation categories (0-2)",
    )
    pred_FOMO: int = Field(ge=10, le=40, description="Predicted Fear of Missing Out score (10-40)")
    motivation_analysis: str = Field(
        description="Clinical analysis of why the user posts (validation, support, etc.)"
    )


class PersonalityOutput(BaseModel):
    pred_BFI_N: int = Field(ge=1, le=10, description="Predicted Neuroticism score (Big Five, 1-10)")
    pred_BFI_E: int = Field(ge=1, le=10, description="Predicted Extraversion score (Big Five, 1-10)")
    pred_BFI_O: int = Field(ge=1, le=10, description="Predicted Openness score (Big Five, 1-10)")
    pred_BFI_A: int = Field(ge=1, le=10, description="Predicted Agreeableness score (Big Five, 1-10)")
    pred_BFI_C: int = Field(ge=1, le=10, description="Predicted Conscientiousness score (Big Five, 1-10)")
    pred_Lonely: int = Field(ge=10, le=40, description="Predicted UCLA Loneliness score (10-40)")
    pred_Brooding: int = Field(ge=5, le=20, description="Predicted Brooding/Rumination score (5-20)")
    pred_Worry: int = Field(ge=16, le=80, description="Predicted PSWQ Worry score (16-80)")
    pred_SWL: int = Field(ge=5, le=35, description="Predicted Satisfaction With Life score (5-35)")
    personality_analysis: str = Field(
        description="Analysis of traits and psychosocial distress based on motivation"
    )


class ClinicalOutput(BaseModel):
    pred_PHQ9_1: int = Field(ge=0, le=3, description="PHQ-9 Item 1: Anhedonia (0-3)")
    pred_PHQ9_2: int = Field(ge=0, le=3, description="PHQ-9 Item 2: Depressed mood (0-3)")
    pred_PHQ9_3: int = Field(ge=0, le=3, description="PHQ-9 Item 3: Sleep issues (0-3)")
    pred_PHQ9_4: int = Field(ge=0, le=3, description="PHQ-9 Item 4: Fatigue (0-3)")
    pred_PHQ9_5: int = Field(ge=0, le=3, description="PHQ-9 Item 5: Appetite changes (0-3)")
    pred_PHQ9_6: int = Field(ge=0, le=3, description="PHQ-9 Item 6: Guilt/Worthlessness (0-3)")
    pred_PHQ9_7: int = Field(ge=0, le=3, description="PHQ-9 Item 7: Concentration issues (0-3)")
    pred_PHQ9_8: int = Field(ge=0, le=3, description="PHQ-9 Item 8: Psychomotor agitation/retardation (0-3)")
    pred_PHQ9_9: int = Field(ge=0, le=3, description="PHQ-9 Item 9: Suicidal ideation (0-3)")
    pred_MDD: int = Field(ge=0, le=1, description="Binary indicator for Major Depressive Disorder (0=No, 1=Yes)")
    clinical_analysis: str = Field(description="Psychiatric formulation of mood disorders")


class RiskOutput(BaseModel):
    pred_SD: int = Field(ge=0, le=6, description="Predicted CSSRS suicide severity (0-6)")
    risk_analysis: str = Field(description="Final C-SSRS based risk assessment justification")
