"""Few-shot example formatting for agents."""

from __future__ import annotations

from typing import Any

from ssrisk.context.budget import truncate_to_budget
from ssrisk.data.schema import UserRecord
from ssrisk.data.serialize import build_user_context


def user_has_ground_truth(user: UserRecord) -> bool:
    """Return True when psychometric labels are available for few-shot examples."""
    return user.labels.get("SD") is not None


def format_few_shot_examples(
    examples: list[UserRecord],
    agent_role: str,
    label_extractor: Any,
    max_examples: int = 3,
    max_context_tokens: int = 1500,
) -> str:
    """Build few-shot block from dev-set users with ground-truth labels."""
    labeled = [ex for ex in examples if user_has_ground_truth(ex)]
    if not labeled:
        return ""

    lines = [f"\n[FEW-SHOT EXAMPLES FOR {agent_role.upper()}]"]
    for ex in labeled[:max_examples]:
        ctx = build_user_context(ex)
        labels = label_extractor(ex)
        lines.append(f"\n--- Example User {ex.user_id} ---")
        lines.append(truncate_to_budget(ctx, max_context_tokens, strategy="tail"))
        lines.append(f"Expected output: {labels}")
    return "\n".join(lines)


def motivation_labels(user: UserRecord) -> dict[str, Any]:
    sd = user.labels.get("SD") or 0
    return {
        "primary_motivation": "self_expression_venting" if sd >= 1 else "entertainment",
        "pred_FOMO": user.labels.get("FOMO"),
    }


def personality_labels(user: UserRecord) -> dict[str, Any]:
    return {
        "pred_BFI_N": user.labels.get("BFI_N"),
        "pred_BFI_E": user.labels.get("BFI_E"),
        "pred_BFI_O": user.labels.get("BFI_O"),
        "pred_BFI_A": user.labels.get("BFI_A"),
        "pred_BFI_C": user.labels.get("BFI_C"),
    }


def psychosocial_labels(user: UserRecord) -> dict[str, Any]:
    return {
        "pred_Lonely": user.labels.get("Lonely"),
        "pred_Brooding": user.labels.get("Brooding"),
        "pred_Worry": user.labels.get("Worry"),
        "pred_SWL": user.labels.get("SWL"),
    }


def clinical_labels(user: UserRecord) -> dict[str, Any]:
    return {
        "pred_PHQ9_1": user.labels.get("PHQ9_1"),
        "pred_PHQ9_9": user.labels.get("PHQ9_9"),
        "pred_MDD": user.labels.get("MDD"),
    }


def risk_labels(user: UserRecord) -> dict[str, Any]:
    return {"pred_SD": user.labels.get("SD")}
