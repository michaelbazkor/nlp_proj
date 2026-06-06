"""Few-shot example formatting for agents."""

from __future__ import annotations

from typing import Any

from ssrisk.data.schema import UserRecord
from ssrisk.data.serialize import build_user_context


def format_few_shot_examples(
    examples: list[UserRecord],
    agent_role: str,
    label_extractor: Any,
    max_examples: int = 3,
) -> str:
    """Build few-shot block from dev-set users with ground-truth labels."""
    if not examples:
        return ""

    lines = [f"\n[FEW-SHOT EXAMPLES FOR {agent_role.upper()}]"]
    for ex in examples[:max_examples]:
        ctx = build_user_context(ex)
        labels = label_extractor(ex)
        lines.append(f"\n--- Example User {ex.user_id} ---")
        lines.append(ctx[:1500] + ("..." if len(ctx) > 1500 else ""))
        lines.append(f"Expected output: {labels}")
    return "\n".join(lines)


def motivation_labels(user: UserRecord) -> dict[str, Any]:
    return {
        "primary_motivation": "self_expression_venting" if user.labels.get("SD", 0) >= 1 else "entertainment",
        "pred_FOMO": user.labels.get("FOMO"),
    }


def personality_labels(user: UserRecord) -> dict[str, Any]:
    return {
        "pred_BFI_N": user.labels.get("BFI_N"),
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
