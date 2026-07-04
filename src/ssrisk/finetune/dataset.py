"""SFT dataset export for per-agent fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ssrisk.context.budget import ContextBudget, load_context_budget
from ssrisk.data.loader import load_user_records
from ssrisk.data.schema import UserRecord
from ssrisk.data.serialize import append_agent_output, build_user_context
from ssrisk.pipeline import split_dev_test
from ssrisk.prompts.few_shot import (
    clinical_labels,
    motivation_labels,
    personality_labels,
    psychosocial_labels,
    risk_labels,
)
from ssrisk.prompts.system import (
    CLINICAL_SYSTEM,
    IMAGE_CAPTION_SYSTEM,
    MOTIVATION_SYSTEM,
    PERSONALITY_SYSTEM,
    PSYCHOSOCIAL_SYSTEM,
    RISK_SYSTEM,
)
from ssrisk.schemas import (
    ClinicalOutput,
    ImageCaptionBatchOutput,
    ImageCaptionOutput,
    MotivationCategory,
    MotivationOutput,
    PersonalityOutput,
    PsychosocialOutput,
    RiskOutput,
)

AGENT_EXPORT_ORDER = (
    "image_caption",
    "motivation",
    "personality",
    "psychosocial",
    "clinical",
    "risk",
)

AGENT_SYSTEM_PROMPTS = {
    "image_caption": IMAGE_CAPTION_SYSTEM,
    "motivation": MOTIVATION_SYSTEM,
    "personality": PERSONALITY_SYSTEM,
    "psychosocial": PSYCHOSOCIAL_SYSTEM,
    "clinical": CLINICAL_SYSTEM,
    "risk": RISK_SYSTEM,
}

AGENT_LABEL_EXTRACTORS: dict[str, Callable[[UserRecord], dict[str, Any]]] = {
    "motivation": motivation_labels,
    "personality": personality_labels,
    "psychosocial": psychosocial_labels,
    "clinical": clinical_labels,
    "risk": risk_labels,
}


def _context_limits(budget: ContextBudget) -> tuple[int | None, int | None, str]:
    return (
        budget.available_input_tokens // 2,
        budget.max_agent_output_tokens,
        budget.truncation_strategy,
    )


def _build_motivation_output(user: UserRecord) -> MotivationOutput:
    labels = motivation_labels(user)
    primary_str = labels.get("primary_motivation", "entertainment")
    try:
        primary = MotivationCategory(primary_str)
    except ValueError:
        primary = MotivationCategory.ENTERTAINMENT
    return MotivationOutput(
        primary_motivation=primary,
        pred_FOMO=int(labels["pred_FOMO"]),
        motivation_analysis="Ground-truth motivation profile from labeled data.",
    )


def _build_personality_output(user: UserRecord) -> PersonalityOutput:
    labels = personality_labels(user)
    return PersonalityOutput(
        pred_BFI_N=int(labels["pred_BFI_N"]),
        pred_BFI_E=int(labels["pred_BFI_E"]),
        pred_BFI_O=int(labels["pred_BFI_O"]),
        pred_BFI_A=int(labels["pred_BFI_A"]),
        pred_BFI_C=int(labels["pred_BFI_C"]),
        personality_analysis="Ground-truth Big Five profile from labeled data.",
    )


def _build_psychosocial_output(user: UserRecord) -> PsychosocialOutput:
    labels = psychosocial_labels(user)
    return PsychosocialOutput(
        pred_Lonely=int(labels["pred_Lonely"]),
        pred_Brooding=int(labels["pred_Brooding"]),
        pred_Worry=int(labels["pred_Worry"]),
        pred_SWL=int(labels["pred_SWL"]),
        psychosocial_analysis="Ground-truth psychosocial scales from labeled data.",
    )


def _build_clinical_output(user: UserRecord) -> ClinicalOutput:
    labels = clinical_labels(user)
    return ClinicalOutput(
        pred_PHQ9_1=int(user.labels.get("PHQ9_1", 0)),
        pred_PHQ9_2=int(user.labels.get("PHQ9_2", 0)),
        pred_PHQ9_3=int(user.labels.get("PHQ9_3", 0)),
        pred_PHQ9_4=int(user.labels.get("PHQ9_4", 0)),
        pred_PHQ9_5=int(user.labels.get("PHQ9_5", 0)),
        pred_PHQ9_6=int(user.labels.get("PHQ9_6", 0)),
        pred_PHQ9_7=int(user.labels.get("PHQ9_7", 0)),
        pred_PHQ9_8=int(user.labels.get("PHQ9_8", 0)),
        pred_PHQ9_9=int(user.labels.get("PHQ9_9", 0)),
        pred_MDD=int(labels["pred_MDD"]),
        clinical_analysis="Ground-truth PHQ-9 and MDD from labeled data.",
    )


def _build_risk_output(user: UserRecord) -> RiskOutput:
    labels = risk_labels(user)
    return RiskOutput(
        pred_SD=int(labels["pred_SD"]),
        risk_analysis="Ground-truth C-SSRS severity from labeled data.",
    )


def _build_image_caption_output(user: UserRecord) -> ImageCaptionBatchOutput:
    captions: list[ImageCaptionOutput] = []
    seen: set[str] = set()
    for post in user.posts:
        for img in post.get("images", []):
            img_id = str(img.get("image_id", img.get("id", "")))
            if not img_id or img_id in seen:
                continue
            seen.add(img_id)
            hint = img.get("hint", "social media image")
            captions.append(
                ImageCaptionOutput(
                    image_id=img_id,
                    caption=str(hint),
                    emotional_tone="neutral",
                )
            )
    if not captions:
        captions.append(ImageCaptionOutput(image_id="1", caption="No images", emotional_tone="neutral"))
    return ImageCaptionBatchOutput(captions=captions)


def build_agent_context(
    user: UserRecord,
    agent_name: str,
    budget: ContextBudget,
) -> str:
    """Build teacher-forced context for a given agent training example."""
    max_post_tokens, max_agent_output_tokens, truncation_strategy = _context_limits(budget)
    context = build_user_context(
        user,
        max_post_tokens=max_post_tokens,
        truncation_strategy=truncation_strategy,
    )

    if agent_name == "image_caption":
        return context

    captions = _build_image_caption_output(user).captions
    context = build_user_context(
        user,
        image_captions=captions,
        max_post_tokens=max_post_tokens,
        truncation_strategy=truncation_strategy,
    )

    if agent_name == "motivation":
        return context

    context = append_agent_output(
        context, "AGENT 1 - MOTIVATION ANALYSIS", _build_motivation_output(user), max_agent_output_tokens
    )
    if agent_name == "personality":
        return context

    context = append_agent_output(
        context, "AGENT 2 - PERSONALITY ANALYSIS", _build_personality_output(user), max_agent_output_tokens
    )
    if agent_name == "psychosocial":
        return context

    context = append_agent_output(
        context, "AGENT 3 - PSYCHOSOCIAL DISTRESS", _build_psychosocial_output(user), max_agent_output_tokens
    )
    if agent_name == "clinical":
        return context

    context = append_agent_output(
        context, "AGENT 4 - CLINICAL DIAGNOSIS", _build_clinical_output(user), max_agent_output_tokens
    )
    return context


def build_assistant_payload(agent_name: str, user: UserRecord) -> dict[str, Any]:
    """Ground-truth assistant JSON for one agent."""
    if agent_name == "image_caption":
        return _build_image_caption_output(user).model_dump()
    if agent_name == "motivation":
        return _build_motivation_output(user).model_dump()
    if agent_name == "personality":
        return _build_personality_output(user).model_dump()
    if agent_name == "psychosocial":
        return _build_psychosocial_output(user).model_dump()
    if agent_name == "clinical":
        return _build_clinical_output(user).model_dump()
    if agent_name == "risk":
        return _build_risk_output(user).model_dump()
    raise ValueError(f"Unknown agent: {agent_name}")


def export_agent_dataset(
    users: list[UserRecord],
    agent_name: str,
    output_path: Path,
    budget: ContextBudget,
) -> int:
    """Write JSONL SFT examples for one agent."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for user in users:
            if user.labels.get("SD") is None and agent_name != "image_caption":
                continue
            row = {
                "system": AGENT_SYSTEM_PROMPTS[agent_name],
                "user": build_agent_context(user, agent_name, budget),
                "assistant": build_assistant_payload(agent_name, user),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_all_datasets(config: dict[str, Any]) -> dict[str, Path]:
    """Export SFT JSONL files for all agents using the dev split."""
    data_cfg = config.get("data", {})
    pipe_cfg = config.get("pipeline", {})
    finetune_cfg = config.get("finetune", {})
    budget = load_context_budget(config)

    users = load_user_records(
        features_path=data_cfg.get("features_path", "data/synthetic_users.csv"),
        posts_path=data_cfg.get("posts_path", "data/posts.json"),
        posts_csv_path=data_cfg.get("posts_csv_path"),
        min_posts=data_cfg.get("min_posts", 10),
        valid_groups=data_cfg.get("valid_groups", [0, 1]),
        max_posts_per_user=data_cfg.get("max_posts_per_user"),
    )
    dev_set, _ = split_dev_test(
        users,
        dev_split=pipe_cfg.get("dev_split", 0.1),
        seed=data_cfg.get("random_seed", 42),
        min_dev_users=pipe_cfg.get("min_dev_users", 3),
    )

    output_dir = Path(finetune_cfg.get("output_dir", "outputs/finetune"))
    paths: dict[str, Path] = {}
    for agent_name in AGENT_EXPORT_ORDER:
        out_path = output_dir / f"{agent_name}.jsonl"
        n = export_agent_dataset(dev_set, agent_name, out_path, budget)
        paths[agent_name] = out_path
        print(f"Exported {n} examples -> {out_path}")
    return paths
