"""Multi-agent pipeline orchestration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ssrisk.agents.clinical import ClinicalAgent
from ssrisk.agents.image_caption import ImageCaptionAgent
from ssrisk.agents.motivation import MotivationAgent
from ssrisk.agents.personality import PersonalityAgent
from ssrisk.agents.risk import RiskAgent
from ssrisk.data.loader import load_user_records
from ssrisk.data.schema import GROUND_TRUTH_MAP, UserRecord
from ssrisk.data.serialize import append_agent_output, build_user_context
from ssrisk.llm.factory import create_llm_client


def split_dev_test(
    users: list[UserRecord],
    dev_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[UserRecord], list[UserRecord]]:
    """Shuffle and split users into dev (few-shot) and test sets."""
    shuffled = users.copy()
    random.Random(seed).shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * dev_split))
    return shuffled[:split_idx], shuffled[split_idx:]


def run_user_pipeline(
    user: UserRecord,
    agents: dict[str, Any],
    dev_set: list[UserRecord],
) -> dict[str, Any]:
    """Run full agent chain for one user."""
    image_agent: ImageCaptionAgent = agents["image_caption"]
    motivation_agent: MotivationAgent = agents["motivation"]
    personality_agent: PersonalityAgent = agents["personality"]
    clinical_agent: ClinicalAgent = agents["clinical"]
    risk_agent: RiskAgent = agents["risk"]

    base_context = build_user_context(user)
    caption_result = image_agent.run(base_context)
    context = build_user_context(user, image_captions=caption_result.captions)

    motivation_res = motivation_agent.run(
        context, few_shot=motivation_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(context, "AGENT 1 - MOTIVATION ANALYSIS", motivation_res)

    personality_res = personality_agent.run(
        context, few_shot=personality_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(context, "AGENT 2 - PERSONALITY & DISTRESS", personality_res)

    clinical_res = clinical_agent.run(
        context, few_shot=clinical_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(context, "AGENT 3 - CLINICAL DIAGNOSIS", clinical_res)

    risk_res = risk_agent.run(context, few_shot=risk_agent.build_few_shot(dev_set))

    result: dict[str, Any] = {
        "UserId": user.user_id,
        "final_risk_analysis": risk_res.risk_analysis,
    }

    for true_col, pred_col in GROUND_TRUTH_MAP.items():
        result[f"true_{true_col}"] = user.labels.get(true_col)
        result[pred_col] = getattr(
            motivation_res if pred_col == "pred_FOMO"
            else personality_res if pred_col.startswith("pred_BFI") or pred_col in {
                "pred_Lonely", "pred_Brooding", "pred_Worry", "pred_SWL"
            }
            else clinical_res if pred_col.startswith("pred_PHQ") or pred_col == "pred_MDD"
            else risk_res,
            pred_col,
        )

    result["pred_primary_motivation"] = motivation_res.primary_motivation.value
    result["motivation_analysis"] = motivation_res.motivation_analysis
    result["personality_analysis"] = personality_res.personality_analysis
    result["clinical_analysis"] = clinical_res.clinical_analysis

    return result


def run_pipeline(config: dict[str, Any]) -> Path:
    """Execute pipeline on test users and save results CSV."""
    data_cfg = config.get("data", {})
    pipe_cfg = config.get("pipeline", {})
    seed = data_cfg.get("random_seed", 42)

    users = load_user_records(
        features_path=data_cfg.get("features_path", "data/synthetic_users.csv"),
        posts_path=data_cfg.get("posts_path", "data/posts.json"),
        min_posts=data_cfg.get("min_posts", 10),
        valid_groups=data_cfg.get("valid_groups", [0, 1]),
    )

    dev_set, test_set = split_dev_test(users, dev_split=pipe_cfg.get("dev_split", 0.1), seed=seed)
    max_test = pipe_cfg.get("max_test_users")
    if max_test:
        test_set = test_set[: int(max_test)]

    client = create_llm_client(config)
    agents = {
        "image_caption": ImageCaptionAgent(client),
        "motivation": MotivationAgent(client),
        "personality": PersonalityAgent(client),
        "clinical": ClinicalAgent(client),
        "risk": RiskAgent(client),
    }

    output_dir = Path(pipe_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for user in tqdm(test_set, desc="Processing users"):
        results.append(run_user_pipeline(user, agents, dev_set))

    import pandas as pd

    results_df = pd.DataFrame(results)
    out_path = output_dir / "pipeline_results.csv"
    results_df.to_csv(out_path, index=False)

    meta = {
        "n_users_total": len(users),
        "n_dev": len(dev_set),
        "n_test": len(test_set),
        "provider": config.get("llm", {}).get("provider", "mock"),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Pipeline complete. Results: {out_path}")
    return out_path
