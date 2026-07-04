"""Multi-agent pipeline orchestration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ssrisk.agents.clinical import ClinicalAgent
from ssrisk.agents.image_caption import ImageCaptionAgent
from ssrisk.agents.motivation import MotivationAgent
from ssrisk.agents.personality import PersonalityAgent
from ssrisk.agents.psychosocial import PsychosocialAgent
from ssrisk.agents.risk import RiskAgent
from ssrisk.context.budget import ContextBudget, load_context_budget
from ssrisk.data.chain import AgentChainResult
from ssrisk.data.loader import load_user_records
from ssrisk.data.schema import GROUND_TRUTH_MAP, UserRecord
from ssrisk.data.serialize import append_agent_output, build_user_context
from ssrisk.llm.agent_factory import create_agent_clients


def split_dev_test(
    users: list[UserRecord],
    dev_split: float = 0.1,
    seed: int = 42,
    min_dev_users: int = 3,
) -> tuple[list[UserRecord], list[UserRecord]]:
    """Shuffle and split users into dev (few-shot) and test sets."""
    shuffled = users.copy()
    random.Random(seed).shuffle(shuffled)
    split_idx = max(min_dev_users, int(len(shuffled) * dev_split))
    if split_idx >= len(shuffled):
        split_idx = max(1, len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def create_agents(
    config: dict[str, Any],
    context_budget: ContextBudget | None = None,
) -> dict[str, Any]:
    """Instantiate all pipeline agents with per-agent LLM clients."""
    clients = create_agent_clients(config)
    budget = context_budget

    return {
        "image_caption": ImageCaptionAgent(clients["image_caption"], context_budget=budget),
        "motivation": MotivationAgent(clients["motivation"], context_budget=budget),
        "personality": PersonalityAgent(clients["personality"], context_budget=budget),
        "psychosocial": PsychosocialAgent(clients["psychosocial"], context_budget=budget),
        "clinical": ClinicalAgent(clients["clinical"], context_budget=budget),
        "risk": RiskAgent(clients["risk"], context_budget=budget),
    }


def run_agent_chain(
    user: UserRecord,
    agents: dict[str, Any],
    dev_set: list[UserRecord],
    context_budget: ContextBudget | None = None,
) -> AgentChainResult:
    """Run agents 0-5 for one user."""
    image_agent: ImageCaptionAgent = agents["image_caption"]
    motivation_agent: MotivationAgent = agents["motivation"]
    personality_agent: PersonalityAgent = agents["personality"]
    psychosocial_agent: PsychosocialAgent = agents["psychosocial"]
    clinical_agent: ClinicalAgent = agents["clinical"]
    risk_agent: RiskAgent = agents["risk"]

    max_post_tokens = None
    max_agent_output_tokens = None
    truncation_strategy = "tail_posts"
    if context_budget is not None:
        max_post_tokens = context_budget.available_input_tokens // 2
        max_agent_output_tokens = context_budget.max_agent_output_tokens
        truncation_strategy = context_budget.truncation_strategy

    base_context = build_user_context(
        user,
        max_post_tokens=max_post_tokens,
        truncation_strategy=truncation_strategy,
    )
    caption_result = image_agent.run(base_context)
    context = build_user_context(
        user,
        image_captions=caption_result.captions,
        max_post_tokens=max_post_tokens,
        truncation_strategy=truncation_strategy,
    )

    motivation_res = motivation_agent.run(
        context, few_shot=motivation_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(
        context, "AGENT 1 - MOTIVATION ANALYSIS", motivation_res, max_agent_output_tokens
    )

    personality_res = personality_agent.run(
        context, few_shot=personality_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(
        context, "AGENT 2 - PERSONALITY ANALYSIS", personality_res, max_agent_output_tokens
    )

    psychosocial_res = psychosocial_agent.run(
        context, few_shot=psychosocial_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(
        context, "AGENT 3 - PSYCHOSOCIAL DISTRESS", psychosocial_res, max_agent_output_tokens
    )

    clinical_res = clinical_agent.run(
        context, few_shot=clinical_agent.build_few_shot(dev_set)
    )
    context = append_agent_output(
        context, "AGENT 4 - CLINICAL DIAGNOSIS", clinical_res, max_agent_output_tokens
    )

    risk_res = risk_agent.run(context, few_shot=risk_agent.build_few_shot(dev_set))

    return AgentChainResult(
        captions=caption_result.captions,
        motivation=motivation_res,
        personality=personality_res,
        psychosocial=psychosocial_res,
        clinical=clinical_res,
        risk=risk_res,
    )


def _resolve_pred_source(pred_col: str, chain: AgentChainResult, pred_sd: int):
    """Map prediction column name to the agent output object."""
    if pred_col == "pred_FOMO":
        return chain.motivation
    if pred_col.startswith("pred_BFI"):
        return chain.personality
    if pred_col in {"pred_Lonely", "pred_Brooding", "pred_Worry", "pred_SWL"}:
        return chain.psychosocial
    if pred_col.startswith("pred_PHQ") or pred_col == "pred_MDD":
        return chain.clinical
    if pred_col == "pred_SD":
        return pred_sd
    raise KeyError(f"Unknown prediction column: {pred_col}")


def build_user_result(
    user: UserRecord,
    chain: AgentChainResult,
    pred_sd: int,
) -> dict[str, Any]:
    """Assemble pipeline result row with ground truth and agent predictions."""
    result: dict[str, Any] = {"UserId": user.user_id}

    for true_col, pred_col in GROUND_TRUTH_MAP.items():
        result[f"true_{true_col}"] = user.labels.get(true_col)
        source = _resolve_pred_source(pred_col, chain, pred_sd)
        if pred_col == "pred_SD":
            result[pred_col] = pred_sd
        else:
            result[pred_col] = getattr(source, pred_col)

    result["pred_primary_motivation"] = chain.motivation.primary_motivation.value
    result["motivation_analysis"] = chain.motivation.motivation_analysis
    result["personality_analysis"] = chain.personality.personality_analysis
    result["psychosocial_analysis"] = chain.psychosocial.psychosocial_analysis
    result["clinical_analysis"] = chain.clinical.clinical_analysis
    if chain.risk is not None:
        result["risk_analysis"] = chain.risk.risk_analysis

    return result


def run_user_pipeline(
    user: UserRecord,
    agents: dict[str, Any],
    dev_set: list[UserRecord],
    context_budget: ContextBudget | None = None,
) -> dict[str, Any]:
    """Run full agent chain and predict suicide severity via RiskAgent."""
    chain = run_agent_chain(user, agents, dev_set, context_budget=context_budget)
    if chain.risk is None:
        raise ValueError("Risk agent output missing.")
    pred_sd = int(chain.risk.pred_SD)
    return build_user_result(user, chain, pred_sd)


def run_pipeline(config: dict[str, Any]) -> Path:
    """Execute pipeline: agent chain on test users with dev-set few-shot."""
    data_cfg = config.get("data", {})
    pipe_cfg = config.get("pipeline", {})
    seed = data_cfg.get("random_seed", 42)
    context_budget = load_context_budget(config)

    users = load_user_records(
        features_path=data_cfg.get("features_path", "data/synthetic_users.csv"),
        posts_path=data_cfg.get("posts_path", "data/posts.json"),
        posts_csv_path=data_cfg.get("posts_csv_path"),
        min_posts=data_cfg.get("min_posts", 10),
        valid_groups=data_cfg.get("valid_groups", [0, 1]),
        max_posts_per_user=data_cfg.get("max_posts_per_user"),
    )

    dev_set, test_set = split_dev_test(
        users,
        dev_split=pipe_cfg.get("dev_split", 0.1),
        seed=seed,
        min_dev_users=pipe_cfg.get("min_dev_users", 3),
    )
    max_test = pipe_cfg.get("max_test_users")
    if max_test:
        test_set = test_set[: int(max_test)]

    agents = create_agents(config, context_budget=context_budget)

    output_dir = Path(pipe_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for user in tqdm(test_set, desc="Test set (agents + Risk)"):
        results.append(
            run_user_pipeline(user, agents, dev_set, context_budget=context_budget)
        )

    results_df = pd.DataFrame(results)
    out_path = output_dir / "pipeline_results.csv"
    results_df.to_csv(out_path, index=False)

    meta = {
        "n_users_total": len(users),
        "n_dev": len(dev_set),
        "n_test": len(test_set),
        "provider": config.get("llm", {}).get("provider", "mock"),
        "pipeline": "llm_risk",
        "data_source": data_cfg.get("posts_csv_path")
        or data_cfg.get("features_path", "data/synthetic_users.csv"),
        "max_posts_per_user": data_cfg.get("max_posts_per_user"),
    }

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Pipeline complete. Results: {out_path}")
    return out_path
