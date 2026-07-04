"""Tests for SSRisk pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ssrisk.context.budget import ContextBudget, estimate_tokens, truncate_posts_in_context
from ssrisk.data.schema import UserRecord
from ssrisk.data.serialize import build_user_context
from ssrisk.data.synthetic import generate_synthetic_users
from ssrisk.evaluation import evaluate_results
from ssrisk.llm.mock import MockLLMClient
from ssrisk.pipeline import create_agents, run_agent_chain, run_user_pipeline, split_dev_test
from ssrisk.schemas import (
    ClinicalOutput,
    MotivationCategory,
    MotivationOutput,
    PersonalityOutput,
    PsychosocialOutput,
)


@pytest.fixture
def sample_user() -> UserRecord:
    df, posts_map = generate_synthetic_users(n_users=5, random_seed=0)
    row = df.iloc[0].to_dict()
    uid = str(row["UserId"])
    return UserRecord.from_row(row, posts=posts_map[uid])


def test_schemas_validate():
    MotivationOutput(
        primary_motivation=MotivationCategory.ENTERTAINMENT,
        pred_FOMO=20,
        motivation_analysis="Test analysis.",
    )
    PersonalityOutput(
        pred_BFI_N=5,
        pred_BFI_E=6,
        pred_BFI_O=5,
        pred_BFI_A=6,
        pred_BFI_C=5,
        personality_analysis="Test.",
    )
    PsychosocialOutput(
        pred_Lonely=20,
        pred_Brooding=10,
        pred_Worry=30,
        pred_SWL=25,
        psychosocial_analysis="Test.",
    )
    ClinicalOutput(
        pred_PHQ9_1=0,
        pred_PHQ9_2=1,
        pred_PHQ9_3=0,
        pred_PHQ9_4=1,
        pred_PHQ9_5=0,
        pred_PHQ9_6=0,
        pred_PHQ9_7=0,
        pred_PHQ9_8=0,
        pred_PHQ9_9=0,
        pred_MDD=0,
        clinical_analysis="Test.",
    )


def test_serializer_includes_sections(sample_user: UserRecord):
    ctx = build_user_context(sample_user)
    assert "[RAW USER PROFILE]" in ctx
    assert "[CHRONOLOGICAL POSTS]" in ctx
    assert sample_user.user_id in ctx


def test_mock_llm_returns_valid_schema(sample_user: UserRecord):
    client = MockLLMClient()
    ctx = build_user_context(sample_user)
    result = client.complete("system", ctx, PsychosocialOutput)
    assert 10 <= result.pred_Lonely <= 40


def test_split_dev_test():
    df, posts_map = generate_synthetic_users(n_users=20, random_seed=1)
    users = [
        UserRecord.from_row(df.iloc[i].to_dict(), posts=posts_map[str(df.iloc[i]["UserId"])])
        for i in range(len(df))
    ]
    dev, test = split_dev_test(users, dev_split=0.1, seed=42)
    assert len(dev) >= 1
    assert len(test) == len(users) - len(dev)


def _make_agents(client: MockLLMClient) -> dict:
    from ssrisk.agents.clinical import ClinicalAgent
    from ssrisk.agents.image_caption import ImageCaptionAgent
    from ssrisk.agents.motivation import MotivationAgent
    from ssrisk.agents.personality import PersonalityAgent
    from ssrisk.agents.psychosocial import PsychosocialAgent
    from ssrisk.agents.risk import RiskAgent

    return {
        "image_caption": ImageCaptionAgent(client),
        "motivation": MotivationAgent(client),
        "personality": PersonalityAgent(client),
        "psychosocial": PsychosocialAgent(client),
        "clinical": ClinicalAgent(client),
        "risk": RiskAgent(client),
    }


def test_pipeline_agent_order(sample_user: UserRecord):
    client = MockLLMClient()
    agents = _make_agents(client)
    df, posts_map = generate_synthetic_users(n_users=5, random_seed=2)
    dev_users = [
        UserRecord.from_row(df.iloc[i].to_dict(), posts=posts_map[str(df.iloc[i]["UserId"])])
        for i in range(2)
    ]
    chain = run_agent_chain(sample_user, agents, dev_users)
    assert chain.motivation is not None
    assert chain.personality is not None
    assert chain.psychosocial is not None
    assert chain.clinical is not None
    assert chain.risk is not None


def test_pipeline_end_to_end(sample_user: UserRecord):
    client = MockLLMClient()
    agents = _make_agents(client)
    df, posts_map = generate_synthetic_users(n_users=5, random_seed=2)
    dev_users = [
        UserRecord.from_row(df.iloc[i].to_dict(), posts=posts_map[str(df.iloc[i]["UserId"])])
        for i in range(2)
    ]
    result = run_user_pipeline(sample_user, agents, dev_users)
    assert "pred_SD" in result
    assert "true_SD" in result
    assert 0 <= result["pred_SD"] <= 6


def test_context_budget_truncates_oldest_posts(sample_user: UserRecord):
    long_posts = []
    for i in range(100):
        long_posts.append(
            {
                "post_id": f"p{i}",
                "text": "word " * 200,
                "date": f"2024-01-{i % 28 + 1:02d}",
                "images": [],
            }
        )
    user = UserRecord(
        user_id=sample_user.user_id,
        profile=sample_user.profile,
        posts=long_posts,
        images=[],
        labels=sample_user.labels,
    )
    budget = ContextBudget(max_input_tokens=2048, reserve_output_tokens=512)
    ctx = build_user_context(
        user,
        max_post_tokens=budget.available_input_tokens // 2,
        truncation_strategy="tail_posts",
    )
    assert "...[older posts omitted to fit context budget]" in ctx
    assert estimate_tokens(ctx) < budget.available_input_tokens


def test_context_truncation_preserves_profile(sample_user: UserRecord):
    long_posts = [{"post_id": "p1", "text": "x " * 5000, "date": "2024-01-01", "images": []}]
    user = UserRecord(
        user_id=sample_user.user_id,
        profile=sample_user.profile,
        posts=long_posts,
        images=[],
        labels=sample_user.labels,
    )
    ctx = truncate_posts_in_context(build_user_context(user), max_tokens=200)
    assert "[RAW USER PROFILE]" in ctx


def test_evaluation_metrics(tmp_path: Path):
    df = pd.DataFrame(
        {
            "true_FOMO": [3, 4, 5],
            "pred_FOMO": [3, 4, 4],
            "true_MDD": [0, 1, 1],
            "pred_MDD": [0, 1, 0],
            "true_SD": [0, 4, 5],
            "pred_SD": [0, 3, 6],
        }
    )
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    report = evaluate_results(path, binary_sd_positive_min=5)
    assert "metrics" in report
    assert "FOMO" in report["metrics"]
    assert "SD_binary" in report["metrics"]
    assert report["metrics"]["SD_binary"]["positive_min"] == 5
    assert report["metrics"]["SD_binary"]["positive_range"] == "[5, 6]"


def test_create_agents_from_config():
    config = {"llm": {"provider": "mock"}, "context": {}}
    agents = create_agents(config)
    assert set(agents.keys()) == {
        "image_caption",
        "motivation",
        "personality",
        "psychosocial",
        "clinical",
        "risk",
    }
