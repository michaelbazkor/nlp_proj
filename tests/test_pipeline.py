"""Tests for SSRisk pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ssrisk.data.schema import UserRecord
from ssrisk.data.serialize import append_agent_output, build_user_context
from ssrisk.data.synthetic import generate_synthetic_users
from ssrisk.evaluation import evaluate_results
from ssrisk.llm.mock import MockLLMClient
from ssrisk.pipeline import run_user_pipeline, split_dev_test
from ssrisk.schemas import ClinicalOutput, MotivationOutput, PersonalityOutput, RiskOutput


@pytest.fixture
def sample_user() -> UserRecord:
    df, posts_map = generate_synthetic_users(n_users=5, random_seed=0)
    row = df.iloc[0].to_dict()
    uid = str(row["UserId"])
    return UserRecord.from_row(row, posts=posts_map[uid])


def test_schemas_validate():
    MotivationOutput(
        primary_motivation="entertainment",
        pred_FOMO=20,
        motivation_analysis="Test analysis.",
    )
    PersonalityOutput(
        pred_BFI_N=5,
        pred_BFI_E=6,
        pred_BFI_O=5,
        pred_BFI_A=6,
        pred_BFI_C=5,
        pred_Lonely=20,
        pred_Brooding=10,
        pred_Worry=30,
        pred_SWL=25,
        personality_analysis="Test.",
    )
    ClinicalOutput(
        pred_PHQ9_1=1,
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
    RiskOutput(pred_SD=0, risk_analysis="Low risk.")


def test_serializer_includes_sections(sample_user: UserRecord):
    ctx = build_user_context(sample_user)
    assert "[RAW USER PROFILE]" in ctx
    assert "[CHRONOLOGICAL POSTS]" in ctx
    assert sample_user.user_id in ctx


def test_mock_llm_returns_valid_schema(sample_user: UserRecord):
    client = MockLLMClient()
    ctx = build_user_context(sample_user)
    result = client.complete("system", ctx, MotivationOutput)
    assert 10 <= result.pred_FOMO <= 40


def test_split_dev_test():
    df, posts_map = generate_synthetic_users(n_users=20, random_seed=1)
    users = [
        UserRecord.from_row(df.iloc[i].to_dict(), posts=posts_map[str(df.iloc[i]["UserId"])])
        for i in range(len(df))
    ]
    dev, test = split_dev_test(users, dev_split=0.1, seed=42)
    assert len(dev) >= 1
    assert len(test) == len(users) - len(dev)


def test_end_to_end_mock_pipeline(sample_user: UserRecord):
    from ssrisk.agents.clinical import ClinicalAgent
    from ssrisk.agents.image_caption import ImageCaptionAgent
    from ssrisk.agents.motivation import MotivationAgent
    from ssrisk.agents.personality import PersonalityAgent
    from ssrisk.agents.risk import RiskAgent

    client = MockLLMClient()
    agents = {
        "image_caption": ImageCaptionAgent(client),
        "motivation": MotivationAgent(client),
        "personality": PersonalityAgent(client),
        "clinical": ClinicalAgent(client),
        "risk": RiskAgent(client),
    }
    df, posts_map = generate_synthetic_users(n_users=5, random_seed=2)
    dev_users = [
        UserRecord.from_row(df.iloc[i].to_dict(), posts=posts_map[str(df.iloc[i]["UserId"])])
        for i in range(2)
    ]
    result = run_user_pipeline(sample_user, agents, dev_users)
    assert "pred_SD" in result
    assert "true_SD" in result
    assert 0 <= result["pred_SD"] <= 6


def test_evaluation_metrics(tmp_path: Path):
    df = pd.DataFrame(
        {
            "true_FOMO": [20, 30, 40],
            "pred_FOMO": [22, 28, 38],
            "true_MDD": [0, 1, 1],
            "pred_MDD": [0, 1, 0],
            "true_SD": [0, 2, 4],
            "pred_SD": [0, 2, 3],
        }
    )
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    report = evaluate_results(path)
    assert "metrics" in report
    assert "FOMO" in report["metrics"]
    assert "SD_binary" in report["metrics"]
