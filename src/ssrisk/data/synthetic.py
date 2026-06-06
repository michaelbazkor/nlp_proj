"""Synthetic data generator with correlated psychometric labels."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from ssrisk.data.loader import save_synthetic_dataset
from ssrisk.data.schema import FEATURE_COLUMNS


POSITIVE_POSTS = [
    "Had an amazing time with friends today!",
    "Feeling blessed and grateful for my family.",
    "Church was wonderful this morning. So much peace.",
    "Great workout! Ready to tackle the week.",
    "Celebrating my sister's wedding this weekend!",
    "Thanksgiving dinner was perfect. Love my people.",
    "New job opportunity came through. Excited!",
    "Beautiful sunset walk with my dog.",
]

NEUTRAL_POSTS = [
    "Grocery shopping done for the week.",
    "Watching the game tonight.",
    "Traffic was terrible this morning.",
    "Need coffee. Monday vibes.",
    "Updated my profile picture.",
    "Rain all day. Staying in.",
]

DISTRESS_POSTS = [
    "I feel so alone today. Nobody ever invites me anywhere.",
    "Can't sleep again. My mind won't stop racing.",
    "Everything feels pointless lately.",
    "Back pain is killing me and I'm exhausted.",
    "Why does everyone else have it together except me?",
    "Had another fight with my partner. I can't do this.",
    "Crying in the bathroom at work again.",
    "I wish someone would just check on me.",
    "Some days I wonder if anyone would notice if I disappeared.",
    "Feeling worthless and stuck.",
]

IMAGE_SCENARIOS = [
    {"id": "1", "latent": "positive", "desc_hint": "group of friends smiling outdoors"},
    {"id": "2", "latent": "neutral", "desc_hint": "everyday object or scenery"},
    {"id": "3", "latent": "distress", "desc_hint": "dim room, solitary figure"},
]


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def generate_synthetic_users(
    n_users: int = 50,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Generate correlated users, labels, posts, and image refs."""
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)

    rows: list[dict[str, Any]] = []
    posts_map: dict[str, list[dict[str, Any]]] = {}

    for i in range(n_users):
        uid = f"U{i + 1:04d}"
        distress = float(rng.beta(2, 5) if rng.random() < 0.36 else rng.beta(1.5, 6))

        bfi_n = _clip_int(1 + distress * 8 + rng.normal(0, 1), 1, 10)
        bfi_e = _clip_int(8 - distress * 5 + rng.normal(0, 1), 1, 10)
        lonely = _clip_int(12 + distress * 22 + rng.normal(0, 3), 10, 40)
        brooding = _clip_int(6 + distress * 10 + rng.normal(0, 2), 5, 20)
        worry = _clip_int(20 + distress * 45 + rng.normal(0, 5), 16, 80)
        swl = _clip_int(28 - distress * 18 + rng.normal(0, 3), 5, 35)
        fomo = _clip_int(15 + distress * 18 + rng.normal(0, 3), 10, 40)
        gad = _clip_int(distress * 15 + rng.normal(0, 2), 0, 21)

        phq_items = [
            _clip_int(distress * 2.5 + rng.normal(0, 0.5), 0, 3) for _ in range(9)
        ]
        phq9 = sum(phq_items)
        mdd = 1 if phq9 >= 10 else 0

        if distress > 0.65 and phq_items[8] >= 2:
            sd = _clip_int(3 + distress * 3, 0, 6)
        elif distress > 0.45:
            sd = _clip_int(1 + distress * 2, 0, 6)
        else:
            sd = 0

        n_posts = int(rng.integers(12, 35))
        status_posts = n_posts

        user_posts: list[dict[str, Any]] = []
        for p in range(n_posts):
            if distress > 0.5 and rng.random() < 0.55:
                text = random.choice(DISTRESS_POSTS)
            elif distress < 0.25 and rng.random() < 0.5:
                text = random.choice(POSITIVE_POSTS)
            else:
                text = random.choice(NEUTRAL_POSTS + DISTRESS_POSTS[:3])

            images = []
            if p % 4 == 0:
                scenario = random.choice(IMAGE_SCENARIOS)
                images.append(
                    {
                        "image_id": f"{uid}_img_{p}",
                        "latent_mood": scenario["latent"],
                        "hint": scenario["desc_hint"],
                    }
                )

            user_posts.append(
                {
                    "post_id": f"{uid}_p{p}",
                    "text": text,
                    "date": f"2024-{(p % 12) + 1:02d}-{(p % 28) + 1:02d}",
                    "images": images,
                }
            )

        posts_map[uid] = user_posts

        row = {
            "UserId": uid,
            "status_posts": status_posts,
            "posts": n_posts,
            "FriendCount": int(rng.integers(50, 800)),
            "age": int(rng.integers(22, 58)),
            "female": int(rng.random() < 0.77),
            "inc_num": int(rng.integers(25000, 95000)),
            "grp": int(rng.choice([0, 1])),
            "FOMO": fomo,
            "BFI_N": bfi_n,
            "BFI_E": bfi_e,
            "BFI_O": _clip_int(rng.normal(6, 1.5), 1, 10),
            "BFI_A": _clip_int(rng.normal(6.5, 1.5), 1, 10),
            "BFI_C": _clip_int(rng.normal(6, 1.5), 1, 10),
            "Lonely": lonely,
            "Brooding": brooding,
            "Worry": worry,
            "SWL": swl,
            "PHQ9": phq9,
            "PHQ9_1": phq_items[0],
            "PHQ9_2": phq_items[1],
            "PHQ9_3": phq_items[2],
            "PHQ9_4": phq_items[3],
            "PHQ9_5": phq_items[4],
            "PHQ9_6": phq_items[5],
            "PHQ9_7": phq_items[6],
            "PHQ9_8": phq_items[7],
            "PHQ9_9": phq_items[8],
            "MDD": mdd,
            "GAD": gad,
            "SD": sd,
            "suicide": 1 if sd >= 1 else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns and col != "posts":
            df[col] = None
    return df, posts_map


def generate_and_save(
    features_path: str,
    posts_path: str,
    n_users: int = 50,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Generate synthetic dataset and persist to disk."""
    df, posts_map = generate_synthetic_users(n_users=n_users, random_seed=random_seed)
    save_synthetic_dataset(df, posts_map, features_path, posts_path)
    return df, posts_map
