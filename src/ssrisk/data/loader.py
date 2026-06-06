"""Load real or synthetic user data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ssrisk.data.schema import FEATURE_COLUMNS, UserRecord


def load_features_csv(path: str | Path) -> pd.DataFrame:
    """Load user feature table from CSV or Excel."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in ["UserId", "status_posts", "grp", "SD"] if c not in df.columns]
    if missing:
        raise ValueError(f"Features file missing required columns: {missing}")
    return df


def load_posts_json(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load posts keyed by UserId."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def filter_valid_users(
    df: pd.DataFrame,
    min_posts: int = 10,
    valid_groups: list[int] | None = None,
) -> pd.DataFrame:
    """Apply study inclusion criteria."""
    valid_groups = valid_groups or [0, 1]
    mask = (df["status_posts"] > min_posts) & (df["grp"].isin(valid_groups))
    return df.loc[mask].copy()


def load_user_records(
    features_path: str | Path,
    posts_path: str | Path | None = None,
    min_posts: int = 10,
    valid_groups: list[int] | None = None,
) -> list[UserRecord]:
    """Load and merge features with posts into UserRecord objects."""
    df = filter_valid_users(
        load_features_csv(features_path),
        min_posts=min_posts,
        valid_groups=valid_groups,
    )
    posts_map = load_posts_json(posts_path) if posts_path else {}

    records: list[UserRecord] = []
    for _, row in df.iterrows():
        uid = str(row["UserId"])
        row_dict = row.to_dict()
        posts = posts_map.get(uid, [])
        images = []
        for post in posts:
            for img in post.get("images", []):
                images.append(img)
        records.append(UserRecord.from_row(row_dict, posts=posts, images=images))
    return records


def save_synthetic_dataset(
    df: pd.DataFrame,
    posts_map: dict[str, list[dict[str, Any]]],
    features_path: str | Path,
    posts_path: str | Path,
) -> None:
    """Persist generated synthetic data."""
    features_path = Path(features_path)
    posts_path = Path(posts_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    posts_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df[cols].to_csv(features_path, index=False)
    with open(posts_path, "w", encoding="utf-8") as f:
        json.dump(posts_map, f, indent=2, ensure_ascii=False)
