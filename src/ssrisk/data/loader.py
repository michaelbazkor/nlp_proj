"""Load real or synthetic user data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ssrisk.data.schema import FEATURE_COLUMNS, UserRecord

POST_TEXT_COLUMNS = ("PostMessage", "PostDescription", "PostStory")


def _post_row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a Facebook post export row to the internal posts.json shape."""
    text_parts = [
        str(row[col]).strip()
        for col in POST_TEXT_COLUMNS
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip()
    ]
    post_id = str(row.get("PostId", "") or row.get("id", ""))
    post: dict[str, Any] = {
        "post_id": post_id,
        "text": " ".join(text_parts),
        "date": str(row.get("PostCreationTime", "")),
        "images": [],
    }
    image_url = row.get("PostAttachmentImageUrl")
    if pd.notna(image_url) and str(image_url).strip():
        hint = row.get("PostAttachmentTitle")
        post["images"].append(
            {
                "image_id": post_id,
                "url": str(image_url).strip(),
                "hint": str(hint).strip() if pd.notna(hint) and str(hint).strip() else None,
            }
        )
    return post


def _first_present(row: pd.Series, *columns: str) -> Any:
    """Return the first non-null value from candidate column names."""
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


def _user_row_from_posts_group(first: pd.Series, user_id: str, post_count: int) -> dict[str, Any]:
    """Build a user feature row from the first post row, including psychometric labels."""
    row_dict: dict[str, Any] = {col: None for col in FEATURE_COLUMNS}
    row_dict["UserId"] = str(user_id)
    row_dict["status_posts"] = _first_present(first, "status_posts") or post_count
    row_dict["FriendCount"] = _first_present(first, "FriendCount", "FriendCount_x", "FriendCount_y")

    for col in FEATURE_COLUMNS:
        if col in {"UserId", "status_posts", "FriendCount"}:
            continue
        if col in first.index and pd.notna(first[col]):
            row_dict[col] = first[col]

    return row_dict


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


def load_posts_csv(path: str | Path) -> pd.DataFrame:
    """Load a post-level Facebook export CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Posts CSV not found: {path}")
    df = pd.read_csv(path)
    if "UserId" not in df.columns:
        raise ValueError("Posts CSV must include a UserId column.")
    return df


def posts_csv_to_user_records(
    df: pd.DataFrame,
    min_posts: int = 10,
    max_posts_per_user: int | None = None,
    valid_groups: list[int] | None = None,
) -> list[UserRecord]:
    """
    Build UserRecord objects from a post-level export (one row per post).

    When psychometric label columns are present on each row (merged features),
    they are taken from the first row per user.
    """
    valid_groups = valid_groups if valid_groups is not None else [0, 1]
    records: list[UserRecord] = []
    for user_id, group in df.groupby("UserId", sort=False):
        posts_df = group.copy()
        first = posts_df.iloc[0]
        grp = _first_present(first, "grp")
        if grp is not None and int(grp) not in valid_groups:
            continue

        if "PostCreationTime" in posts_df.columns:
            posts_df = posts_df.sort_values("PostCreationTime")
        if max_posts_per_user:
            posts_df = posts_df.tail(int(max_posts_per_user))

        posts = [_post_row_to_dict(row) for _, row in posts_df.iterrows()]
        posts = [post for post in posts if post["text"] or post["images"]]
        if len(posts) < min_posts:
            continue

        row_dict = _user_row_from_posts_group(first, str(user_id), len(posts))
        images = [img for post in posts for img in post.get("images", [])]
        records.append(UserRecord.from_row(row_dict, posts=posts, images=images))
    return records


def load_user_records_from_posts_csv(
    path: str | Path,
    min_posts: int = 10,
    max_posts_per_user: int | None = None,
    valid_groups: list[int] | None = None,
) -> list[UserRecord]:
    """Load users directly from a post-level Facebook export CSV."""
    return posts_csv_to_user_records(
        load_posts_csv(path),
        min_posts=min_posts,
        max_posts_per_user=max_posts_per_user,
        valid_groups=valid_groups,
    )


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
    posts_csv_path: str | Path | None = None,
    min_posts: int = 10,
    valid_groups: list[int] | None = None,
    max_posts_per_user: int | None = None,
) -> list[UserRecord]:
    """Load and merge features with posts into UserRecord objects."""
    if posts_csv_path:
        return load_user_records_from_posts_csv(
            posts_csv_path,
            min_posts=min_posts,
            max_posts_per_user=max_posts_per_user,
            valid_groups=valid_groups,
        )

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
