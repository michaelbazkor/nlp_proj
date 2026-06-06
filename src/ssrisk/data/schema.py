"""Data schema definitions derived from the Ophir et al. study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Core feature columns used by the pipeline and evaluation
FEATURE_COLUMNS = [
    "UserId",
    "status_posts",
    "posts",
    "FriendCount",
    "age",
    "female",
    "inc_num",
    "grp",
    "FOMO",
    "BFI_N",
    "BFI_E",
    "BFI_O",
    "BFI_A",
    "BFI_C",
    "Lonely",
    "Brooding",
    "Worry",
    "SWL",
    "PHQ9",
    "PHQ9_1",
    "PHQ9_2",
    "PHQ9_3",
    "PHQ9_4",
    "PHQ9_5",
    "PHQ9_6",
    "PHQ9_7",
    "PHQ9_8",
    "PHQ9_9",
    "MDD",
    "GAD",
    "SD",
    "suicide",
]

PROFILE_COLUMNS = [
    "FriendCount",
    "status_posts",
    "age",
    "female",
    "inc_num",
]

GROUND_TRUTH_MAP = {
    "FOMO": "pred_FOMO",
    "BFI_N": "pred_BFI_N",
    "BFI_E": "pred_BFI_E",
    "BFI_O": "pred_BFI_O",
    "BFI_A": "pred_BFI_A",
    "BFI_C": "pred_BFI_C",
    "Lonely": "pred_Lonely",
    "Brooding": "pred_Brooding",
    "Worry": "pred_Worry",
    "SWL": "pred_SWL",
    "PHQ9_1": "pred_PHQ9_1",
    "PHQ9_2": "pred_PHQ9_2",
    "PHQ9_3": "pred_PHQ9_3",
    "PHQ9_4": "pred_PHQ9_4",
    "PHQ9_5": "pred_PHQ9_5",
    "PHQ9_6": "pred_PHQ9_6",
    "PHQ9_7": "pred_PHQ9_7",
    "PHQ9_8": "pred_PHQ9_8",
    "PHQ9_9": "pred_PHQ9_9",
    "MDD": "pred_MDD",
    "SD": "pred_SD",
}


@dataclass
class UserRecord:
    """Single user with profile, labels, posts, and image references."""

    user_id: str
    profile: dict[str, Any]
    labels: dict[str, Any]
    posts: list[dict[str, Any]]
    images: list[dict[str, Any]]

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        posts: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> "UserRecord":
        profile = {k: row.get(k) for k in PROFILE_COLUMNS}
        labels = {k: row.get(k) for k in GROUND_TRUTH_MAP}
        labels["PHQ9"] = row.get("PHQ9")
        labels["GAD"] = row.get("GAD")
        labels["suicide"] = row.get("suicide")
        return cls(
            user_id=str(row["UserId"]),
            profile=profile,
            labels=labels,
            posts=posts or [],
            images=images or [],
        )
