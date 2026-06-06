"""TabSTAR-style user context serialization."""

from __future__ import annotations

from typing import Any

from ssrisk.data.schema import UserRecord
from ssrisk.schemas import ImageCaptionOutput


def build_user_context(
    user: UserRecord,
    image_captions: list[ImageCaptionOutput] | None = None,
    extra_sections: list[str] | None = None,
) -> str:
    """
    Serialize user profile, chronological posts, and image descriptions
    into a single text block for LLM agents (TabSTAR-style).
    """
    p = user.profile
    gender = "Female" if p.get("female") == 1 else "Male"

    sections = [
        "[RAW USER PROFILE]",
        f"UserId: {user.user_id}",
        f"Platform Friends: {p.get('FriendCount', 'N/A')}",
        f"Total Posts (Year): {p.get('status_posts', 'N/A')}",
        f"Age: {p.get('age', 'N/A')}",
        f"Gender: {gender}",
        f"Annual Income: ${p.get('inc_num', 'N/A')}",
        "",
        "[CHRONOLOGICAL POSTS]",
    ]

    posts = sorted(user.posts, key=lambda x: x.get("date", ""))
    for i, post in enumerate(posts):
        text = post.get("text", "")
        sections.append(f'(Post {i + 1}, {post.get("date", "unknown")}): "{text}"')
        for img in post.get("images", []):
            img_id = img.get("image_id", img.get("id", i))
            sections.append(f"  [Image {img_id}] attached to this post")

    caption_map = {c.image_id: c for c in (image_captions or [])}
    if caption_map or any(post.get("images") for post in posts):
        sections.extend(["", "[IMAGE DESCRIPTIONS]"])
        seen: set[str] = set()
        for post in posts:
            for img in post.get("images", []):
                img_id = str(img.get("image_id", img.get("id", "")))
                if img_id in seen:
                    continue
                seen.add(img_id)
                if img_id in caption_map:
                    cap = caption_map[img_id]
                    sections.append(
                        f'[Image {img_id}]: "{cap.caption}" (tone: {cap.emotional_tone})'
                    )
                else:
                    hint = img.get("hint", "visual content")
                    sections.append(f"[Image {img_id}]: (pending caption; hint: {hint})")

    if extra_sections:
        sections.extend(extra_sections)

    return "\n".join(sections)


def append_agent_output(context: str, agent_name: str, payload: Any) -> str:
    """Append structured agent output to accumulated context."""
    if hasattr(payload, "model_dump_json"):
        body = payload.model_dump_json(indent=2)
    else:
        body = str(payload)
    return f"{context}\n\n[{agent_name}]\n{body}"
