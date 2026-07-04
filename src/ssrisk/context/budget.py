"""Token budget estimation and prompt truncation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (chars / 4) without external tokenizer."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_budget(text: str, max_tokens: int, strategy: str = "tail") -> str:
    """Truncate text to fit within max_tokens."""
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    if strategy == "head":
        return text[:max_chars] + "\n...[truncated]"
    return "...[truncated]\n" + text[-max_chars:]


@dataclass
class ContextBudget:
    max_input_tokens: int = 8192
    reserve_output_tokens: int = 1024
    truncation_strategy: str = "tail_posts"
    max_few_shot_tokens: int = 1500
    max_agent_output_tokens: int = 800

    @property
    def available_input_tokens(self) -> int:
        return max(256, self.max_input_tokens - self.reserve_output_tokens)


def load_context_budget(config: dict[str, Any] | None) -> ContextBudget:
    """Build ContextBudget from config dict."""
    cfg = (config or {}).get("context", {})
    return ContextBudget(
        max_input_tokens=int(cfg.get("max_input_tokens", 8192)),
        reserve_output_tokens=int(cfg.get("reserve_output_tokens", 1024)),
        truncation_strategy=str(cfg.get("truncation_strategy", "tail_posts")),
        max_few_shot_tokens=int(cfg.get("max_few_shot_tokens", 1500)),
        max_agent_output_tokens=int(cfg.get("max_agent_output_tokens", 800)),
    )


def fit_prompt(
    system: str,
    few_shot: str,
    user: str,
    budget: ContextBudget | None = None,
) -> tuple[str, str, str]:
    """
    Allocate token budget across system, few-shot, and user context.
    Returns (system, few_shot, user) possibly truncated.
    """
    budget = budget or ContextBudget()
    available = budget.available_input_tokens

    system_tokens = estimate_tokens(system)
    few_shot_budget = min(budget.max_few_shot_tokens, available // 4)
    if few_shot:
        few_shot = truncate_to_budget(few_shot, few_shot_budget, strategy="tail")

    few_shot_tokens = estimate_tokens(few_shot)
    user_budget = available - system_tokens - few_shot_tokens
    if user_budget < 256:
        user_budget = max(256, available - system_tokens)
        few_shot = truncate_to_budget(few_shot, max(0, available - system_tokens - user_budget))

    user = truncate_to_budget(user, user_budget, strategy="tail")

    total = estimate_tokens(system) + estimate_tokens(few_shot) + estimate_tokens(user)
    if total > available:
        logger.warning(
            "Prompt still exceeds budget after truncation: %d tokens (limit %d)",
            total,
            available,
        )
    return system, few_shot, user


def truncate_agent_output_payload(payload: Any, max_tokens: int) -> str:
    """Serialize agent output, truncating long analysis fields if needed."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        return str(payload)

    analysis_keys = (
        "motivation_analysis",
        "personality_analysis",
        "psychosocial_analysis",
        "clinical_analysis",
        "risk_analysis",
    )
    max_chars = max_tokens * CHARS_PER_TOKEN
    for key in analysis_keys:
        val = data.get(key)
        if isinstance(val, str) and len(val) > max_chars:
            data[key] = val[:max_chars] + "...[truncated]"

    return json.dumps(data, indent=2)


_POST_LINE_RE = re.compile(r'^\(Post \d+,')


def truncate_posts_in_context(context: str, max_tokens: int, strategy: str = "tail_posts") -> str:
    """
    Truncate chronological posts section, dropping oldest posts first.
    Preserves profile and non-post sections.
    """
    if strategy != "tail_posts":
        return truncate_to_budget(context, max_tokens, strategy="tail")

    marker = "[CHRONOLOGICAL POSTS]"
    if marker not in context:
        return truncate_to_budget(context, max_tokens, strategy="tail")

    before, rest = context.split(marker, 1)
    lines = rest.split("\n")
    post_lines: list[str] = []
    after_lines: list[str] = []
    in_posts = True
    for line in lines:
        if in_posts and line.startswith("[") and not _POST_LINE_RE.match(line) and line.strip():
            in_posts = False
        if in_posts:
            post_lines.append(line)
        else:
            after_lines.append(line)

    header_budget = estimate_tokens(before + marker)
    after_text = "\n".join(after_lines)
    after_budget = estimate_tokens(after_text)
    post_budget = max_tokens - header_budget - after_budget
    if post_budget <= 0:
        return truncate_to_budget(context, max_tokens, strategy="tail")

    kept: list[str] = []
    used = 0
    for line in reversed(post_lines):
        line_tokens = estimate_tokens(line + "\n")
        if used + line_tokens > post_budget and kept:
            break
        kept.insert(0, line)
        used += line_tokens

    if len(kept) < len([l for l in post_lines if l.strip()]):
        kept.insert(0, "...[older posts omitted to fit context budget]")

    posts_section = "\n".join(kept)
    result = f"{before}{marker}\n{posts_section}"
    if after_lines:
        result += "\n" + after_text
    return result
