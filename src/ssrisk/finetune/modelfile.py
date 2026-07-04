"""Ollama Modelfile generation for per-agent specialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ssrisk.finetune.dataset import AGENT_EXPORT_ORDER, AGENT_SYSTEM_PROMPTS


def generate_modelfile(agent_name: str, base_model: str) -> str:
    """Build Modelfile content for one agent."""
    system = AGENT_SYSTEM_PROMPTS[agent_name].replace('"', '\\"')
    return (
        f'FROM {base_model}\n\n'
        f'PARAMETER temperature 0.2\n\n'
        f'SYSTEM """{system}"""\n'
    )


def write_modelfiles(config: dict[str, Any]) -> dict[str, Path]:
    """Write Modelfile per agent and print ollama create commands."""
    finetune_cfg = config.get("finetune", {})
    base_model = finetune_cfg.get("base_model") or config.get("llm", {}).get("model", "llama3.1")
    output_dir = Path(finetune_cfg.get("modelfile_dir", "outputs/modelfiles"))
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    print("\nRun these commands after fine-tuning (or to bake in system prompts):\n")
    for agent_name in AGENT_EXPORT_ORDER:
        content = generate_modelfile(agent_name, base_model)
        path = output_dir / f"{agent_name}.Modelfile"
        path.write_text(content, encoding="utf-8")
        paths[agent_name] = path
        model_name = f"ssrisk-{agent_name}"
        print(f"  ollama create {model_name} -f {path}")
    print(
        "\nThen set agents.<name>.model in config.yaml to the created model names "
        "(e.g. ssrisk-motivation).\n"
    )
    return paths
