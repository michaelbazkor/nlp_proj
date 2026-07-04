"""Command-line interface for SSRisk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ssrisk.config import load_config
from ssrisk.data.synthetic import generate_and_save
from ssrisk.evaluation import save_evaluation_report
from ssrisk.finetune.dataset import export_all_datasets
from ssrisk.finetune.modelfile import write_modelfiles
from ssrisk.pipeline import run_pipeline


def _cmd_finetune_export(config: dict) -> None:
    export_all_datasets(config)


def _cmd_finetune_modelfiles(config: dict) -> None:
    write_modelfiles(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSRisk: Multi-agent suicide risk detection pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ("generate-data", "run", "evaluate", "all"):
        sub = subparsers.add_parser(cmd, help=f"Pipeline: {cmd}")
        sub.add_argument("--config", default="config.yaml", help="Path to config YAML")

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tuning utilities")
    finetune_sub = finetune_parser.add_subparsers(dest="finetune_command", required=True)
    for subcmd in ("export", "modelfiles"):
        ft = finetune_sub.add_parser(subcmd, help=f"Fine-tune: {subcmd}")
        ft.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = parser.parse_args()

    if args.command == "finetune":
        config = load_config(args.config)
        if args.finetune_command == "export":
            _cmd_finetune_export(config)
        elif args.finetune_command == "modelfiles":
            _cmd_finetune_modelfiles(config)
        return

    config = load_config(args.config)

    if args.command == "generate-data":
        data_cfg = config.get("data", {})
        generate_and_save(
            features_path=data_cfg.get("features_path", "data/synthetic_users.csv"),
            posts_path=data_cfg.get("posts_path", "data/posts.json"),
            n_users=data_cfg.get("n_users", 50),
            random_seed=data_cfg.get("random_seed", 42),
        )
        print("Synthetic data generated.")
        return

    if args.command == "run":
        run_pipeline(config)
        return

    if args.command == "evaluate":
        pipe_cfg = config.get("pipeline", {})
        eval_cfg = config.get("evaluation", {})
        results_path = Path(pipe_cfg.get("output_dir", "outputs")) / "pipeline_results.csv"
        save_evaluation_report(
            results_path,
            pipe_cfg.get("output_dir", "outputs"),
            binary_sd_positive_min=eval_cfg.get("binary_sd_positive_min", 5),
        )
        return

    if args.command == "all":
        data_cfg = config.get("data", {})
        generate_and_save(
            features_path=data_cfg.get("features_path", "data/synthetic_users.csv"),
            posts_path=data_cfg.get("posts_path", "data/posts.json"),
            n_users=data_cfg.get("n_users", 50),
            random_seed=data_cfg.get("random_seed", 42),
        )
        results_path = run_pipeline(config)
        save_evaluation_report(
            results_path,
            config.get("pipeline", {}).get("output_dir", "outputs"),
            binary_sd_positive_min=config.get("evaluation", {}).get("binary_sd_positive_min", 5),
        )


if __name__ == "__main__":
    main()
