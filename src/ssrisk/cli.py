"""Command-line interface for SSRisk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ssrisk.config import load_config
from ssrisk.data.synthetic import generate_and_save
from ssrisk.evaluation import save_evaluation_report
from ssrisk.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSRisk: Multi-agent suicide risk detection pipeline"
    )
    parser.add_argument(
        "command",
        choices=["generate-data", "run", "evaluate", "all"],
        help="Pipeline command",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
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
            binary_sd_threshold=eval_cfg.get("binary_sd_threshold", 1),
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
            binary_sd_threshold=config.get("evaluation", {}).get("binary_sd_threshold", 1),
        )


if __name__ == "__main__":
    main()
