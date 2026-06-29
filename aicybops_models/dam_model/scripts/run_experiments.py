#!/usr/bin/env python3
"""
Run experiment sweeps: multiple (config, seed) combinations, one MLflow run per variant.

Usage:
    # Two configs, three seeds
    python run_experiments.py --configs configs/dam_config.json other/config.json --seeds 0 1 2

    # With evaluation and summary CSV
    python run_experiments.py --configs configs/dam_config.json --seeds 0 1 --run-evaluation --output-dir results

    # From sweep file (JSON)
    python run_experiments.py --sweep sweep.json

    # Quick test (limited samples)
    python run_experiments.py --configs configs/dam_config.json --seeds 0 --quick-test --epochs 2
"""

import argparse
import json
import sys
from pathlib import Path

# dam_model root for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "processing"))

from pipelines.experiments import ExperimentRunner, SweepSpec


def load_sweep_file(path: str) -> dict:
    """Load sweep spec from JSON or YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")
    with open(p) as f:
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML sweep files. Install with: pip install pyyaml")
        return json.load(f)


def parse_overrides(s: str) -> dict:
    """Parse --overrides JSON string into a dict."""
    if not s:
        return {}
    return json.loads(s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiment sweep: configs x seeds, one MLflow run per (config, seed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Sweep definition: either CLI or file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--configs",
        nargs="+",
        help="Paths to dam_config.json files",
    )
    group.add_argument(
        "--sweep",
        type=str,
        metavar="PATH",
        help="Path to sweep spec file (JSON or YAML) with config_paths, seeds, etc.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds (default: 42). Ignored if --sweep is used.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="DAM_experiments",
        help="MLflow experiment name (default: DAM_experiments)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        metavar="JSON",
        help='JSON object of overrides (e.g. {"epochs": 2, "batch_size": 16})',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (shortcut; overrides key in --overrides)",
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run evaluation after each training run and log metrics",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to evaluation config JSON (default: minimal baseline config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for experiment_summary.csv",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first run failure",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Limit samples (max_train_samples, max_test_samples) for faster runs",
    )

    args = parser.parse_args()

    if args.sweep:
        spec_dict = load_sweep_file(args.sweep)
        config_paths = spec_dict.get("config_paths")
        seeds = spec_dict.get("seeds", [42])
        if not config_paths:
            print("Sweep file must contain 'config_paths' (list of paths).", file=sys.stderr)
            return 1
        # CLI overrides sweep file
        if args.seeds != [42]:
            seeds = args.seeds
        experiment_name = args.experiment_name or spec_dict.get("experiment_name", "DAM_experiments")
        tracking_uri = args.tracking_uri if args.tracking_uri is not None else spec_dict.get("tracking_uri")
        overrides = spec_dict.get("overrides") or {}
        if args.overrides:
            overrides = {**overrides, **parse_overrides(args.overrides)}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        run_evaluation = args.run_evaluation or spec_dict.get("run_evaluation", False)
        eval_config = spec_dict.get("eval_config")
        if args.eval_config:
            with open(args.eval_config) as f:
                eval_config = json.load(f)
        output_dir = args.output_dir or spec_dict.get("output_dir")
        fail_fast = args.fail_fast or spec_dict.get("fail_fast", False)
        quick_test = args.quick_test or spec_dict.get("quick_test", False)
    else:
        config_paths = args.configs
        seeds = args.seeds
        experiment_name = args.experiment_name
        tracking_uri = args.tracking_uri
        overrides = parse_overrides(args.overrides) if args.overrides else {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        run_evaluation = args.run_evaluation
        eval_config = None
        if args.eval_config:
            with open(args.eval_config) as f:
                eval_config = json.load(f)
        output_dir = args.output_dir
        fail_fast = args.fail_fast
        quick_test = args.quick_test

    spec = SweepSpec(
        config_paths=config_paths,
        seeds=seeds,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        overrides=overrides or None,
        run_evaluation=run_evaluation,
        eval_config=eval_config,
        output_dir=output_dir,
        fail_fast=fail_fast,
        quick_test=quick_test,
    )

    runner = ExperimentRunner(spec)
    result = runner.run()

    if result.get("failed_runs"):
        print(f"\n{len(result['failed_runs'])} run(s) failed:", file=sys.stderr)
        for r in result["failed_runs"]:
            print(f"  {r['run_name']}: {r['error']}", file=sys.stderr)
    if result.get("summary_csv_path"):
        print(f"\nSummary: {result['summary_csv_path']}")

    return 1 if result.get("failed_runs") else 0


if __name__ == "__main__":
    sys.exit(main())
