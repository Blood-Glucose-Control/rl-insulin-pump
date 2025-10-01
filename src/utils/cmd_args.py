import argparse
import yaml
from pathlib import Path


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description="RL Insulin Pump")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        required=True,
        help="The configuration file path.",
    )

    # Optionally merge command-line overrides into the config
    parser.add_argument("--seed", type=int, help="Random seed to override config.")
    # ----------------------------------------------------------------

    args = parser.parse_args()
    cfg_path = Path(args.cfg_file)
    if not cfg_path.is_file():
        parser.error(f"Configuration file {cfg_path} does not exist or is not a file.")
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        parser.error(f"Error parsing configuration file: {exc}")

    # Optionally merge command-line overrides into the config
    if args.seed is not None:
        cfg["seed"] = args.seed
    # ----------------------------------------------------------------

    return cfg
