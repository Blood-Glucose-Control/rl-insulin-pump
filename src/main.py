#!/usr/bin/env python3

import numpy as np
import torch
from scripts.experiments.network_architecture import (
    grid_search,
    plot_results,
    visualize_results,
)
from src.training.experiment_runner import ExperimentRunner
from src.utils.cmd_args import parse_args
import logging

from src.utils.reporting import sg_analyze

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_device(cfg):
    """Select device based on availability and configuration."""
    # Auto-detect if not provided in the config
    if torch.cuda.is_available():
        device = "cuda"
    elif "device" in cfg and cfg["device"]:
        match cfg["device"]:
            case "cuda":
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    logger.warning("CUDA not available, falling back to CPU.")
                    device = "cpu"
            case "mps":
                if torch.backends.mps.is_available():
                    device = "mps"
                else:
                    logger.warning("MPS not available, falling back to CPU.")
                    device = "cpu"
            case "cpu":
                device = "cpu"
            case _:
                logger.error(f"Unknown device '{cfg['device']}'. Falling back to CPU.")
                device = "cpu"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    # Update the config so that downstream functions use the correct device
    cfg["device"] = device
    return device


def main():
    # Parse configuration from YAML
    cfg = parse_args()
    # Set a fixed seed for reproducibility
    np.random.seed(cfg["seed"])

    # Select device and update the configuration
    select_device(cfg)

    # Decide on the mode based on configuration
    mode = cfg.get("mode", "train").lower()

    runner = ExperimentRunner(cfg)

    if mode == "train":
        print(f"Training with configuration: {cfg}")
        runner.train()
    elif mode == "predict":
        runner.predict()
    elif mode == "grid_search":
        # Run grid search over network configurations
        results = grid_search()
        # Visualize and save results
        visualize_results(results)
        # Also print results to console
        plot_results(results)
    elif mode == "analyze":
        sg_analyze(cfg.get("files_path", None), cfg.get("save_path", None))
    else:
        logger.error(
            f"Unknown mode '{mode}'. Please choose 'train', 'predict', 'analyze', or 'grid_search'."
        )


if __name__ == "__main__":
    main()
