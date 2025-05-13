import numpy as np
import torch
from experiments.network_architecture import (
    grid_search,
    plot_results,
    visualize_results,
)
from src.training.experiment_runner import ExperimentRunner
from src.utils.cmd_args import parse_args
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_device(cfg):
    """Select device based on availability and configuration."""
    # Auto-detect if not provided in the config
    if "device" in cfg and cfg["device"]:
        device = cfg["device"]
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
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
        runner.train(cfg)
    elif mode == "predict":
        runner.predict(cfg)
    elif mode == "grid_search":
        # Run grid search over network configurations
        results = grid_search(cfg)
        # Visualize and save results
        visualize_results(results)
        # Also print results to console
        plot_results(results)
    elif mode == "analyze":
        runner.analyze(cfg)
    else:
        logger.error(
            f"Unknown mode '{mode}'. Please choose 'train', 'predict', 'report', or 'grid_search'."
        )


if __name__ == "__main__":
    main()
