#!/usr/bin/env python3

import numpy as np
from scripts.experiments.network_architecture import (
    grid_search,
    plot_results,
    visualize_results,
)
from src.training.experiment_runner import ExperimentRunner
from src.utils.cmd_args import parse_args
from src.utils.config import Config
import logging

from src.utils.reporting import sg_analyze

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Parse configuration from YAML
    cfg = parse_args()
    config = Config(cfg)
    # Set a fixed seed for reproducibility
    np.random.seed(config.seed)

    # Decide on the mode based on configuration
    modes = config.modes

    runner = ExperimentRunner(cfg, config)

    for mode in modes:
        mode = mode.lower()
        logger.info(f"""\n
                    =====================================\n
                    Running mode: {mode}\n
                    =====================================\n""")
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
            sg_analyze(
                config.predict_results_path,
                config.analysis_results_path,
            )
        else:
            logger.error(
                f"Unknown mode '{mode}'. Please choose 'train', 'predict', 'analyze', or 'grid_search'."
            )


if __name__ == "__main__":
    main()
