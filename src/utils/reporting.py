"""
Reporting and Analysis Utilities

This module provides tools for analyzing simulation results, generating reports,
and visualizing data from insulin pump reinforcement learning experiments.

Key functionality includes:
- Analyzing saved patient trajectories
- Generating performance reports and metrics
- Creating visualizations for glucose control performance

The module serves as a central location for all reporting and analysis tools
used throughout the project, facilitating standardized evaluation of different
control algorithms and patient outcomes.
"""

import logging
import pandas as pd
from simglucose.analysis.report import report


def sg_analyze(files_path=None, save_path=None):
    """
    Analyze saved trajectories and generate a report.

    This function processes CSV files containing patient glucose data,
    combines them, and generates a standardized report in PDF format.
    The report includes key metrics and visualizations.

    Args:
        files_path (str): Path to the directory containing saved trajectory files.
        save_path (str, optional): Path to save the generated report. If None,
            a timestamped report will be created in the reporting directory.

    Returns:
        Path: The path to the generated report file.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing saved trajectories...")
    from pathlib import Path

    path = Path(__file__).parent.parent.parent  # Adjust path to the project root
    # Ensure files_path exists and contains CSV files
    if files_path is None:
        print("No files_path provided, using simglucose example to generate report.")
        result_path = path / "results/simglucose_reports/example_data"
    else:
        result_path = (
            Path(files_path) if not isinstance(files_path, Path) else files_path
        )

    result_filenames = list(result_path.glob("*.csv"))

    if not result_filenames:
        raise FileNotFoundError(f"No CSV files found in {result_path}")

    patient_names = [f.stem for f in result_filenames]
    df = pd.concat(
        [pd.read_csv(str(f), index_col=0) for f in result_filenames],
        keys=patient_names,
    )

    # Handle save_path directory creation
    if save_path is None:
        datetimestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"results/simglucose_reports/report_{datetimestamp}")
        save_path.mkdir(parents=False, exist_ok=False)
    else:
        save_path = Path(save_path)
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the report
    report(df, save_path=str(save_path))
    logger.info(f"Report saved to {save_path}")

    return save_path
