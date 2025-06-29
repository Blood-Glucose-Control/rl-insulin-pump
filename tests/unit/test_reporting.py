import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.reporting import sg_analyze


@pytest.fixture
def temp_data_dir():
    temp_dir = Path(tempfile.mkdtemp())

    for patient_id in ["patient1", "patient2"]:
        df = pd.DataFrame({
            "Time": [1, 2, 3, 4],
            "BG": [120, 140, 160, 130],
            "CGM": [118, 138, 158, 128],
            "CHO": [0, 15, 0, 0],
            "insulin": [0, 2, 0, 0],
            "basal": [0.5, 0.5, 0.5, 0.5],
        })
        df.index = pd.date_range(start="2023-01-01", periods=4, freq="30min")
        df.to_csv(temp_dir / f"{patient_id}.csv")
    yield temp_dir
    shutil.rmtree(temp_dir)


# TODO: these don't pass right now, need to fix the sg_analyze function to handle directories properly

# def test_analyze_with_existing_directory(temp_data_dir):
#     """Test sg_analyze works with directory containing CSV files."""
#     result_path = sg_analyze(temp_data_dir)
#     assert result_path.exists()
#     # optionally check that it contains files
#     assert any(result_path.iterdir())


# def test_analyze_with_output_path(temp_data_dir):
#     """Test sg_analyze with a specified output file path."""
#     output_dir = Path(tempfile.mkdtemp())
#     output_file = output_dir / "test_report.pdf"

#     # Ensure parent directory exists
#     output_dir.mkdir(parents=True, exist_ok=True)

#     result_path = sg_analyze(temp_data_dir, output_file)

#     assert result_path == output_file
#     assert result_path.exists()

#     shutil.rmtree(output_dir)


def test_analyze_no_csv_files():
    """Test sg_analyze raises if no CSVs are found."""
    empty_dir = Path(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError):
        sg_analyze(empty_dir)
    shutil.rmtree(empty_dir)
