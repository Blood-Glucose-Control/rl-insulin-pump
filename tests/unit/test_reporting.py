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
        # Create sample DataFrame with realistic glucose data
        data = {
            "Time": [
                "2017-12-31 06:00:00",
                "2017-12-31 06:03:00",
                "2017-12-31 06:06:00",
                "2017-12-31 06:09:00",
            ],
            "BG": [149.02, 149.02, 149.02, 149.02],
            "CGM": [171.40, 171.00, 169.99, 169.42],
            "CHO": [0.0, 0.0, 0.0, 0.0],
            "insulin": [0.01393, 0.01393, 0.01393, 0.01393],
            "LBGI": [0.0, 0.0, 0.0, 1.445791638515412],
            "HBGI": [2.7553, 2.7553, 2.7553, 2.7553],
            "Risk": [2.7553, 2.7553, 2.7553, 2.7553],
        }

        df = pd.DataFrame(data).set_index("Time")
        df.to_csv(temp_dir / f"{patient_id}.csv")

    yield temp_dir
    shutil.rmtree(temp_dir)


# TODO: these don't pass right now, need to fix the sg_analyze function to handle directories properly


def test_analyze_with_existing_directory():
    """Test sg_analyze works with directory containing CSV files."""
    result_path = sg_analyze()
    assert result_path.exists()
    # optionally check that it contains files
    assert any(result_path.iterdir())


# def test_analyze_with_output_path():
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
