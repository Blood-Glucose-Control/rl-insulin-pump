import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.utils.reporting import sg_analyze

class TestSG_AnalyzeFunction(unittest.TestCase):
    def setUp(self):
        """Create temporary test data directory with mock CSVs."""
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a few mock CSV files
        for patient_id in ["patient1", "patient2"]:
            # Create sample data with required columns for report function
            df = pd.DataFrame({
                "Time": [1, 2, 3, 4],
                "BG": [120, 140, 160, 130],
                "CGM": [118, 138, 158, 128],
                "CHO": [0, 15, 0, 0],
                "insulin": [0, 2, 0, 0]
            })
            df.index = pd.date_range(start="2023-01-01", periods=4, freq="30min")
            df.to_csv(self.temp_dir / f"{patient_id}.csv")
            
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_with_existing_directory(self):
        """Test analyze function with a directory containing CSV files."""
        result_path = sg_analyze(self.temp_dir)
        self.assertTrue(result_path.exists())
        
    def test_analyze_with_output_path(self):
        """Test analyze function with specified output path."""
        output_dir = Path(tempfile.mkdtemp())
        output_file = output_dir / "test_report.pdf"
        result_path = sg_analyze(self.temp_dir, output_file)
        self.assertEqual(result_path, output_file)
        self.assertTrue(result_path.exists())
        shutil.rmtree(output_dir)
    
    def test_analyze_no_csv_files(self):
        """Test analyze function with directory containing no CSV files."""
        empty_dir = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            sg_analyze(empty_dir)
        shutil.rmtree(empty_dir)


if __name__ == '__main__':
    unittest.main()