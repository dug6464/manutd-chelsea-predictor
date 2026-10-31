"""
Unit tests for data preprocessing and cleaning using Python unittest.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import (
    canonicalize_team_name,
    clean_match_data,
    TARGET_MAP,
)


class TestPreprocessor(unittest.TestCase):
    def test_team_canonicalization(self):
        self.assertEqual(canonicalize_team_name("Man United"), "Man United")
        self.assertEqual(canonicalize_team_name("Manchester United"), "Man United")
        self.assertEqual(canonicalize_team_name("man utd"), "Man United")
        self.assertEqual(canonicalize_team_name("Chelsea"), "Chelsea")
        self.assertEqual(canonicalize_team_name("chelsea fc"), "Chelsea")
        self.assertEqual(canonicalize_team_name("Spurs"), "Tottenham")
        self.assertEqual(canonicalize_team_name("Arsenal FC"), "Arsenal")

    def test_clean_match_data_sorting_and_targets(self):
        sample_data = pd.DataFrame([
            {
                "Date": "15/01/2022",
                "Time": "15:00",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Man City",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "HS": 12,
                "AS": 8,
                "HST": 5,
                "AST": 2,
                "HC": 6,
                "AC": 3,
            },
            {
                "Date": "10/01/2022",
                "Time": "20:00",
                "HomeTeam": "Man United",
                "AwayTeam": "Aston Villa",
                "FTHG": 2,
                "FTAG": 2,
                "FTR": "D",
                "HS": 14,
                "AS": 11,
                "HST": 6,
                "AST": 4,
                "HC": 7,
                "AC": 4,
            },
        ])

        clean = clean_match_data(sample_data)
        self.assertEqual(len(clean), 2)
        # Ensure chronological order
        self.assertEqual(clean.iloc[0]["HomeTeam"], "Man United")
        self.assertEqual(clean.iloc[1]["HomeTeam"], "Chelsea")
        # Ensure target mapping
        self.assertEqual(clean.iloc[0]["target"], TARGET_MAP["D"])
        self.assertEqual(clean.iloc[1]["target"], TARGET_MAP["H"])
        # Ensure xG proxy calculated
        self.assertIn("xg_home_proxy", clean.columns)
        self.assertGreater(clean.iloc[0]["xg_home_proxy"], 0)


if __name__ == "__main__":
    unittest.main()
