"""
Unit tests for leakage-free feature extraction using Python unittest.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import EPLFeatureExtractor, build_engineered_features
from src.preprocessor import clean_match_data


class TestFeatureEngineering(unittest.TestCase):
    def test_zero_leakage_and_rolling_calculations(self):
        raw_matches = pd.DataFrame([
            {
                "Date": "01/09/2023",
                "Time": "15:00",
                "HomeTeam": "Man United",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "HS": 10,
                "AS": 10,
                "HST": 5,
                "AST": 4,
                "HC": 5,
                "AC": 5,
            },
            {
                "Date": "08/09/2023",
                "Time": "15:00",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
                "HS": 8,
                "AS": 8,
                "HST": 4,
                "AST": 3,
                "HC": 4,
                "AC": 4,
            },
            {
                "Date": "15/09/2023",
                "Time": "15:00",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Man United",
                "FTHG": 0,
                "FTAG": 3,
                "FTR": "A",
                "HS": 6,
                "AS": 12,
                "HST": 2,
                "AST": 7,
                "HC": 3,
                "AC": 6,
            },
        ])

        clean = clean_match_data(raw_matches)
        features_df, extractor = build_engineered_features(clean)

        # First match: neither team has prior games, so default initial form
        row0 = features_df.iloc[0]
        self.assertEqual(row0["h2h_matches_played"], 0.0)

        # Third match: Man United has played 1 prior game (win vs Chelsea = 3 pts)
        # Arsenal has played 1 prior game (draw vs Chelsea = 1 pt)
        row2 = features_df.iloc[2]
        self.assertEqual(row2["HomeTeam"], "Arsenal")
        self.assertEqual(row2["AwayTeam"], "Man United")
        # Away team (Man United) should have 3 form points from their match on 01/09
        self.assertEqual(row2["away_form_pts_5"], 3.0)
        # Home team (Arsenal) should have 1 form point from their match on 08/09
        self.assertEqual(row2["home_form_pts_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
