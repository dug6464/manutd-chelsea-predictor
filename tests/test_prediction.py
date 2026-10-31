"""
Unit tests for MatchPredictor inference engine using Python unittest.
"""

import sys
from pathlib import Path
import unittest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import MatchPredictor


class TestPredictionEngine(unittest.TestCase):
    def test_prediction_output_integrity(self):
        model_file = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
        if not model_file.exists():
            self.skipTest("best_model.joblib does not exist yet. Run pipeline first.")

        predictor = MatchPredictor(model_file)
        result = predictor.predict_fixture("Man United", "Chelsea")

        self.assertIn("probabilities", result)
        probs = result["probabilities"]
        self.assertIn("Man United_win", probs)
        self.assertIn("Draw", probs)
        self.assertIn("Chelsea_win", probs)

        # Check sum of probabilities ≈ 1.0
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Check fair odds
        odds = result["fair_odds"]
        self.assertGreater(odds["Man United_win"], 1.0)
        self.assertGreater(odds["Draw"], 1.0)
        self.assertGreater(odds["Chelsea_win"], 1.0)

        # Check key factors
        self.assertIn("key_factors", result)
        self.assertIn(result["predicted_outcome"], ["Home Win", "Draw", "Away Win"])


if __name__ == "__main__":
    unittest.main()
