"""
Prediction Engine Module for Premier League Match Ups.
Takes any two teams (e.g., Manchester United vs Chelsea), computes the live point-in-time
features from latest historical match state, and predicts calibrated outcome probabilities and fair odds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd

from .data_loader import load_epl_dataset
from .feature_engineering import EPLFeatureExtractor, FEATURE_COLUMNS
from .preprocessor import canonicalize_team_name, clean_match_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Production-ready predictor for fixture outcomes between any Premier League clubs.
    """

    def __init__(self, model_bundle_path: Optional[Path] = None):
        if model_bundle_path is None:
            model_bundle_path = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
        
        if not model_bundle_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_bundle_path}. Please run train.py or run_pipeline.py first."
            )

        self.bundle = joblib.load(model_bundle_path)
        self.scaler = self.bundle["scaler"]
        self.best_name = self.bundle["best_model_name"]
        self.feature_columns = self.bundle["feature_columns"]
        self.extractor: Optional[EPLFeatureExtractor] = None
        self._initialize_extractor()

    def _initialize_extractor(self):
        """Loads dataset and replays history up to the latest known match."""
        raw_df = load_epl_dataset()
        clean_df = clean_match_data(raw_df)

        self.extractor = EPLFeatureExtractor()
        for _, row in clean_df.iterrows():
            match_date = pd.to_datetime(row["match_datetime"])
            self.extractor.update_history(
                home_team=row["HomeTeam"],
                away_team=row["AwayTeam"],
                match_date=match_date,
                fthg=int(row["FTHG"]),
                ftag=int(row["FTAG"]),
                ftr=str(row["FTR"]),
                hs=int(row.get("HS", 0)),
                as_=int(row.get("AS", 0)),
                hst=int(row.get("HST", 0)),
                ast=int(row.get("AST", 0)),
                xg_home=float(row.get("xg_home_proxy", 1.2)),
                xg_away=float(row.get("xg_away_proxy", 1.0)),
            )

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates calibrated win/draw/loss probabilities for a fixture.
        """
        home_team = canonicalize_team_name(home_team)
        away_team = canonicalize_team_name(away_team)

        if match_date is None:
            match_date_ts = pd.Timestamp.now()
        else:
            match_date_ts = pd.to_datetime(match_date)

        # 1. Extract feature row before kickoff
        feats = self.extractor.extract_features_for_match(home_team, away_team, match_date_ts)
        feat_df = pd.DataFrame([feats])[self.feature_columns]

        # 2. Score with models
        X_raw = feat_df.values
        X_scaled = self.scaler.transform(X_raw)

        # Model predictions
        lr_probs = self.bundle["lr_model"].predict_proba(X_scaled)[0]
        rf_probs = self.bundle["rf_model"].predict_proba(X_raw)[0]
        gb_probs = self.bundle["gb_model"].predict_proba(X_raw)[0]

        probs_to_blend = [lr_probs, rf_probs, gb_probs]
        if self.bundle.get("xgb_model") is not None:
            try:
                xgb_probs = self.bundle["xgb_model"].predict_proba(X_raw)[0]
                probs_to_blend.append(xgb_probs)
            except Exception:
                pass

        # Select best model or ensemble
        if self.best_name == "Logistic_Regression":
            selected_probs = lr_probs
        elif self.best_name == "Random_Forest":
            selected_probs = rf_probs
        elif self.best_name == "Gradient_Boosting":
            selected_probs = gb_probs
        elif self.best_name == "XGBoost" and self.bundle.get("xgb_model") is not None:
            selected_probs = xgb_probs
        else:
            selected_probs = np.mean(probs_to_blend, axis=0)

        p_home = float(selected_probs[0])
        p_draw = float(selected_probs[1])
        p_away = float(selected_probs[2])

        # Normalize to ensure sum is strictly 1.0
        total_p = p_home + p_draw + p_away
        p_home /= total_p
        p_draw /= total_p
        p_away /= total_p

        # Fair decimal odds (no bookmaker margin)
        fair_odds_home = round(1.0 / p_home, 2)
        fair_odds_draw = round(1.0 / p_draw, 2)
        fair_odds_away = round(1.0 / p_away, 2)

        # Estimated goal expectation
        home_xg_proj = round(feats.get("home_xg_5", 1.4) * 1.1, 2)
        away_xg_proj = round(feats.get("away_xg_5", 1.2) * 0.95, 2)

        predicted_class_idx = int(np.argmax([p_home, p_draw, p_away]))
        outcomes = ["Home Win", "Draw", "Away Win"]
        predicted_outcome = outcomes[predicted_class_idx]

        return {
            "fixture": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date_ts.strftime("%Y-%m-%d"),
            "model_used": self.best_name,
            "probabilities": {
                f"{home_team}_win": round(p_home, 4),
                "Draw": round(p_draw, 4),
                f"{away_team}_win": round(p_away, 4),
            },
            "fair_odds": {
                f"{home_team}_win": fair_odds_home,
                "Draw": fair_odds_draw,
                f"{away_team}_win": fair_odds_away,
            },
            "predicted_outcome": predicted_outcome,
            "projected_xg": {
                home_team: home_xg_proj,
                away_team: away_xg_proj,
            },
            "key_factors": {
                "home_form_pts_last5": feats.get("home_form_pts_5"),
                "away_form_pts_last5": feats.get("away_form_pts_5"),
                "form_pts_diff": feats.get("form_pts_5_diff"),
                "h2h_matches_on_record": feats.get("h2h_matches_played"),
                "h2h_home_pts_last3": feats.get("h2h_home_pts_last3"),
                "home_rest_days": feats.get("home_rest_days"),
                "away_rest_days": feats.get("away_rest_days"),
            },
        }
