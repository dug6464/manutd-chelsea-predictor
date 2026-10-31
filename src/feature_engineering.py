"""
Feature Engineering Module for English Premier League Matches.
Guarantees STRICT pre-kickoff point-in-time calculation with zero data leakage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # Home team rolling form
    "home_form_pts_3",
    "home_form_pts_5",
    "home_form_pts_10",
    "home_win_rate_5",
    "home_draw_rate_5",
    "home_clean_sheets_5",
    "home_gf_5",
    "home_ga_5",
    "home_gd_5",
    "home_xg_5",
    "home_xga_5",
    "home_shots_target_5",
    "home_rest_days",
    "home_matches_14d",
    "home_venue_win_rate_5",
    "home_venue_gf_5",
    
    # Away team rolling form
    "away_form_pts_3",
    "away_form_pts_5",
    "away_form_pts_10",
    "away_win_rate_5",
    "away_draw_rate_5",
    "away_clean_sheets_5",
    "away_gf_5",
    "away_ga_5",
    "away_gd_5",
    "away_xg_5",
    "away_xga_5",
    "away_shots_target_5",
    "away_rest_days",
    "away_matches_14d",
    "away_venue_win_rate_5",
    "away_venue_ga_5",
    
    # Differentials
    "form_pts_5_diff",
    "gd_5_diff",
    "xg_5_diff",
    "rest_diff",
    "congestion_diff",
    "cumulative_ppg_diff",
    
    # Head-to-Head
    "h2h_matches_played",
    "h2h_home_pts_last3",
    "h2h_away_pts_last3",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_avg_goals_last5",
    "h2h_days_since_last",
]


class TeamMatchRecord:
    __slots__ = (
        "date",
        "venue",
        "opponent",
        "goals_for",
        "goals_against",
        "result",
        "points",
        "xg_for",
        "xg_against",
        "shots_target_for",
        "season_year",
    )

    def __init__(
        self,
        date: pd.Timestamp,
        venue: str,
        opponent: str,
        goals_for: int,
        goals_against: int,
        result: str,
        points: int,
        xg_for: float,
        xg_against: float,
        shots_target_for: int,
        season_year: int,
    ):
        self.date = date
        self.venue = venue
        self.opponent = opponent
        self.goals_for = goals_for
        self.goals_against = goals_against
        self.result = result
        self.points = points
        self.xg_for = xg_for
        self.xg_against = xg_against
        self.shots_target_for = shots_target_for
        self.season_year = season_year


class EPLFeatureExtractor:
    """
    Maintains chronological state to construct point-in-time features
    without looking ahead into the match outcome.
    """

    def __init__(self):
        # Maps team_name -> list of TeamMatchRecord
        self.team_history: Dict[str, List[TeamMatchRecord]] = defaultdict(list)
        # Maps frozenset({teamA, teamB}) -> list of H2H match dicts
        self.h2h_history: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)

    def extract_features_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Calculates all features before the match is played.
        Uses ONLY records where record.date < match_date.
        """
        home_hist = [m for m in self.team_history[home_team] if m.date < match_date]
        away_hist = [m for m in self.team_history[away_team] if m.date < match_date]

        # Calculate home rolling stats
        home_feats = self._calc_team_features(home_hist, match_date, is_home=True)
        # Calculate away rolling stats
        away_feats = self._calc_team_features(away_hist, match_date, is_home=False)

        # Calculate H2H features
        h2h_pair = frozenset([home_team, away_team])
        h2h_hist = [m for m in self.h2h_history[h2h_pair] if m["date"] < match_date]
        h2h_feats = self._calc_h2h_features(home_team, away_team, h2h_hist, match_date)

        # Comparative / differential features
        diff_feats = {
            "form_pts_5_diff": home_feats["home_form_pts_5"] - away_feats["away_form_pts_5"],
            "gd_5_diff": home_feats["home_gd_5"] - away_feats["away_gd_5"],
            "xg_5_diff": home_feats["home_xg_5"] - away_feats["away_xg_5"],
            "rest_diff": home_feats["home_rest_days"] - away_feats["away_rest_days"],
            "congestion_diff": home_feats["home_matches_14d"] - away_feats["away_matches_14d"],
            "cumulative_ppg_diff": self._calc_ppg(home_hist, match_date.year) - self._calc_ppg(away_hist, match_date.year),
        }

        # Merge all features
        features = {}
        features.update(home_feats)
        features.update(away_feats)
        features.update(diff_feats)
        features.update(h2h_feats)

        return features

    def update_history(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        fthg: int,
        ftag: int,
        ftr: str,
        hs: int,
        as_: int,
        hst: int,
        ast: int,
        xg_home: float,
        xg_away: float,
    ) -> None:
        """Appends the completed match to state after features have been recorded."""
        season_year = match_date.year if match_date.month >= 8 else match_date.year - 1

        # Home team record
        home_res = "W" if ftr == "H" else ("D" if ftr == "D" else "L")
        home_pts = 3 if home_res == "W" else (1 if home_res == "D" else 0)
        self.team_history[home_team].append(
            TeamMatchRecord(
                date=match_date,
                venue="home",
                opponent=away_team,
                goals_for=fthg,
                goals_against=ftag,
                result=home_res,
                points=home_pts,
                xg_for=xg_home,
                xg_against=xg_away,
                shots_target_for=hst,
                season_year=season_year,
            )
        )

        # Away team record
        away_res = "W" if ftr == "A" else ("D" if ftr == "D" else "L")
        away_pts = 3 if away_res == "W" else (1 if away_res == "D" else 0)
        self.team_history[away_team].append(
            TeamMatchRecord(
                date=match_date,
                venue="away",
                opponent=home_team,
                goals_for=ftag,
                goals_against=fthg,
                result=away_res,
                points=away_pts,
                xg_for=xg_away,
                xg_against=xg_home,
                shots_target_for=ast,
                season_year=season_year,
            )
        )

        # H2H record
        h2h_pair = frozenset([home_team, away_team])
        self.h2h_history[h2h_pair].append({
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "fthg": fthg,
            "ftag": ftag,
            "ftr": ftr,
            "total_goals": fthg + ftag,
        })

    def _calc_team_features(
        self,
        history: List[TeamMatchRecord],
        current_date: pd.Timestamp,
        is_home: bool,
    ) -> Dict[str, float]:
        prefix = "home_" if is_home else "away_"
        
        if not history:
            # Default prior values if team has no recorded history yet
            return {
                f"{prefix}form_pts_3": 4.0,
                f"{prefix}form_pts_5": 7.0,
                f"{prefix}form_pts_10": 14.0,
                f"{prefix}win_rate_5": 0.40,
                f"{prefix}draw_rate_5": 0.25,
                f"{prefix}clean_sheets_5": 1.0,
                f"{prefix}gf_5": 1.4,
                f"{prefix}ga_5": 1.3,
                f"{prefix}gd_5": 0.1,
                f"{prefix}xg_5": 1.35,
                f"{prefix}xga_5": 1.30,
                f"{prefix}shots_target_5": 4.5,
                f"{prefix}rest_days": 7.0,
                f"{prefix}matches_14d": 2.0,
                f"{prefix}venue_win_rate_5": 0.45 if is_home else 0.35,
                f"{prefix}venue_{'gf' if is_home else 'ga'}_5": 1.5 if is_home else 1.3,
            }

        last3 = history[-3:]
        last5 = history[-5:]
        last10 = history[-10:]

        pts_3 = sum(m.points for m in last3)
        pts_5 = sum(m.points for m in last5)
        pts_10 = sum(m.points for m in last10)

        n5 = max(1, len(last5))
        win_rate_5 = sum(1 for m in last5 if m.result == "W") / n5
        draw_rate_5 = sum(1 for m in last5 if m.result == "D") / n5
        clean_sheets_5 = sum(1 for m in last5 if m.goals_against == 0)

        gf_5 = sum(m.goals_for for m in last5) / n5
        ga_5 = sum(m.goals_against for m in last5) / n5
        gd_5 = gf_5 - ga_5

        xg_5 = sum(m.xg_for for m in last5) / n5
        xga_5 = sum(m.xg_against for m in last5) / n5
        sot_5 = sum(m.shots_target_for for m in last5) / n5

        # Rest days & Congestion
        last_match_date = history[-1].date
        rest_days = min(30.0, max(1.0, (current_date - last_match_date).total_seconds() / 86400.0))
        matches_14d = sum(
            1 for m in history
            if 0 <= (current_date - m.date).total_seconds() / 86400.0 <= 14.0
        )

        # Venue-specific form
        target_venue = "home" if is_home else "away"
        venue_matches = [m for m in history if m.venue == target_venue][-5:]
        nv = max(1, len(venue_matches))
        venue_win_rate = sum(1 for m in venue_matches if m.result == "W") / nv
        
        venue_metric_name = f"{prefix}venue_{'gf' if is_home else 'ga'}_5"
        if is_home:
            venue_metric_val = sum(m.goals_for for m in venue_matches) / nv if venue_matches else 1.5
        else:
            venue_metric_val = sum(m.goals_against for m in venue_matches) / nv if venue_matches else 1.3

        return {
            f"{prefix}form_pts_3": float(pts_3),
            f"{prefix}form_pts_5": float(pts_5),
            f"{prefix}form_pts_10": float(pts_10),
            f"{prefix}win_rate_5": float(round(win_rate_5, 3)),
            f"{prefix}draw_rate_5": float(round(draw_rate_5, 3)),
            f"{prefix}clean_sheets_5": float(clean_sheets_5),
            f"{prefix}gf_5": float(round(gf_5, 2)),
            f"{prefix}ga_5": float(round(ga_5, 2)),
            f"{prefix}gd_5": float(round(gd_5, 2)),
            f"{prefix}xg_5": float(round(xg_5, 2)),
            f"{prefix}xga_5": float(round(xga_5, 2)),
            f"{prefix}shots_target_5": float(round(sot_5, 2)),
            f"{prefix}rest_days": float(round(rest_days, 1)),
            f"{prefix}matches_14d": float(matches_14d),
            f"{prefix}venue_win_rate_5": float(round(venue_win_rate, 3)),
            venue_metric_name: float(round(venue_metric_val, 2)),
        }

    def _calc_h2h_features(
        self,
        home_team: str,
        away_team: str,
        h2h_history: List[Dict[str, Any]],
        current_date: pd.Timestamp,
    ) -> Dict[str, float]:
        if not h2h_history:
            return {
                "h2h_matches_played": 0.0,
                "h2h_home_pts_last3": 4.0,
                "h2h_away_pts_last3": 4.0,
                "h2h_home_win_rate": 0.38,
                "h2h_draw_rate": 0.28,
                "h2h_avg_goals_last5": 2.6,
                "h2h_days_since_last": 180.0,
            }

        last3 = h2h_history[-3:]
        last5 = h2h_history[-5:]

        home_pts_3 = 0
        away_pts_3 = 0
        for m in last3:
            if m["home_team"] == home_team:
                if m["ftr"] == "H":
                    home_pts_3 += 3
                elif m["ftr"] == "D":
                    home_pts_3 += 1
                    away_pts_3 += 1
                else:
                    away_pts_3 += 3
            else:
                if m["ftr"] == "H":
                    away_pts_3 += 3
                elif m["ftr"] == "D":
                    home_pts_3 += 1
                    away_pts_3 += 1
                else:
                    home_pts_3 += 3

        total_h2h = len(h2h_history)
        home_wins = sum(
            1 for m in h2h_history
            if (m["home_team"] == home_team and m["ftr"] == "H")
            or (m["away_team"] == home_team and m["ftr"] == "A")
        )
        draws = sum(1 for m in h2h_history if m["ftr"] == "D")

        avg_goals_5 = sum(m["total_goals"] for m in last5) / len(last5)
        days_since = min(730.0, max(1.0, (current_date - h2h_history[-1]["date"]).total_seconds() / 86400.0))

        return {
            "h2h_matches_played": float(total_h2h),
            "h2h_home_pts_last3": float(home_pts_3),
            "h2h_away_pts_last3": float(away_pts_3),
            "h2h_home_win_rate": float(round(home_wins / total_h2h, 3)),
            "h2h_draw_rate": float(round(draws / total_h2h, 3)),
            "h2h_avg_goals_last5": float(round(avg_goals_5, 2)),
            "h2h_days_since_last": float(round(days_since, 1)),
        }

    def _calc_ppg(self, history: List[TeamMatchRecord], current_year: int) -> float:
        curr_season_records = [m for m in history if m.season_year == current_year]
        if not curr_season_records:
            return 1.35
        return float(round(sum(m.points for m in curr_season_records) / len(curr_season_records), 3))


def build_engineered_features(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, EPLFeatureExtractor]:
    """
    Takes clean matches sorted chronologically and builds feature rows
    strictly using pre-kickoff information.
    """
    logger.info("Extracting leakage-free pre-kickoff features for %d matches...", len(clean_df))
    extractor = EPLFeatureExtractor()
    feature_rows: List[Dict[str, Any]] = []

    for idx, row in clean_df.iterrows():
        match_date = pd.to_datetime(row["match_datetime"])
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]

        # 1. Extract features BEFORE updating state
        feats = extractor.extract_features_for_match(h_team, a_team, match_date)

        # Retain metadata and ground truth
        feats["match_datetime"] = match_date
        feats["date_str"] = row["Date"]
        feats["HomeTeam"] = h_team
        feats["AwayTeam"] = a_team
        feats["FTHG"] = row["FTHG"]
        feats["FTAG"] = row["FTAG"]
        feats["FTR"] = row["FTR"]
        feats["target"] = row["target"]
        
        # Include bookmaker odds for evaluation / benchmark comparisons
        feats["B365H"] = row.get("B365H", np.nan)
        feats["B365D"] = row.get("B365D", np.nan)
        feats["B365A"] = row.get("B365A", np.nan)

        feature_rows.append(feats)

        # 2. Update state WITH completed match data
        extractor.update_history(
            home_team=h_team,
            away_team=a_team,
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

    out_df = pd.DataFrame(feature_rows)
    logger.info("Feature engineering complete: %d rows, %d feature columns.", len(out_df), len(FEATURE_COLUMNS))
    return out_df, extractor
