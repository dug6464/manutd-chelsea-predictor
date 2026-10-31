#!/usr/bin/env python3
"""
CLI Match Outcome Predictor for Manchester United vs Chelsea (and all EPL Fixtures).
Usage:
    python predictor.py
    python predictor.py --home "Man United" --away "Chelsea"
    python predictor.py --home "Chelsea" --away "Man United" --date "2026-11-15"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fix Windows console UTF-8 output encoding if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predict import MatchPredictor


def print_ascii_bar(label: str, prob: float, width: int = 35):
    """Prints a styled ASCII percentage bar."""
    filled = int(round(prob * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"  {label:<18} [{bar}] {prob * 100:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Predict Premier League Fixture Outcome (e.g. Manchester United vs Chelsea) using Machine Learning"
    )
    parser.add_argument(
        "--home",
        type=str,
        default="Man United",
        help="Home team name (default: 'Man United')",
    )
    parser.add_argument(
        "--away",
        type=str,
        default="Chelsea",
        help="Away team name (default: 'Chelsea')",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fixture date (YYYY-MM-DD). Defaults to upcoming/current date.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format.",
    )

    args = parser.parse_args()

    try:
        predictor = MatchPredictor()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please train the model first by running:\n    python run_pipeline.py\n")
        sys.exit(1)

    result = predictor.predict_fixture(
        home_team=args.home,
        away_team=args.away,
        match_date=args.date,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    home = result["home_team"]
    away = result["away_team"]
    probs = result["probabilities"]
    odds = result["fair_odds"]
    kf = result["key_factors"]
    xg = result["projected_xg"]

    print("\n" + "=" * 65)
    print(f"       PREMIER LEAGUE ML PREDICTION ENGINE")
    print(f"       Fixture: {home} (H) vs {away} (A)")
    print(f"       Target Date: {result['match_date']} | Model: {result['model_used']}")
    print("=" * 65)

    print("\n[+] OUTCOME PROBABILITIES & FAIR DECIMAL ODDS:")
    print("-" * 65)
    p_h = probs[f"{home}_win"]
    p_d = probs["Draw"]
    p_a = probs[f"{away}_win"]

    print_ascii_bar(f"{home} Win", p_h)
    print(f"    Fair Odds: {odds[f'{home}_win']:.2f} (Implied Prob: {p_h * 100:.1f}%)")
    
    print_ascii_bar("Draw", p_d)
    print(f"    Fair Odds: {odds['Draw']:.2f} (Implied Prob: {p_d * 100:.1f}%)")
    
    print_ascii_bar(f"{away} Win", p_a)
    print(f"    Fair Odds: {odds[f'{away}_win']:.2f} (Implied Prob: {p_a * 100:.1f}%)")

    print("\n[*] MOST PROBABLE OUTCOME:")
    print(f"    -> {result['predicted_outcome'].upper()}")

    print("\n[+] PROJECTED EXPECTED GOALS (xG):")
    print(f"    {home}: {xg[home]:.2f} xG  |  {away}: {xg[away]:.2f} xG")

    print("\n[+] KEY PRE-MATCH CONTEXT FACTORS:")
    print(f"    - {home} Last 5 Games Form: {kf['home_form_pts_last5']:.0f} pts")
    print(f"    - {away} Last 5 Games Form: {kf['away_form_pts_last5']:.0f} pts")
    print(f"    - Form Points Differential: {kf['form_pts_diff']:+.0f} pts")
    print(f"    - Rest Days: {home} ({kf['home_rest_days']:.0f}d) vs {away} ({kf['away_rest_days']:.0f}d)")
    print(f"    - H2H Matches Recorded: {kf['h2h_matches_on_record']:.0f}")
    print(f"    - {home} Points in Last 3 H2H: {kf['h2h_home_pts_last3']:.0f} / 9 pts")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
