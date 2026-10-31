"""
Preprocessor Module for EPL Match Data.
Handles team name canonicalization, datetime parsing, sorting, target encoding, and validation.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEAM_CANONICAL_MAP: Dict[str, str] = {
    "man united": "Man United",
    "manchester united": "Man United",
    "man utd": "Man United",
    "manchester utd": "Man United",
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "man city": "Man City",
    "manchester city": "Man City",
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "aston villa": "Aston Villa",
    "villa": "Aston Villa",
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "wolves": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "crystal palace": "Crystal Palace",
    "palace": "Crystal Palace",
    "brentford": "Brentford",
    "fulham": "Fulham",
    "everton": "Everton",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "nott'm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "southampton": "Southampton",
    "burnley": "Burnley",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "luton": "Luton",
    "luton town": "Luton",
    "ipswich": "Ipswich",
    "ipswich town": "Ipswich",
    "watford": "Watford",
    "norwich": "Norwich",
    "norwich city": "Norwich",
}

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def canonicalize_team_name(name: str) -> str:
    """Normalize and canonicalize team names."""
    if not isinstance(name, str):
        return str(name)
    cleaned = name.strip().lower()
    return TEAM_CANONICAL_MAP.get(cleaned, name.strip())


def parse_match_date(date_series: pd.Series) -> pd.Series:
    """Robust multi-format date parser."""
    return pd.to_datetime(date_series, format="mixed", dayfirst=True, errors="coerce")


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw EPL match DataFrame:
    - Filters rows with missing targets or essential stats.
    - Standardizes team names.
    - Standardizes and sorts by datetime.
    - Encodes target variable (0: Home Win, 1: Draw, 2: Away Win).
    - Computes xG proxy and clean numeric stats.
    """
    df = df.copy()
    logger.info("Starting data cleaning with %d rows...", len(df))

    # Basic required columns
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Standardize team names
    df["HomeTeam"] = df["HomeTeam"].apply(canonicalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(canonicalize_team_name)

    # Discard non-valid matches (e.g., team playing itself or blank names)
    df = df[df["HomeTeam"] != df["AwayTeam"]]

    # Standardize result
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df = df[df["FTR"].isin(["H", "D", "A"])]

    # Parse Date & Time
    parsed_dates = parse_match_date(df["Date"])
    df["parsed_date"] = parsed_dates
    df = df.dropna(subset=["parsed_date"])

    # Time parsing
    if "Time" not in df.columns or df["Time"].isnull().all():
        df["Time"] = "15:00"
    else:
        df["Time"] = df["Time"].fillna("15:00")

    # Construct match_datetime
    try:
        df["match_datetime"] = pd.to_datetime(
            df["parsed_date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
            errors="coerce"
        )
    except Exception:
        df["match_datetime"] = df["parsed_date"]

    df["match_datetime"] = df["match_datetime"].fillna(df["parsed_date"])

    # Sort strictly chronologically
    df = df.sort_values(by=["match_datetime", "HomeTeam"]).reset_index(drop=True)

    # Coerce numeric columns
    numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Encode target
    df["target"] = df["FTR"].map(TARGET_MAP)

    # Expected goals (xG) proxy model:
    # Industry-standard empirical formula when shot coordinates are not present:
    # Shot on Target (0.28) + Off-Target Shot (0.05) + Corner (0.03)
    df["xg_home_proxy"] = (
        df["HST"] * 0.28 +
        (df["HS"] - df["HST"]).clip(lower=0) * 0.05 +
        df["HC"] * 0.03
    ).round(2)
    df["xg_away_proxy"] = (
        df["AST"] * 0.28 +
        (df["AS"] - df["AST"]).clip(lower=0) * 0.05 +
        df["AC"] * 0.03
    ).round(2)

    logger.info("Cleaning finished. %d clean matches available.", len(df))
    return df
