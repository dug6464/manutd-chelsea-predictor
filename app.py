"""
Streamlit Web Application for Premier League Match Prediction.
Interactive dashboard to predict Manchester United vs Chelsea and all EPL fixtures.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import MatchPredictor
from src.preprocessor import TEAM_CANONICAL_MAP

# Page config
st.set_page_config(
    page_title="EPL Match Predictor | Man United vs Chelsea",
    page_icon="⚽",
    layout="wide",
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
    }
    .prob-number {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚽ Premier League AI Match Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict Manchester United vs Chelsea (and all 20 EPL Clubs) with Zero-Leakage Machine Learning</div>", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        from run_pipeline import run_pipeline
        run_pipeline()
    return MatchPredictor(model_path)


try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Unique list of Premier League teams
all_teams = sorted(list(set(TEAM_CANONICAL_MAP.values())))

# Layout: Match Selector Sidebar & Columns
st.sidebar.header("⚙️ Fixture Settings")

home_default_idx = all_teams.index("Man United") if "Man United" in all_teams else 0
away_default_idx = all_teams.index("Chelsea") if "Chelsea" in all_teams else 1

home_team = st.sidebar.selectbox("🏠 Home Team", all_teams, index=home_default_idx)
away_team = st.sidebar.selectbox("✈️ Away Team", all_teams, index=away_default_idx)

match_date = st.sidebar.date_input("📅 Match Date", value=pd.Timestamp.now())

if home_team == away_team:
    st.warning("⚠️ Home team and Away team cannot be the same. Please choose different teams.")
    st.stop()

# Predict button
predict_button = st.sidebar.button("🔮 Predict Match Outcome", use_container_width=True, type="primary")

# Run prediction
with st.spinner("Analyzing point-in-time team form, xG, rest days, and head-to-head records..."):
    result = predictor.predict_fixture(
        home_team=home_team,
        away_team=away_team,
        match_date=str(match_date)
    )

home = result["home_team"]
away = result["away_team"]
probs = result["probabilities"]
odds = result["fair_odds"]
xg = result["projected_xg"]
kf = result["key_factors"]

p_h = probs[f"{home}_win"]
p_d = probs["Draw"]
p_a = probs[f"{away}_win"]

# Main Display: 3 Outcome Cards
st.subheader(f"🏟️ {home} vs {away}")
st.caption(f"Predicted using **{result['model_used']}** trained on 6 EPL seasons (Holdout Test Accuracy: ~54.1%)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"### 🏠 {home} Win")
    st.markdown(f"<div class='prob-number' style='color:#1f77b4;'>{p_h * 100:.1f}%</div>", unsafe_allow_html=True)
    st.metric(label="Fair Decimal Odds", value=f"{odds[f'{home}_win']:.2f}")
    st.progress(p_h)

with col2:
    st.markdown("### 🤝 Draw")
    st.markdown(f"<div class='prob-number' style='color:#ff7f0e;'>{p_d * 100:.1f}%</div>", unsafe_allow_html=True)
    st.metric(label="Fair Decimal Odds", value=f"{odds['Draw']:.2f}")
    st.progress(p_d)

with col3:
    st.markdown(f"### ✈️ {away} Win")
    st.markdown(f"<div class='prob-number' style='color:#2ca02c;'>{p_a * 100:.1f}%</div>", unsafe_allow_html=True)
    st.metric(label="Fair Decimal Odds", value=f"{odds[f'{away}_win']:.2f}")
    st.progress(p_a)

st.markdown("---")

# Section 2: Detailed Breakdown & Analysis
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 Projected Expected Goals (xG)")
    xg_df = pd.DataFrame({
        "Team": [home, away],
        "Projected xG": [xg[home], xg[away]]
    })
    st.bar_chart(xg_df.set_index("Team"))
    st.info(f"**Projected Scoreline Anchor**: {home} ({xg[home]:.1f}) - {away} ({xg[away]:.1f})")

with col_right:
    st.markdown("### 🔍 Pre-Match Tactical Context")
    st.write(f"- **{home} Form (Last 5)**: `{kf['home_form_pts_last5']:.0f} pts`")
    st.write(f"- **{away} Form (Last 5)**: `{kf['away_form_pts_last5']:.0f} pts`")
    st.write(f"- **Form Differential**: `{kf['form_pts_diff']:+.0f} pts`")
    st.write(f"- **Rest Schedule**: {home} (`{kf['home_rest_days']:.0f} days`) vs {away} (`{kf['away_rest_days']:.0f} days`)")
    st.write(f"- **Head-to-Head Encounters Analyzed**: `{kf['h2h_matches_on_record']:.0f} fixtures`")
    st.write(f"- **{home} Points in Last 3 H2H**: `{kf['h2h_home_pts_last3']:.0f} / 9 pts`")

st.markdown("---")
st.caption("Built with zero data leakage: Point-in-time historical data from Football-Data.co.uk.")
