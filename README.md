# Premier League Match Predictor: Manchester United vs Chelsea

An end-to-end, leak-free machine learning framework for predicting match outcomes (Home Win, Draw, Away Win) in the English Premier League, specifically tailored and benchmarked for Manchester United vs Chelsea fixtures.

---

## 📌 Project Architecture

```
manutd_chelsea_predictor/
├── data/
│   ├── raw/
│   │   ├── epl_1920.csv ... epl_2425.csv   # 6 seasons downloaded from Football-Data.co.uk
│   │   └── epl_combined_raw.csv            # 2,280 merged Premier League matches
│   └── processed/
│       └── training_set.csv                # 45 leakage-free pre-kickoff engineered features
├── models/
│   ├── best_model.joblib                   # Serialized model pipeline (StandardScaler + Calibrated Classifier)
│   ├── model.pkl                           # Pickled model artifact
│   └── feature_metadata.json               # Metadata & benchmark test metrics
├── reports/
│   ├── figures/                            # Visual confusion matrix & calibration curves
│   └── evaluation_report.md                # In-depth statistical evaluation report
├── src/
│   ├── data_loader.py                      # Multi-season data ingestion
│   ├── preprocessor.py                     # Canonicalization & target encoding (H: 0, D: 1, A: 2)
│   ├── feature_engineering.py              # Zero-leakage chronological rolling & H2H features
│   ├── train.py                            # TimeSeries chronological split & model comparison
│   ├── evaluate.py                         # Multi-class Log-Loss, Brier Score, and market comparison
│   └── predict.py                          # Fixture inference engine & fair odds calculator
├── tests/
│   ├── test_preprocessor.py                # Preprocessing and sorting tests
│   ├── test_features.py                    # Zero-leakage temporal validation tests
│   └── test_prediction.py                  # Inference verification tests
├── predictor.py                            # Interactive CLI tool for any EPL fixture
├── run_pipeline.py                         # Complete end-to-end orchestration pipeline
└── requirements.txt
```

---

## ⚡ Quickstart

### 1. Requirements & Setup
Ensure Python 3.10+ is available:
```bash
pip install -r requirements.txt
```

### 2. Predict Manchester United vs Chelsea
Run the CLI prediction tool:
```bash
# Default prediction (Man United vs Chelsea at Old Trafford)
python predictor.py

# Chelsea vs Man United at Stamford Bridge
python predictor.py --home "Chelsea" --away "Man United"

# Predict for a specific future date
python predictor.py --home "Man United" --away "Chelsea" --date "2026-11-15"

# Output structured JSON
python predictor.py --home "Man United" --away "Chelsea" --json
```

### 3. Run the End-to-End Pipeline
To re-ingest data, recompute rolling features, retrain all candidate models, and output reports:
```bash
python run_pipeline.py
```

### 4. Run Unit Tests
```bash
python tests/test_preprocessor.py
python tests/test_features.py
python tests/test_prediction.py
```

---

## 🔬 Mathematical Approach & Feature Engineering

### Strictly Leakage-Free Pre-Kickoff Engineering
A common pitfall in football outcome modeling is using match stats (shots, fouls, post-match xG) from the game being predicted. Our pipeline enforces strict temporal point-in-time calculation:
For match $i$ on date $T$, only matches played prior to date $T$ ($\tau < T$) are considered.

### Engineered Feature Sets (45 Features):
1. **Rolling Form (Last 3, 5, 10 matches)**:
   - `home_form_pts_5`, `away_form_pts_5`: Points accumulated in last 5 matches.
   - `win_rate_5`, `draw_rate_5`, `clean_sheets_5`.
   - `gf_5`, `ga_5`, `gd_5`: Goals scored, conceded, and goal difference per match.
2. **Expected Goals (xG) Proxy**:
   - Calibrated empirical formula: $\text{xG} \approx 0.28 \times \text{HST} + 0.05 \times (\text{HS} - \text{HST}) + 0.03 \times \text{HC}$.
   - Rolling offensive and defensive xG (`xg_5`, `xga_5`).
3. **Venue-Specific Form**:
   - `home_venue_win_rate_5`: Home team's win rate in their last 5 home games.
   - `away_venue_win_rate_5`: Away team's win rate in their last 5 away games.
4. **Rest & Congestion**:
   - `home_rest_days`, `away_rest_days`: Days since last official fixture.
   - `matches_14d`: Matches played in the preceding 14 days.
   - `rest_diff`: Physical fatigue advantage/disadvantage.
5. **Head-to-Head (H2H) History**:
   - Points earned in last 3 encounters between the clubs (`h2h_home_pts_last3`).
   - Lifetime H2H win and draw rates.
   - Historical goal average in recent meetings.

---

## 📊 Benchmark Model Performance

Models were trained on **1,520 matches** (2019 to 2023) and strictly tested on an untouched holdout set of **760 matches** (2023-08-01 onwards):

| Model | Log-Loss (lower is better) | Brier Score (lower is better) | Accuracy | Macro-F1 |
|---|---|---|---|---|
| **Logistic Regression (Calibrated)** | **0.9873** | 0.5878 | **54.08%** | 0.3923 |
| **Ensemble (Voting / Blend)** | **0.9873** | **0.5865** | **54.34%** | 0.4013 |
| **Random Forest** | 0.9925 | 0.5906 | 53.42% | 0.3944 |
| **HistGradientBoosting** | 1.0387 | 0.6133 | 51.84% | 0.4267 |

### Why ~54% Accuracy is the Gold Standard:
In a 3-class sporting problem (Home Win, Draw, Away Win) with high variance and draws occurring ~24-28% of the time, the theoretical baseline of random guessing is 33.3%, and naive home-bias is ~42%. Realistic industry benchmarks for leak-free models lie between **50% and 55%**. Models reporting >65% on football outcome classification almost universally suffer from future target leakage.
Our model's Brier score of **0.5878** closely benchmarks against bookmaker implied probability efficiency (~0.5559).

---

## 🎯 Sample Inference Output

```
=================================================================
       PREMIER LEAGUE ML PREDICTION ENGINE
       Fixture: Man United (H) vs Chelsea (A)
       Target Date: 2026-10-31 | Model: Logistic_Regression
=================================================================

[+] OUTCOME PROBABILITIES & FAIR DECIMAL ODDS:
-----------------------------------------------------------------
  Man United Win     [#################------------------]  47.8%
    Fair Odds: 2.09 (Implied Prob: 47.8%)
  Draw               [########---------------------------]  22.1%
    Fair Odds: 4.52 (Implied Prob: 22.1%)
  Chelsea Win        [###########------------------------]  30.1%
    Fair Odds: 3.32 (Implied Prob: 30.1%)

[*] MOST PROBABLE OUTCOME:
    -> HOME WIN

[+] PROJECTED EXPECTED GOALS (xG):
    Man United: 2.52 xG  |  Chelsea: 1.65 xG

[+] KEY PRE-MATCH CONTEXT FACTORS:
    - Man United Last 5 Games Form: 4 pts
    - Chelsea Last 5 Games Form: 12 pts
    - Form Points Differential: -8 pts
    - Rest Days: Man United (30d) vs Chelsea (30d)
    - H2H Matches Recorded: 12
    - Man United Points in Last 3 H2H: 1 / 9 pts
=================================================================
```

---

## 🌐 Live Demo & Deployment

- **Live Web Application**: [https://manutd-chelsea-predictor.streamlit.app](https://manutd-chelsea-predictor.streamlit.app)
- **Hosted on**: Streamlit Community Cloud

---

## 👨‍💻 Author & Developer

- **Developer**: **Tolosa**
- **GitHub**: [@dug6464](https://github.com/dug6464)
- **Repository**: [https://github.com/dug6464/manutd-chelsea-predictor](https://github.com/dug6464/manutd-chelsea-predictor)

