#!/usr/bin/env python3
"""
End-to-End Orchestration Pipeline for Manchester United vs Chelsea Match Predictor.
Executes: Data Ingestion -> Preprocessing -> Leakage-Free Feature Engineering ->
          Model Training & Cross-Validation -> Holdout Evaluation -> Diagnostic Artifacts.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_epl_dataset
from src.evaluate import generate_evaluation_artifacts
from src.feature_engineering import build_engineered_features
from src.predict import MatchPredictor
from src.preprocessor import clean_match_data
from src.train import train_and_compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


def run_pipeline():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("  STARTING END-TO-END EPL MATCH PREDICTION PIPELINE")
    logger.info("=================================================================")

    # Step 1: Data Ingestion
    logger.info("\n--- [Step 1/5] Ingesting EPL Historical Data ---")
    raw_df = load_epl_dataset(raw_dir=PROJECT_ROOT / "data" / "raw")
    logger.info("Loaded raw dataset: %d matches, %d raw columns.", len(raw_df), len(raw_df.columns))

    # Step 2: Cleaning & Preprocessing
    logger.info("\n--- [Step 2/5] Cleaning and Standardizing Records ---")
    clean_df = clean_match_data(raw_df)
    logger.info("Clean dataset: %d validated matches.", len(clean_df))

    # Step 3: Feature Engineering (Strictly Pre-Kickoff)
    logger.info("\n--- [Step 3/5] Extracting Point-in-Time Rolling & H2H Features ---")
    features_df, _ = build_engineered_features(clean_df)
    
    processed_csv_path = PROJECT_ROOT / "data" / "processed" / "training_set.csv"
    features_df.to_csv(processed_csv_path, index=False)
    logger.info("Saved processed dataset with features to %s (%d rows)", processed_csv_path, len(features_df))

    # Step 4: Model Training, Cross-Validation & Comparison
    logger.info("\n--- [Step 4/5] Training Classifiers (LR, RF, HistGB, XGBoost, Ensemble) ---")
    training_results = train_and_compare_models(
        features_df=features_df,
        models_dir=PROJECT_ROOT / "models",
        split_date="2023-08-01",
    )
    best_model_name = training_results["pipeline_bundle"]["best_model_name"]
    logger.info("Training complete. Winner: %s", best_model_name)

    # Step 5: Evaluation, Calibration & Artifact Generation
    logger.info("\n--- [Step 5/5] Generating Diagnostic Visualizations and Evaluation Report ---")
    report_path = generate_evaluation_artifacts(
        y_test=training_results["y_test"],
        best_probs=training_results["test_probs"]["Ensemble"] if best_model_name == "Ensemble_Model" else training_results["test_probs"][best_model_name.replace(" ", "_")],
        model_name=best_model_name,
        pipeline_bundle=training_results["pipeline_bundle"],
        reports_dir=PROJECT_ROOT / "reports",
        test_df=training_results["test_df"],
    )
    logger.info("Evaluation report successfully written to %s", report_path)

    # Demonstration: Run predictions for target fixtures
    logger.info("\n=================================================================")
    logger.info("  RUNNING SAMPLE PREDICTIONS ON TARGET FIXTURES")
    logger.info("=================================================================")
    predictor = MatchPredictor(PROJECT_ROOT / "models" / "best_model.joblib")

    # Fixture 1: Manchester United vs Chelsea at Old Trafford
    res1 = predictor.predict_fixture("Man United", "Chelsea")
    logger.info("\n[Fixture 1] %s at Old Trafford:", res1["fixture"])
    for k, v in res1["probabilities"].items():
        logger.info("  - %s: %.1f%% (Fair Odds: %.2f)", k, v * 100, res1["fair_odds"][k])
    logger.info("  Most likely outcome: %s", res1["predicted_outcome"])

    # Fixture 2: Chelsea vs Manchester United at Stamford Bridge
    res2 = predictor.predict_fixture("Chelsea", "Man United")
    logger.info("\n[Fixture 2] %s at Stamford Bridge:", res2["fixture"])
    for k, v in res2["probabilities"].items():
        logger.info("  - %s: %.1f%% (Fair Odds: %.2f)", k, v * 100, res2["fair_odds"][k])
    logger.info("  Most likely outcome: %s", res2["predicted_outcome"])

    elapsed = round(time.time() - start_time, 2)
    logger.info("\n=================================================================")
    logger.info("  PIPELINE EXECUTION COMPLETED IN %.2f SECONDS", elapsed)
    logger.info("=================================================================")


if __name__ == "__main__":
    run_pipeline()
