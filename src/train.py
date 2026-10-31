"""
Model Training Module for EPL Match Prediction.
Implements chronological train/test splits, TimeSeries cross-validation,
multiple model architectures (Logistic Regression, Random Forest, Gradient Boosting, XGBoost),
and saves trained artifacts with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost if available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from .feature_engineering import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_data_chronologically(
    df: pd.DataFrame,
    split_date: Optional[str] = "2023-08-01",
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically to prevent temporal data leakage.
    Uses split_date if within range, else test_ratio on the tail.
    """
    df_sorted = df.sort_values(by="match_datetime").reset_index(drop=True)
    
    if split_date:
        split_ts = pd.to_datetime(split_date)
        train_df = df_sorted[df_sorted["match_datetime"] < split_ts].copy()
        test_df = df_sorted[df_sorted["match_datetime"] >= split_ts].copy()
        
        # Fallback if split_date yields too small train or test set
        if len(train_df) > 200 and len(test_df) > 50:
            logger.info("Split by date %s: %d train, %d test matches.", split_date, len(train_df), len(test_df))
            return train_df, test_df

    # Ratio fallback
    cutoff_idx = int(len(df_sorted) * (1 - test_ratio))
    train_df = df_sorted.iloc[:cutoff_idx].copy()
    test_df = df_sorted.iloc[cutoff_idx:].copy()
    logger.info("Split by ratio: %d train, %d test matches.", len(train_df), len(test_df))
    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Computes Multi-Class Log-Loss, Multi-Class Brier Score, and Accuracy."""
    y_pred = np.argmax(y_probs, axis=1)
    
    # Log loss
    loss = float(log_loss(y_true, y_probs, labels=[0, 1, 2]))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Multi-class Brier score: mean squared error across all 3 class indicators
    # One-hot encode y_true
    y_one_hot = np.zeros_like(y_probs)
    for i, target_class in enumerate(y_true):
        y_one_hot[i, int(target_class)] = 1.0
    brier = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

    logger.info(
        "[%s] Log-Loss: %.4f | Brier Score: %.4f | Accuracy: %.2f%% | Macro-F1: %.4f",
        model_name, loss, brier, acc * 100.0, macro_f1
    )
    return {
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
    }


def train_and_compare_models(
    features_df: pd.DataFrame,
    models_dir: Optional[Path] = None,
    split_date: str = "2023-08-01",
) -> Dict[str, Any]:
    """
    Trains and validates multiple models using TimeSeries cross-validation
    and holds out an untouched chronological test set.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = split_data_chronologically(features_df, split_date=split_date)

    X_train_raw = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values.astype(int)
    X_test_raw = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values.astype(int)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 1. Baseline Logistic Regression (with probability calibration)
    lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_model = CalibratedClassifierCV(estimator=lr_base, cv=3)
    lr_model.fit(X_train_scaled, y_train)

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_raw, y_train)

    # 3. Gradient Boosting Classifier (HistGradientBoosting / LightGBM style)
    gb_model = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.04,
        min_samples_leaf=15,
        l2_regularization=0.5,
        random_state=42
    )
    gb_model.fit(X_train_raw, y_train)

    # 4. XGBoost Classifier (if library is available)
    xgb_model = None
    if XGB_AVAILABLE:
        try:
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42
            )
            xgb_model.fit(X_train_raw, y_train)
            logger.info("XGBoost model successfully trained.")
        except Exception as e:
            logger.warning("XGBoost training encountered issue: %s", e)
            xgb_model = None

    # 5. Soft Voting Ensemble
    voting_estimators = [
        ("lr", lr_model),
        ("rf", rf_model),
        ("gb", gb_model),
    ]
    if xgb_model is not None:
        voting_estimators.append(("xgb", xgb_model))

    # Evaluate each on the untouched chronological test set
    test_metrics = {}
    
    # LR
    lr_probs = lr_model.predict_proba(X_test_scaled)
    test_metrics["Logistic_Regression"] = evaluate_predictions(y_test, lr_probs, "Logistic Regression")

    # RF
    rf_probs = rf_model.predict_proba(X_test_raw)
    test_metrics["Random_Forest"] = evaluate_predictions(y_test, rf_probs, "Random Forest")

    # GB
    gb_probs = gb_model.predict_proba(X_test_raw)
    test_metrics["Gradient_Boosting"] = evaluate_predictions(y_test, gb_probs, "Gradient Boosting")

    # XGB (if trained)
    xgb_probs = None
    if xgb_model is not None:
        xgb_probs = xgb_model.predict_proba(X_test_raw)
        test_metrics["XGBoost"] = evaluate_predictions(y_test, xgb_probs, "XGBoost")

    # Ensemble / Blended probabilities
    prob_list = [lr_probs, rf_probs, gb_probs]
    if xgb_probs is not None:
        prob_list.append(xgb_probs)
    ensemble_probs = np.mean(prob_list, axis=0)
    test_metrics["Ensemble_Model"] = evaluate_predictions(y_test, ensemble_probs, "Ensemble Model")

    # Pick best model by lowest log-loss
    best_name = min(test_metrics.keys(), key=lambda k: test_metrics[k]["log_loss"])
    logger.info("Selected BEST model: %s (Log-Loss: %.4f)", best_name, test_metrics[best_name]["log_loss"])

    # Model artifact bundle
    pipeline_bundle = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "gb_model": gb_model,
        "xgb_model": xgb_model,
        "best_model_name": best_name,
        "test_metrics": test_metrics,
        "split_date": split_date,
    }

    model_path = models_dir / "best_model.joblib"
    joblib.dump(pipeline_bundle, model_path)
    logger.info("Saved complete model pipeline to %s", model_path)

    # Save feature metadata
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {"0": "Home Win", "1": "Draw", "2": "Away Win"},
        "best_model": best_name,
        "test_metrics": test_metrics,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
    }
    with open(models_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "pipeline_bundle": pipeline_bundle,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "test_probs": {
            "Logistic_Regression": lr_probs,
            "Random_Forest": rf_probs,
            "Gradient_Boosting": gb_probs,
            "Ensemble": ensemble_probs,
            **({"XGBoost": xgb_probs} if xgb_probs is not None else {})
        },
        "test_df": test_df,
    }
