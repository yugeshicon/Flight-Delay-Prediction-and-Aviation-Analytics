"""Model training and evaluation module for Flight Delay Prediction.

Trains multiple classification candidates (Logistic Regression, Decision Tree,
Random Forest, LightGBM/XGBoost) and a delay duration regressor. Evaluates models
on unseen test data, logs performance metrics, and exports champion artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.cleaning import clean_flight_data
from src.data_loader import load_flight_data
from src.pipeline import fit_preprocessor, prepare_features, save_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_classifier(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a binary classifier.

    Args:
        name: Name of the candidate model.
        model: Fitted classification estimator.
        X_test: Transformed feature test matrix.
        y_test: Ground truth binary targets.

    Returns:
        Dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    logger.info(
        "[%s] Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | ROC-AUC: %.4f",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


def train_models(
    df: Optional[pd.DataFrame] = None,
    sample_size: Optional[int] = 100000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any, Any, Any]:
    """Train candidate classification models and delay regressor.

    Args:
        df: Optional pre-loaded flight DataFrame. If None, loaded via data_loader.
        sample_size: Optional sampling count for training efficiency (default: 100k).
        test_size: Fraction of data reserved for testing (default: 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: (comparison_results, champion_classifier, delay_regressor, preprocessor)
    """
    if df is None:
        logger.info("Loading flight data for model training (sample_size=%s)...", sample_size)
        df = load_flight_data(dataset_type="selected", nrows=sample_size)

    # 1. Clean and filter to completed flights for modeling
    df_clean = clean_flight_data(df, for_modeling=True)
    logger.info("Cleaned dataset ready for modeling: %d rows", len(df_clean))

    # 2. Extract targets
    y_cls = df_clean["is_delayed"].values
    y_reg = df_clean["delay_minutes"].values

    # 3. Train/Test Split BEFORE feature transformation to avoid data leakage
    df_train, df_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        df_clean,
        y_cls,
        y_reg,
        test_size=test_size,
        stratify=y_cls,
        random_state=random_state,
    )
    logger.info(
        "Train set: %d rows (%.1f%% delayed) | Test set: %d rows (%.1f%% delayed)",
        len(df_train),
        y_train_cls.mean() * 100,
        len(df_test),
        y_test_cls.mean() * 100,
    )

    # 4. Fit Preprocessor exclusively on training data
    preprocessor, X_train = fit_preprocessor(df_train)
    X_test_df = prepare_features(df_test)
    X_test = preprocessor.transform(X_test_df)
    logger.info("Feature matrices ready: X_train %s, X_test %s", X_train.shape, X_test.shape)

    # 5. Candidate Classifiers
    candidate_models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    if HAS_LIGHTGBM:
        candidate_models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    # Train and evaluate all classifiers
    comparison_metrics = []
    trained_classifiers = {}

    for name, model in candidate_models.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train_cls)
        metrics = evaluate_classifier(name, model, X_test, y_test_cls)
        comparison_metrics.append(metrics)
        trained_classifiers[name] = model

    # Select champion classifier based on ROC-AUC
    best_candidate = max(comparison_metrics, key=lambda m: m["roc_auc"])
    champion_name = best_candidate["model_name"]
    champion_model = trained_classifiers[champion_name]
    logger.info("Selected Champion Classifier: %s (ROC-AUC: %.4f)", champion_name, best_candidate["roc_auc"])

    # 6. Train Delay Duration Regressor on delayed flights in train set
    delayed_train_mask = y_train_cls == 1
    X_train_delayed = X_train[delayed_train_mask]
    y_train_delayed_mins = y_train_reg[delayed_train_mask]

    logger.info("Training Delay-Duration Regressor on %d delayed training flights...", len(X_train_delayed))
    if HAS_LIGHTGBM:
        regressor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        regressor = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    regressor.fit(X_train_delayed, y_train_delayed_mins)

    # Evaluate regressor on delayed test flights
    delayed_test_mask = y_test_cls == 1
    X_test_delayed = X_test[delayed_test_mask]
    y_test_delayed_mins = y_test_reg[delayed_test_mask]

    y_pred_mins = regressor.predict(X_test_delayed)
    reg_mae = round(float(mean_absolute_error(y_test_delayed_mins, y_pred_mins)), 2)
    reg_rmse = round(float(root_mean_squared_error(y_test_delayed_mins, y_pred_mins)), 2)
    logger.info("Delay Regressor Test Performance: MAE = %.2f mins | RMSE = %.2f mins", reg_mae, reg_rmse)

    # 7. Extract Feature Importances from Champion Model
    feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    importances = []
    if hasattr(champion_model, "feature_importances_"):
        raw_imp = champion_model.feature_importances_
        sorted_indices = np.argsort(raw_imp)[::-1]
        for idx in sorted_indices[:15]:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            importances.append({"feature": fname, "importance": round(float(raw_imp[idx]), 4)})
    elif hasattr(champion_model, "coef_"):
        raw_coef = np.abs(champion_model.coef_[0])
        sorted_indices = np.argsort(raw_coef)[::-1]
        for idx in sorted_indices[:15]:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            importances.append({"feature": fname, "importance": round(float(raw_coef[idx]), 4)})

    # 8. Save Artifacts
    clf_path = MODELS_DIR / "flight_delay_classifier.pkl"
    reg_path = MODELS_DIR / "flight_delay_regressor.pkl"
    prep_path = MODELS_DIR / "preprocessor_pipeline.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"

    joblib.dump(champion_model, clf_path)
    joblib.dump(regressor, reg_path)
    save_preprocessor(preprocessor, prep_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "champion_model": champion_name,
        "sample_size_trained": len(df_clean),
        "test_size": test_size,
        "comparison_metrics": comparison_metrics,
        "regression_metrics": {"mae_mins": reg_mae, "rmse_mins": reg_rmse},
        "top_features": importances,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("All artifacts saved successfully to %s", MODELS_DIR)
    return metadata, champion_model, regressor, preprocessor


if __name__ == "__main__":
    train_models(sample_size=150000)
