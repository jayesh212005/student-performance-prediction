"""Automated tests for Student Performance Prediction pipeline."""

import json
from pathlib import Path
import joblib

from src.data_loader import load_student_performance, split_features_target
from src.preprocessing import make_train_val_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_and_split():
    df = load_student_performance()
    assert df.shape == (649, 33)
    X, y = split_features_target(df)
    assert X.shape == (649, 10)
    splits = make_train_val_test_split(X, y, random_state=42)
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == 649


def test_saved_artifacts_and_prediction():
    artifacts = PROJECT_ROOT / "artifacts"
    assert (artifacts / "best_model.joblib").exists()
    assert (artifacts / "all_models.joblib").exists()
    assert (artifacts / "metrics_summary.json").exists()

    with open(artifacts / "metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert set(summary["models"].keys()) == {"Linear Regression", "Ridge Regression", "Decision Tree"}

    models = joblib.load(artifacts / "all_models.joblib")
    df = load_student_performance()
    X, _ = split_features_target(df)
    sample = X.iloc[[0]]
    for name, pipe in models.items():
        pred = float(pipe.predict(sample)[0])
        assert -5.0 <= pred <= 25.0, f"Unexpected prediction from {name}: {pred}"
