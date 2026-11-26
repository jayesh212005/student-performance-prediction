"""Train, tune, evaluate, and analyze Linear Regression, Ridge, and Decision Tree Regression."""

import json
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import PROJECT_ROOT, load_student_performance, split_features_target
from src.preprocessing import DatasetSplits, build_preprocessor, make_train_val_test_split

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def get_candidate_models() -> Dict[str, Tuple[Any, List[Dict[str, Any]]]]:
    """Return the three candidate regressors and their validation hyperparameter grids."""
    return {
        "Linear Regression": (LinearRegression(), [{}]),
        "Ridge Regression": (
            Ridge(random_state=42),
            [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 5.0}, {"alpha": 10.0}, {"alpha": 25.0}],
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            [
                {"max_depth": 3, "min_samples_leaf": 10},
                {"max_depth": 4, "min_samples_leaf": 8},
                {"max_depth": 5, "min_samples_leaf": 5},
                {"max_depth": 6, "min_samples_leaf": 5},
            ],
        ),
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics: MAE, RMSE, and R2."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def tune_and_train_models(splits: DatasetSplits) -> Tuple[Dict[str, Pipeline], Dict[str, Any]]:
    """Tune each regressor on Validation RMSE and evaluate on Validation & Test."""
    candidates = get_candidate_models()
    trained_models: Dict[str, Pipeline] = {}
    results: Dict[str, Any] = {}

    for name, (estimator, param_grid) in candidates.items():
        best_val_rmse = float("inf")
        best_pipeline = None
        best_params = None
        best_val_metrics = None

        for params in param_grid:
            reg = clone(estimator).set_params(**params)
            pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("regressor", reg)])
            pipe.fit(splits.X_train, splits.y_train)
            val_pred = pipe.predict(splits.X_val)
            val_metrics = compute_metrics(splits.y_val, val_pred)

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_pipeline = pipe
                best_params = params
                best_val_metrics = val_metrics

        test_pred = best_pipeline.predict(splits.X_test)
        test_metrics = compute_metrics(splits.y_test, test_pred)

        trained_models[name] = best_pipeline
        results[name] = {
            "best_params": best_params,
            "validation": best_val_metrics,
            "test": test_metrics,
        }

    return trained_models, results


def run_error_analysis(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Analyze largest residuals on the held-out test set."""
    y_pred = best_model.predict(X_test)
    df = X_test.copy()
    df["actual_G3"] = y_test.values
    df["predicted_G3"] = np.round(y_pred, 2)
    df["error"] = np.round(df["predicted_G3"] - df["actual_G3"], 2)
    df["abs_error"] = np.abs(df["error"])

    worst_cases = (
        df.sort_values("abs_error", ascending=False)
        .head(5)[["G1", "G2", "absences", "studytime", "failures", "actual_G3", "predicted_G3", "error"]]
        .to_dict(orient="records")
    )
    return {
        "mean_error": round(float(df["error"].mean()), 4),
        "median_abs_error": round(float(df["abs_error"].median()), 4),
        "max_abs_error": round(float(df["abs_error"].max()), 4),
        "largest_errors": worst_cases,
    }


def run_fairness_audit(
    best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, raw_test_df: pd.DataFrame
) -> Dict[str, List[Dict[str, Any]]]:
    """Check prediction accuracy across demographic groups on held-out test data."""
    df = raw_test_df.copy().reset_index(drop=True)
    df["y_true"] = y_test.values
    df["y_pred"] = best_model.predict(X_test)

    audit: Dict[str, List[Dict[str, Any]]] = {}
    for col in ["sex", "address"]:
        rows = []
        for group_val, grp in df.groupby(col):
            rows.append(
                {
                    "group": str(group_val),
                    "count": int(len(grp)),
                    "avg_actual_G3": round(float(grp["y_true"].mean()), 2),
                    "avg_predicted_G3": round(float(grp["y_pred"].mean()), 2),
                    "mae": round(float(mean_absolute_error(grp["y_true"], grp["y_pred"])), 3),
                }
            )
        audit[col] = rows
    return audit


def generate_plots(
    trained_models: Dict[str, Pipeline],
    results: Dict[str, Any],
    splits: DatasetSplits,
    best_model_name: str,
) -> None:
    """Save regression evaluation and interpretability plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Model comparison (MAE & RMSE)
    rows = []
    for name, res in results.items():
        rows.append({"Model": name, "Metric": "MAE", "Score": res["test"]["mae"]})
        rows.append({"Model": name, "Metric": "RMSE", "Score": res["test"]["rmse"]})
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=pd.DataFrame(rows), x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Test Set Error Comparison (Lower is Better)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 2. Actual vs Predicted & Residuals for best model
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(splits.X_test)
    residuals = y_pred - splits.y_test.values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(splits.y_test, y_pred, alpha=0.65, color="#2b6cb0")
    axes[0].plot([0, 20], [0, 20], "r--", linewidth=1.2)
    axes[0].set_title(f"Actual vs. Predicted Final Marks ({best_model_name})")
    axes[0].set_xlabel("Actual G3 (0-20)")
    axes[0].set_ylabel("Predicted G3 (0-20)")

    axes[1].scatter(y_pred, residuals, alpha=0.65, color="#dd6b20")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Residual Plot ({best_model_name})")
    axes[1].set_xlabel("Predicted G3")
    axes[1].set_ylabel("Residual (Predicted - Actual)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3. Feature coefficients from Ridge Regression
    ridge_pipe = trained_models["Ridge Regression"]
    feature_names = ridge_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = ridge_pipe.named_steps["regressor"].coef_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
        .sort_values("abs_coef", ascending=False)
        .head(10)
        .sort_values("coef", ascending=True)
    )
    plt.figure(figsize=(9, 4.8))
    colors = ["#2e8b57" if c > 0 else "#d9534f" for c in imp_df["coef"]]
    plt.barh(imp_df["feature"], imp_df["coef"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 10 Features Affecting Final Marks (Ridge Regression)")
    plt.xlabel("Coefficient (+ increases predicted mark, - decreases)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    """Run training, evaluation, error analysis, fairness audit, and save artifacts."""
    df = load_student_performance()
    X, y = split_features_target(df)
    splits = make_train_val_test_split(X, y, random_state=42)

    trained_models, results = tune_and_train_models(splits)
    best_model_name = min(results.keys(), key=lambda m: results[m]["validation"]["rmse"])
    best_model = trained_models[best_model_name]

    error_analysis = run_error_analysis(best_model, splits.X_test, splits.y_test)
    raw_splits = make_train_val_test_split(df, y, random_state=42)
    fairness_audit = run_fairness_audit(best_model, splits.X_test, splits.y_test, raw_splits.X_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    joblib.dump(trained_models, ARTIFACTS_DIR / "all_models.joblib")

    summary = {
        "dataset": {
            "total_records": int(len(df)),
            "num_features": int(X.shape[1]),
            "target": "G3 (Final Grade, 0-20)",
            "splits": {
                "train": int(len(splits.X_train)),
                "validation": int(len(splits.X_val)),
                "test": int(len(splits.X_test)),
            },
        },
        "best_model_name": best_model_name,
        "models": results,
        "error_analysis": error_analysis,
        "fairness_audit": fairness_audit,
    }
    with open(ARTIFACTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    generate_plots(trained_models, results, splits, best_model_name)

    print(f"Best model (selected on Validation RMSE): {best_model_name}")
    for name, res in results.items():
        t = res["test"]
        print(f"  {name:20s} | MAE={t['mae']:.3f} | RMSE={t['rmse']:.3f} | R2={t['r2']:.3f}")


if __name__ == "__main__":
    main()
