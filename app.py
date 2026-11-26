"""Interactive Streamlit prototype for Student Academic Performance Prediction."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_student_performance

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "High-Performing Student": {
        "G1": 16,
        "G2": 17,
        "absences": 2,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "higher": "yes",
    },
    "Average Student": {
        "G1": 11,
        "G2": 12,
        "absences": 6,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "no",
        "higher": "yes",
    },
    "At-Risk Student (Needs Support)": {
        "G1": 7,
        "G2": 6,
        "absences": 18,
        "studytime": 1,
        "failures": 2,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "higher": "no",
    },
}


@st.cache_resource
def load_artifacts():
    """Load trained models and evaluation summary."""
    models = joblib.load(ARTIFACTS_DIR / "all_models.joblib")
    with open(ARTIFACTS_DIR / "metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    df = load_student_performance()
    return models, summary, df


def main() -> None:
    st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
    st.title("Student Academic Performance Prediction")
    st.caption("A decision-support demonstration using the UCI Student Performance dataset (0–20 grading scale).")

    if not (ARTIFACTS_DIR / "all_models.joblib").exists():
        st.error("Trained models not found. Please run `python -m src.train_evaluate` first.")
        return

    models, summary, _ = load_artifacts()

    with st.sidebar:
        st.header("Prediction Settings")
        preset_name = st.selectbox("Sample Student Profile", list(PRESET_PROFILES.keys()))
        model_names = list(models.keys())
        selected_model_name = st.selectbox("Regression Model", model_names, index=0)
        pass_mark = st.slider("Pass Mark Threshold (0–20)", min_value=5.0, max_value=15.0, value=10.0, step=0.5)

    tab_predict, tab_eval, tab_ethics = st.tabs(
        ["Predict Final Marks", "Model Comparison & Errors", "Dataset & Ethics"]
    )

    with tab_predict:
        preset = PRESET_PROFILES[preset_name]
        user_input: Dict[str, Any] = {}

        st.subheader("Student Inputs (10 Core Attributes)")
        with st.expander("Internal Assessments & Attendance", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                user_input["G1"] = st.slider("1st Period Grade (G1, 0–20)", 0, 20, int(preset["G1"]))
            with c2:
                user_input["G2"] = st.slider("2nd Period Grade (G2, 0–20)", 0, 20, int(preset["G2"]))
            with c3:
                user_input["absences"] = st.number_input("School Absences", 0, 93, int(preset["absences"]))

        with st.expander("Study Patterns & Academic Support", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                user_input["studytime"] = st.slider(
                    "Weekly Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)", 1, 4, int(preset["studytime"])
                )
                user_input["failures"] = st.slider("Past Class Failures", 0, 3, int(preset["failures"]))
            with c2:
                user_input["schoolsup"] = st.selectbox(
                    "Extra School Support", ["no", "yes"], index=["no", "yes"].index(preset["schoolsup"])
                )
                user_input["famsup"] = st.selectbox(
                    "Family Study Support", ["no", "yes"], index=["no", "yes"].index(preset["famsup"])
                )
            with c3:
                user_input["paid"] = st.selectbox(
                    "Extra Paid Classes", ["no", "yes"], index=["no", "yes"].index(preset["paid"])
                )
                user_input["activities"] = st.selectbox(
                    "Extracurricular Activities", ["no", "yes"], index=["no", "yes"].index(preset["activities"])
                )
            with c4:
                user_input["higher"] = st.selectbox(
                    "Plans Higher Education", ["no", "yes"], index=["no", "yes"].index(preset["higher"])
                )

        input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        model = models[selected_model_name]
        pred_g3 = float(min(max(model.predict(input_df)[0], 0.0), 20.0))

        st.divider()
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Final Mark (G3)", f"{pred_g3:.1f} / 20")
        r2.metric("Status", "ON TRACK (Pass)" if pred_g3 >= pass_mark else "AT RISK (Below Pass)")
        r3.metric("Model Used", selected_model_name)

        st.progress(min(max(pred_g3 / 20.0, 0.0), 1.0))
        if pred_g3 < pass_mark:
            st.warning(
                f"Predicted mark ({pred_g3:.1f}) is below the {pass_mark:.1f} pass threshold. "
                "Consider scheduling tutoring or an advisor check-in."
            )
        else:
            st.success(f"Predicted mark ({pred_g3:.1f}) is at or above the {pass_mark:.1f} pass threshold.")

    with tab_eval:
        st.subheader("Test Set Performance (Held-Out Students)")
        rows = []
        for name, m_data in summary["models"].items():
            t = m_data["test"]
            rows.append({"Model": name, "MAE": t["mae"], "RMSE": t["rmse"], "R2 Score": t["r2"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        p1, p2 = st.columns(2)
        with p1:
            st.image(str(PLOTS_DIR / "model_comparison.png"), caption="MAE & RMSE Comparison")
            st.image(str(PLOTS_DIR / "feature_importance.png"), caption="Top Features Affecting Marks")
        with p2:
            st.image(str(PLOTS_DIR / "actual_vs_predicted.png"), caption="Actual vs. Predicted & Residuals")

    with tab_ethics:
        st.subheader("Dataset Summary & Fairness Check")
        st.write(
            "- **Source:** UCI Student Performance (`student-por.csv`) — 649 student records, 10 core academic/study features.\n"
            "- **Limitations:** Small sample from two Portuguese schools; does not capture personal emergencies or teaching quality."
        )
        for attr, records in summary["fairness_audit"].items():
            st.markdown(f"**Group: `{attr}`**")
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
