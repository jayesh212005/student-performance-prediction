# Student Academic Performance Prediction

---

## 1. Problem Statement
- **What is the problem?** Teachers and advisors often notice struggling students too late in the term. Predicting final marks early from attendance, internal assessments, and study habits helps identify students who may need extra support.
- **Who is it for?** Teachers, academic advisors, and students.
- **What is the benefit?** Early support, timely tutoring, and clear insight into which study habits matter most.

---

## 2. Dataset
- **Source:** UCI Machine Learning Repository — *Student Performance* (`student-por.csv`), saved at `data/student_performance.csv`.
- **Size:** **649 student records**, **10 core academic and study features** (selected from 33 original columns), and **1 target column** (`G3`, final mark from `0` to `20`).
- **Selected Features (10):** `G1` (1st period grade), `G2` (2nd period grade), `absences`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `higher`.
- **Limitations:** Small sample from two Portuguese schools; does not capture personal emergencies or teaching differences.

---

## 3. Algorithms Compared
1. **Linear Regression:** Simple, transparent baseline showing how each feature shifts the predicted mark.
2. **Ridge Regression:** Regularized linear model (tuned `alpha` on the validation set).
3. **Decision Tree Regression:** Rule-based model (tuned `max_depth` and `min_samples_leaf` on the validation set).

---

## 4. Evaluation & Error Analysis (60% Train / 20% Val / 20% Test)

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** *(Best Val RMSE)* | 0.906 | **1.453** | **0.807** |
| **Ridge Regression** | 0.906 | **1.453** | **0.807** |
| **Decision Tree** | **0.876** | 1.551 | 0.780 |

Charts are saved in `artifacts/plots/` (`model_comparison.png`, `actual_vs_predicted.png`, `feature_importance.png`).

---

## 5. Ethics & Responsible Use
- **Privacy:** Student grades and attendance are private; real systems must anonymize records and limit access.
- **Fairness:** We excluded demographic attributes (`sex`, `address`) from model inputs and audited test errors across those groups.
- **Responsible Use:** Never use a predicted low mark to discourage a student or deny opportunities—use it only to offer timely help.

---

## 6. How to Run
```bash
pip install -r requirements.txt
python -m src.train_evaluate
python -m pytest -v
streamlit run app.py
```
