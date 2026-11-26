"""Dataset loading utilities for the UCI Student Performance dataset."""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

NUMERIC_FEATURES = [
    "G1",
    "G2",
    "absences",
    "studytime",
    "failures",
]

CATEGORICAL_FEATURES = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
]

TARGET_COLUMN = "G3"


def load_student_performance(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Student Performance dataset from disk, or download from UCI if missing."""
    if data_path.exists():
        return pd.read_csv(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    resp = urllib.request.urlopen(UCI_URL, timeout=30)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        with zf.open("student-por.csv") as f:
            df = pd.read_csv(f, sep=";")
    df.to_csv(data_path, index=False)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate selected feature matrix X (10 core features) and continuous target y (G3)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y
