from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["age", "monthly_spend", "support_tickets"]
CATEGORICAL_FEATURES = ["plan", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def training_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [22, 29.0, 1, "basic", "west", 0],
            [25, 35.0, 2, "basic", "north", 0],
            [31, 42.0, 1, "standard", "west", 0],
            [34, 55.0, 1, "standard", "south", 0],
            [39, 65.0, 2, "premium", "east", 0],
            [45, 72.0, 2, "premium", "north", 0],
            [52, 80.0, 1, "premium", "west", 0],
            [28, 30.0, 6, "basic", "east", 1],
            [33, 38.0, 5, "basic", "south", 1],
            [41, 45.0, 7, "standard", "east", 1],
            [47, 50.0, 6, "standard", "south", 1],
            [55, 58.0, 8, "premium", "east", 1],
            [60, 62.0, 7, "premium", "south", 1],
            [37, 40.0, 9, "basic", "north", 1],
            [49, 68.0, 3, "standard", "west", 0],
            [26, 28.0, 1, "basic", "west", 0],
            [44, 75.0, 2, "premium", "north", 0],
            [58, 70.0, 5, "standard", "east", 1],
            [30, 46.0, 3, "standard", "north", 0],
            [51, 52.0, 6, "basic", "south", 1],
        ],
        columns=FEATURES + [TARGET],
    )


def build_model() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train_model(output_path: str | Path | None = None) -> Pipeline:
    data = training_data()
    model = build_model()
    model.fit(data[FEATURES], data[TARGET])
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    return model


def validate_input(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if data.empty:
        raise ValueError("Input DataFrame must not be empty")
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    extra = set(data.columns) - set(FEATURES)
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")
    if data[NUMERIC_FEATURES].isna().all(axis=None):
        raise ValueError("Numeric features cannot all be missing")
    for c in NUMERIC_FEATURES:
        non_null = data[c].dropna()
        if not pd.api.types.is_numeric_dtype(non_null):
            raise TypeError(f"Feature '{c}' must be numeric")
        if np.isinf(non_null.to_numpy()).any():
            raise ValueError(f"Feature '{c}' must contain finite values")
    return data[FEATURES].copy()


def predict(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(validate_input(data)))


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
