# home_credit/modeling/train.py

import pandas as pd
import numpy as np
import joblib
import logging
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)
mlflow.set_experiment("home-credit-default-risk")

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("home-credit-default-risk/data/processed")
MODEL_DIR = Path("home-credit-default-risk/models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "TARGET"
RANDOM_STATE = 42

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("home-credit-default-risk/data/processed/train_merged.csv")
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -------------------------------
# Preprocessing pipeline
# -------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# -------------------------------
# Models to train
# -------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

results = []
best_auc = 0
best_model = None
best_name = ""

for name, clf in models.items():
    logger.info(f"Training {name}")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    with mlflow.start_run(run_name=name):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_valid)
        proba = pipe.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, proba)
        acc = accuracy_score(y_valid, preds)
        f1 = f1_score(y_valid, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(pipe, "model")

        results.append({
            "model": name, "roc_auc": auc,
            "accuracy": acc, "f1": f1
        })

        if auc > best_auc:
            best_auc = auc
            best_model = pipe
            best_name = name

# -------------------------------
# Save best model
# -------------------------------
joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
logger.info(f" Best model ({best_name}) saved with AUC: {best_auc:.4f}")
