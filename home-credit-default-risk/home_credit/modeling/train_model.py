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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins
DATA_PATH = Path("home-credit-default-risk/data/processed/train_merged.csv")
MODEL_PATH = Path("models/best_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# MLflow
mlflow.set_experiment("home-credit-default-risk")

def load_data():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape}")
    return df

def build_pipeline(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    return pipe

def main():
    df = load_data()
    TARGET = "TARGET"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X)

    param_grid = {
        "model__max_depth": [3, 5],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name="xgboost_gridsearch"):
        logger.info("Training with GridSearchCV...")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        logger.info(f"Best params: {best_params}")
        logger.info(f"Best ROC AUC: {best_score:.4f}")

        # Log via MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Sauvegarde du modèle
        joblib.dump(best_model, MODEL_PATH)
        logger.info(f"Modèle sauvegardé → {MODEL_PATH}")

if __name__ == "__main__":
    main()
