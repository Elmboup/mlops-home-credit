from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from home_credit.config import MODELS_DIR, PROCESSED_DATA_DIR
from sklearn.model_selection import train_test_split, GridSearchCV
from home_credit.config import RANDOM_SEED, TEST_SIZE

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=True):
    """
    Split les données en train/test avec stratification par défaut.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

def train_model(model, X_train, y_train):
    """
    Entraîne un modèle sklearn classique.
    """
    model.fit(X_train, y_train)
    return model

def grid_search(model, param_grid, X_train, y_train, scoring="recall", cv=5, n_jobs=-1):
    """
    Effectue une recherche d'hyperparamètres par GridSearchCV.
    Retourne le meilleur estimateur, ses paramètres et le meilleur score.
    Optimise par défaut sur le rappel (recall).
    """
    gs = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def compare_gridsearch_models(results):
    """
    Prend une liste de tuples (name, best_estimator, best_params, best_score) issus de grid_search
    et retourne le meilleur modèle, son nom, ses paramètres et son score.
    """
    if not results:
        raise ValueError("La liste des résultats est vide.")
    best = max(results, key=lambda x: x[3])  # x[3] = best_score (recall)
    best_name, best_model, best_params, best_score = best
    return best_name, best_model, best_params, best_score

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
