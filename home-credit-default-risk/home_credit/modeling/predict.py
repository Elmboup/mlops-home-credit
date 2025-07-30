import joblib
from pathlib import Path

def predict(model, X):
    """Retourne les prédictions de classes du modèle."""
    return model.predict(X)

def predict_proba(model, X):
    """Retourne les probabilités prédites par le modèle (colonne 1 = proba positive)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Le modèle ne supporte pas predict_proba.")

def save_model(model, path):
    """Sauvegarde le modèle dans le dossier models/ au format joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(path)

# --- Code CLI existant (commenté, à adapter si besoin) ---
# from loguru import logger
# from tqdm import tqdm
# import typer
# from home_credit.config import MODELS_DIR, PROCESSED_DATA_DIR
# app = typer.Typer()
# @app.command()
# def main(
#     features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
# ):
#     logger.info("Performing inference for model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Inference complete.")
# if __name__ == "__main__":
#     app()