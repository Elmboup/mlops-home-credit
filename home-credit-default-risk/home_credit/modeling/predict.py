import pandas as pd
import joblib
import argparse
from pathlib import Path
from sklearn.base import BaseEstimator

# Load pipeline and predict
def load_model(model_path: str) -> BaseEstimator:
    return joblib.load(model_path)

def predict(model: BaseEstimator, input_data: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict_proba(input_data)[:, 1]  # Proba de défaut (class 1)
    return pd.DataFrame({"SK_ID_CURR": input_data.index, "default_proba": preds})

def main(model_path: str, input_csv: str, output_csv: str):
    model = load_model(model_path)
    data = pd.read_csv(input_csv, index_col="SK_ID_CURR")
    predictions = predict(model, data)
    predictions.to_csv(output_csv, index=False)
    print(f" Prédictions sauvegardées dans {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.model_path, args.input_csv, args.output_csv)
