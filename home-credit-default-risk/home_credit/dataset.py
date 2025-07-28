# src/data/make_dataset.py

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Répertoire racine des données
DATA_DIR = Path(__file__).resolve().parents[2] / "home-credit-default-risk/data"


def load_raw_data():
    logger.info("📦 Chargement des fichiers bruts...")
    return {
        "train": pd.read_csv(DATA_DIR / "raw/application_train.csv"),
        "test": pd.read_csv(DATA_DIR / "raw/application_test.csv"),
        "bureau": pd.read_csv(DATA_DIR / "raw/bureau.csv"),
        "bureau_balance": pd.read_csv(DATA_DIR / "raw/bureau_balance.csv"),
        "previous_app": pd.read_csv(DATA_DIR / "raw/previous_application.csv"),
        "pos_cash": pd.read_csv(DATA_DIR / "raw/POS_CASH_balance.csv"),
        "credit_card": pd.read_csv(DATA_DIR / "raw/credit_card_balance.csv"),
        "insta_payments": pd.read_csv(DATA_DIR / "raw/installments_payments.csv"),
    }


def aggregate_data(data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les agrégations et jointures à une base (train ou test)
    """
    train = df.copy()

    # Bureau balance
    bureau_balance_agg = data["bureau_balance"].groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["count", "min", "max"]
    }).reset_index()
    bureau_balance_agg.columns = ["_".join(col).strip("_") for col in bureau_balance_agg.columns]

    bureau = data["bureau"].merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "SK_ID_BUREAU": "count",
        "CREDIT_ACTIVE": "nunique",
        "AMT_CREDIT_SUM": "sum"
    }).reset_index()
    bureau_agg.columns = ["SK_ID_CURR", "bureau_count", "credit_active_unique", "total_credit_sum"]
    train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Previous application
    previous_agg = data["previous_app"].groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "sum"
    }).reset_index()
    previous_agg.columns = ["SK_ID_CURR", "prev_app_count", "avg_amt_app", "total_amt_credit"]
    train = train.merge(previous_agg, on="SK_ID_CURR", how="left")

    # POS CASH
    pos_cash_agg = data["pos_cash"].groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "MONTHS_BALANCE": "mean"
    }).reset_index()
    pos_cash_agg.columns = ["SK_ID_CURR", "pos_cash_count", "avg_months_balance"]
    train = train.merge(pos_cash_agg, on="SK_ID_CURR", how="left")

    # Credit Card
    credit_card_agg = data["credit_card"].groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_BALANCE": "mean"
    }).reset_index()
    credit_card_agg.columns = ["SK_ID_CURR", "credit_card_count", "avg_credit_balance"]
    train = train.merge(credit_card_agg, on="SK_ID_CURR", how="left")

    # Installment Payments
    installments_agg = data["insta_payments"].groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_PAYMENT": "sum",
        "DAYS_ENTRY_PAYMENT": "mean"
    }).reset_index()
    installments_agg.columns = ["SK_ID_CURR", "installments_count", "total_amt_payment", "avg_days_entry_payment"]
    train = train.merge(installments_agg, on="SK_ID_CURR", how="left")

    return train


def save_dataset(df: pd.DataFrame, filename: str):
    output_path = DATA_DIR / "processed" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sauvegarde du fichier : {output_path.name}")
    df.to_csv(output_path, index=False)


def main():
    data = load_raw_data()

    logger.info("Traitement TRAIN...")
    train_merged = aggregate_data(data, data["train"])
    save_dataset(train_merged, "train_merged.csv")

    logger.info("Traitement TEST...")
    test_merged = aggregate_data(data, data["test"])
    save_dataset(test_merged, "test_merged.csv")

    logger.info("Traitement terminé avec succès.")


if __name__ == "__main__":
    main()
