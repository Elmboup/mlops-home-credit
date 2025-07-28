# features.py – Génération de features et exploration avancée

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
sns.set(style="whitegrid")


def create_age_bins(df: pd.DataFrame, age_col: str = "DAYS_BIRTH") -> pd.DataFrame:
    """Ajoute une colonne de tranches d'âge"""
    df = df.copy()
    df["AGE_BIN"] = pd.cut(
        df[age_col] / -365,
        bins=[20, 30, 40, 50, 60, 70],
        labels=["20-30", "30-40", "40-50", "50-60", "60-70"]
    )
    return df


def plot_income_by_age_bin(df: pd.DataFrame) -> None:
    """Boxplot des revenus par tranche d'âge et TARGET"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="AGE_BIN", y="AMT_INCOME_TOTAL", hue="TARGET", data=df)
    plt.title("Distribution des revenus par tranche d'âge et TARGET")
    plt.ylabel("AMT_INCOME_TOTAL")
    plt.tight_layout()
    plt.show()


def plot_pairplot(df: pd.DataFrame, cols: list[str], sample_size: int = 1000) -> None:
    """Pairplot sur un sous-ensemble de features"""
    sample_df = df[cols].dropna().sample(sample_size, random_state=42)
    sns.pairplot(sample_df, hue="TARGET", diag_kind="kde", palette="Set1")
    plt.suptitle("Pairplot des variables financières et d'âge", y=1.02)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> None:
    """Heatmap des variables les plus corrélées à TARGET"""
    num_df = df.select_dtypes(include=["int64", "float64"])
    corrmat = num_df.corr()
    top_corr_features = corrmat["TARGET"].abs().sort_values(ascending=False).head(top_n).index
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap des variables les plus corrélées à TARGET")
    plt.tight_layout()
    plt.show()


def run_feature_analysis(df: pd.DataFrame) -> None:
    """Pipeline d’analyse exploratoire avancée"""
    df = create_age_bins(df)
    logger.info("Tranches d'âge ajoutées")
    plot_income_by_age_bin(df)

    key_cols = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "TARGET"
    ]
    plot_pairplot(df, key_cols)
    plot_correlation_heatmap(df)
    logger.info("EDA visuelle terminée")
