# home_credit/dashboard/app.py
import streamlit as st

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.pipeline import Pipeline



# Paths
MODEL_PATH = Path("models/best_model.joblib")
DATA_PATH = Path("home-credit-default-risk/data/processed/test_merged.csv")

# Charger le modèle
@st.cache
def load_model():
    return joblib.load(MODEL_PATH)

# Charger les données test
@st.cache
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Titre du dashboard
st.set_page_config(page_title="Dashboard Crédit Client", layout="wide")
st.title("Dashboard - Crédit Client & Risque de Défaut")

# Charger modèle et données
model = load_model()
test_df = load_data()
client_ids = test_df["SK_ID_CURR"].values

# Sélection du client
selected_client_id = st.selectbox("Sélectionner un client", client_ids)
client_data = test_df[test_df["SK_ID_CURR"] == selected_client_id]

# Prédiction
prediction = model.predict(client_data)[0]
proba = model.predict_proba(client_data)[0][1]

st.markdown(f"### Prédiction : {'🟥 Risque élevé' if prediction == 1 else '🟩 Faible risque'}")
st.metric("Probabilité de défaut", f"{proba:.2%}")

# Afficher les infos principales
with st.expander("Informations du client"):
    st.dataframe(client_data.T)

# Interprétation SHAP
with st.expander("Interprétation du modèle (SHAP)"):
    st.markdown("Voici les variables les plus influentes pour ce client :")

    # Extraire les étapes du pipeline
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Appliquer le prétraitement
    X_test_processed = preprocessor.transform(test_df)
    X_client_processed = preprocessor.transform(client_data)

    # SHAP explainer
    explainer = shap.Explainer(classifier, X_test_processed)
    shap_values = explainer(X_client_processed)

    fig_shap = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)
