# import streamlit as st
# import pandas as pd
# import requests
# import shap
# import joblib
# import matplotlib.pyplot as plt
# from home_credit.config import MODELS_DIR

# # Configuration
# st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
# st.title("📊 Credit Scoring Dashboard")

# API_URL = "http://localhost:8000/predict"

# # Charger le modèle localement pour SHAP
# model = joblib.load(MODELS_DIR / "best_model.joblib")

# # Choix du mode d'entrée
# mode = st.radio("📥 Méthode de saisie des données :", ["📄 Uploader un fichier JSON", "✍️ Saisir manuellement"])

# input_df = None

# # Option 1 : Upload JSON
# if mode == "📄 Uploader un fichier JSON":
#     uploaded_file = st.file_uploader("Téléchargez un fichier JSON", type=["json"])
#     if uploaded_file is not None:
#         try:
#             input_data = pd.read_json(uploaded_file, typ='series')
#             input_df = pd.DataFrame([input_data])
#             st.success("✅ Données chargées avec succès")
#             st.json(input_data.to_dict())
#         except Exception as e:
#             st.error(f"Erreur lors du chargement du fichier : {e}")

# # Option 2 : Formulaire manuel (simplifié)
# elif mode == "✍️ Saisir manuellement":
#     st.subheader("Saisie manuelle de quelques données client (exemple)")
#     data = {}
#     data["SK_ID_CURR"] = st.number_input("ID Client", min_value=100000, step=1)
#     data["NAME_CONTRACT_TYPE"] = st.selectbox("Type de contrat", ["Cash loans", "Revolving loans"])
#     data["CODE_GENDER"] = st.selectbox("Genre", ["M", "F"])
#     data["FLAG_OWN_CAR"] = st.selectbox("Possède une voiture ?", ["Y", "N"])
#     data["AMT_INCOME_TOTAL"] = st.number_input("Revenu annuel", min_value=0.0)
#     data["AMT_CREDIT"] = st.number_input("Montant crédit", min_value=0.0)
#     data["DAYS_BIRTH"] = st.number_input("Jours depuis naissance", value=-12000)
#     data["EXT_SOURCE_2"] = st.number_input("Score source externe 2", min_value=0.0, max_value=1.0, value=0.5)

#     # ➕ Ajoute ici d'autres champs si nécessaire
#     input_df = pd.DataFrame([data])

#     st.success("✅ Données saisies avec succès")
#     st.json(data)

# # Envoyer les données à l’API
# if input_df is not None and st.button("Prédire"):
#     try:
#         # Envoi au format dict
#         response = requests.post(API_URL, json=input_df.iloc[0].to_dict())
#         result = response.json()

#         st.markdown("## 🧾 Résultat de la prédiction")
#         st.markdown(f"**Décision** : `{result['decision']}`")
#         st.metric("Probabilité d'approbation", f"{result['probability_approved']:.2%}")
#         st.metric("Probabilité de refus", f"{result['probability_refused']:.2%}")

#         # Affichage SHAP
#         st.markdown("### 🔍 Explication SHAP")
#         explainer = shap.Explainer(model)
#         shap_values = explainer(input_df)

#         fig = plt.figure()
#         shap.plots.waterfall(shap_values[0], show=False)
#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f" Erreur pendant la prédiction : {e}")


import streamlit as st
import pandas as pd
import requests
import shap
import joblib
import matplotlib.pyplot as plt
from home_credit.config import MODELS_DIR

# Configuration page
st.set_page_config(page_title="Credit Scoring", layout="wide")
st.title(" Credit Scoring Dashboard")

# Charger le modèle localement pour SHAP
model = joblib.load(MODELS_DIR / "best_model.joblib")

#  URL de l'API (en local)
API_URL = "http://localhost:8000/predict"

# --- 1. UPLOAD DES DONNÉES ---
st.header(" 1. Upload des données client")

uploaded_file = st.file_uploader("Chargez un fichier CSV avec les données d'un client", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Affiche le DataFrame brut
    st.subheader("Aperçu des données chargées :")
    st.dataframe(input_df)

    # --- 2. PRÉDICTION ---
    st.header(" 2. Prédiction du score")

    # Convertir la première ligne en dictionnaire pour l'API
    input_data = input_df.iloc[0].to_dict()

    if st.button("Faire la prédiction"):
        try:
            response = requests.post(API_URL, json=input_data)
            result = response.json()

            st.success(" Prédiction réussie !")
            st.markdown(f"###  Décision : **{result['decision']}**")
            st.metric(" Probabilité d'approbation", f"{result['probability_approved']:.2%}")
            st.metric(" Probabilité de refus", f"{result['probability_refused']:.2%}")

            # --- 3. EXPLICATION SHAP ---
            st.header(" 3. Explication SHAP")

            # Préparation de l’explainer et des valeurs SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            # Graphique SHAP pour le client
            st.subheader("Impact des variables sur la prédiction")
            fig = plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f" Erreur lors de la prédiction : {e}")
else:
    st.info("Veuillez charger un fichier CSV pour commencer.")
