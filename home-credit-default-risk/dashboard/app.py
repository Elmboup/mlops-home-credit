# import streamlit as st
# import pandas as pd
# import requests
# import shap
# import joblib
# import matplotlib.pyplot as plt
# from home_credit.config import MODELS_DIR

# # Configuration page
# st.set_page_config(page_title="Credit Scoring", layout="wide")
# st.title(" Credit Scoring Dashboard")

# # Charger le modèle localement pour SHAP
# model = joblib.load(MODELS_DIR / "best_model.joblib")

# #  URL de l'API (en local)
# API_URL = "http://localhost:8000/predict"

# # --- 1. UPLOAD DES DONNÉES ---
# st.header(" 1. Upload des données client")

# uploaded_file = st.file_uploader("Chargez un fichier CSV avec les données d'un client", type=["csv"])

# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)

#     # Affiche le DataFrame brut
#     st.subheader("Aperçu des données chargées :")
#     st.dataframe(input_df)

#     # --- 2. PRÉDICTION ---
#     st.header(" 2. Prédiction du score")

#     # Convertir la première ligne en dictionnaire pour l'API
#     input_data = input_df.iloc[0].to_dict()

#     if st.button("Faire la prédiction"):
#         try:
#             response = requests.post(API_URL, json=input_data)
#             result = response.json()
#             st.write("Réponse brute de l'API :", result)

#             st.success(" Prédiction réussie !")
#             st.markdown(f"###  Décision : **{result['decision']}**")
#             st.metric(" Probabilité d'approbation", f"{result['probability_approved']:.2%}")
#             st.metric(" Probabilité de refus", f"{result['probability_refused']:.2%}")

#             # --- 3. EXPLICATION SHAP ---
#             st.header(" 3. Explication SHAP")
            
#             st.write("Étapes du pipeline :", model.named_steps)

#             # Préparation de l’explainer et des valeurs SHAP
#             explainer = shap.Explainer(model)
#             shap_values = explainer(input_df)

#             # Graphique SHAP pour le client
#             st.subheader("Impact des variables sur la prédiction")
#             fig = plt.figure()
#             shap.plots.waterfall(shap_values[0], show=False)
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f" Erreur lors de la prédiction : {e}")
# else:
#     st.info("Veuillez charger un fichier CSV pour commencer.")

import streamlit as st
import pandas as pd
import requests
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
from home_credit.config import MODELS_DIR

# Configuration page
st.set_page_config(page_title="Credit Scoring", layout="wide")
st.title("📊 Credit Scoring Dashboard")

# Charger le modèle localement pour SHAP
model = joblib.load(MODELS_DIR / "best_model.joblib")

# 🔗 URL de l'API (en local)
API_URL = "http://localhost:8000/predict"

# --- 1. UPLOAD DES DONNÉES ---
st.header("📤 1. Upload des données client")

uploaded_file = st.file_uploader("Chargez un fichier CSV avec les données d'un client", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Affiche le DataFrame brut
    st.subheader("Aperçu des données chargées :")
    st.dataframe(input_df)

    # --- 2. PRÉDICTION ---
    st.header("🎯 2. Prédiction du score")

    # Convertir la première ligne en dictionnaire pour l'API
    input_data = input_df.iloc[0].to_dict()

    if st.button("Faire la prédiction"):
        try:
            response = requests.post(API_URL, json=input_data)
            result = response.json()
            st.write("Réponse brute de l'API :", result)

            st.success("✅ Prédiction réussie !")
            st.markdown(f"### 🏆 Décision : **{result['decision']}**")
            st.metric("✅ Probabilité d'approbation", f"{result['probability_approved']:.2%}")
            st.metric("❌ Probabilité de refus", f"{result['probability_refused']:.2%}")

            # --- 3. EXPLICATION SHAP ---
            st.header("🔍 3. Explication SHAP")
            
            st.write("Étapes du pipeline :", list(model.named_steps.keys()))

            try:
                # SOLUTION 1: Utiliser le modèle final après preprocessing
                st.info("🔄 Tentative avec le modèle final du pipeline...")
                
                # Identifier les étapes du pipeline
                pipeline_steps = list(model.named_steps.keys())
                st.write(f"Étapes disponibles: {pipeline_steps}")
                
                # Utiliser directement les noms connus du pipeline
                preprocessor_key = 'preprocessing'
                model_key = 'model'
                
                st.write(f"Preprocessor: {preprocessor_key}, Modèle: {model_key}")
                
                # Transformer les données avec l'étape preprocessing
                preprocessor = model.named_steps['preprocessing']
                transformed_data = preprocessor.transform(input_df)
                
                # Obtenir le modèle final
                final_model = model.named_steps['model']
                    
                # Créer l'explainer avec le modèle final et les données transformées
                explainer = shap.Explainer(final_model)
                shap_values = explainer(transformed_data)
                
                # Graphique SHAP
                st.subheader("Impact des variables sur la prédiction")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
                plt.close()
                    
            except Exception as shap_error:
                st.warning(f"⚠️ Erreur avec SHAP sur le pipeline: {shap_error}")
                
                try:
                    # SOLUTION 2: TreeExplainer pour les modèles basés sur les arbres
                    st.info("🌳 Tentative avec TreeExplainer...")
                    
                    # Identifier le modèle final directement
                    final_model = model.named_steps['model']
                    
                    if final_model and any(x in str(type(final_model)).lower() for x in ['lgb', 'xgb', 'forest', 'tree']):
                        # Transformer les données avec preprocessing seulement
                        transformed_data = model.named_steps['preprocessing'].transform(input_df)
                        
                        explainer = shap.TreeExplainer(final_model)
                        shap_values = explainer.shap_values(transformed_data)
                        
                        # Pour la classification binaire, prendre les valeurs pour la classe positive
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Classe positive
                        
                        st.subheader("Impact des variables sur la prédiction (TreeExplainer)")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.summary_plot(shap_values, transformed_data, plot_type="bar", show=False)
                        st.pyplot(fig)
                        plt.close()
                        
                    else:
                        raise Exception("Modèle non compatible avec TreeExplainer")
                        
                except Exception as tree_error:
                    st.warning(f"⚠️ TreeExplainer échoué: {tree_error}")
                    
                    try:
                        # SOLUTION 3: KernelExplainer (plus lent mais universel)
                        st.info("🔄 Tentative avec KernelExplainer...")
                        
                        # Créer un échantillon de background (première ligne)
                        background = input_df.iloc[:1]
                        
                        # Fonction de prédiction
                        def predict_fn(X):
                            if isinstance(X, np.ndarray):
                                X = pd.DataFrame(X, columns=input_df.columns)
                            return model.predict_proba(X)[:, 1]  # Probabilité classe positive
                        
                        explainer = shap.KernelExplainer(predict_fn, background)
                        shap_values = explainer.shap_values(input_df.iloc[:1])
                        
                        st.subheader("Impact des variables sur la prédiction (KernelExplainer)")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=input_df.iloc[0].values,
                                feature_names=input_df.columns.tolist()
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as kernel_error:
                        st.error(f"❌ KernelExplainer échoué: {kernel_error}")
                        
                        # SOLUTION 4: Feature Importance (Fallback final)
                        st.info("📊 Affichage de l'importance des features (fallback)")
                        
                        try:
                            # Trouver le modèle avec feature_importances_
                            final_model = model.named_steps['model']
                            
                            if hasattr(final_model, 'feature_importances_'):
                                importances = final_model.feature_importances_
                                
                                # Obtenir les noms des features après preprocessing
                                try:
                                    feature_names = model.named_steps['preprocessing'].get_feature_names_out()
                                except:
                                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                                
                                # Créer le graphique
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names[:len(importances)],
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                st.subheader("Top 15 Features les plus importantes")
                                fig, ax = plt.subplots(figsize=(12, 8))
                                top_15 = importance_df.head(15)
                                ax.barh(range(len(top_15)), top_15['Importance'])
                                ax.set_yticks(range(len(top_15)))
                                ax.set_yticklabels(top_15['Feature'])
                                ax.set_xlabel('Importance')
                                ax.set_title('Feature Importance')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                # Tableau des importances
                                st.dataframe(importance_df.head(20))
                                
                            else:
                                st.error("Impossible d'extraire l'importance des features du modèle")
                                
                        except Exception as importance_error:
                            st.error(f"Erreur lors du calcul de l'importance: {importance_error}")

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
            st.write("Détails de l'erreur:", str(e))
else:
    st.info("ℹ️ Veuillez charger un fichier CSV pour commencer.")

# --- SECTION DEBUG ---
with st.expander("🔧 Informations de debug"):
    st.write("**Type du modèle:**", type(model))
    if hasattr(model, 'named_steps'):
        st.write("**Étapes du pipeline:**")
        for step_name, step_obj in model.named_steps.items():
            st.write(f"- {step_name}: {type(step_obj)}")
    
    st.markdown("""
    ### Solutions pour l'erreur SHAP:
    1. **TreeExplainer**: Pour LightGBM, XGBoost, RandomForest
    2. **KernelExplainer**: Solution universelle mais plus lente
    3. **Feature Importance**: Fallback simple et rapide
    
    ### Versions recommandées:
    ```bash
    pip install shap==0.42.1 scikit-learn==1.3.0
    ```
    """)