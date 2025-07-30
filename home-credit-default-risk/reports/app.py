import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Scoring", layout="centered")
st.title("🏦 Home Credit Default Risk Scoring")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

uploaded_file = st.file_uploader("📁 Upload your customer file (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col="SK_ID_CURR")
    preds = model.predict_proba(df)[:, 1]
    df["Default Probability"] = preds
    st.success("Scoring terminé !")
    st.dataframe(df[["Default Probability"]])
    st.download_button("📥 Télécharger les résultats", df.to_csv(index=False), file_name="scoring_results.csv")
