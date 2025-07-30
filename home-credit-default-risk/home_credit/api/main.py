# API Gningue
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import numpy as np
from home_credit.config import MODELS_DIR

app = FastAPI(title="Credit Scoring API", description="API for predicting credit approval using best_model.joblib")

# Définir le modèle Pydantic pour les données d'entrée
class ClientData(BaseModel):
    SK_ID_CURR: float
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    OWN_CAR_AGE: Optional[float] = None
    FLAG_MOBIL: float
    FLAG_EMP_PHONE: float
    FLAG_WORK_PHONE: float
    FLAG_CONT_MOBILE: float
    FLAG_PHONE: float
    FLAG_EMAIL: float
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: float
    REGION_RATING_CLIENT_W_CITY: float
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: float
    REG_REGION_NOT_LIVE_REGION: float
    REG_REGION_NOT_WORK_REGION: float
    LIVE_REGION_NOT_WORK_REGION: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    LIVE_CITY_NOT_WORK_CITY: float
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: float
    EXT_SOURCE_3: Optional[float] = None
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    TOTALAREA_MODE: Optional[float] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[str] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    FLAG_DOCUMENT_2: float
    FLAG_DOCUMENT_3: float
    FLAG_DOCUMENT_4: float
    FLAG_DOCUMENT_5: float
    FLAG_DOCUMENT_6: float
    FLAG_DOCUMENT_7: float
    FLAG_DOCUMENT_8: float
    FLAG_DOCUMENT_9: float
    FLAG_DOCUMENT_10: float
    FLAG_DOCUMENT_11: float
    FLAG_DOCUMENT_12: float
    FLAG_DOCUMENT_13: float
    FLAG_DOCUMENT_14: float
    FLAG_DOCUMENT_15: float
    FLAG_DOCUMENT_16: float
    FLAG_DOCUMENT_17: float
    FLAG_DOCUMENT_18: float
    FLAG_DOCUMENT_19: float
    FLAG_DOCUMENT_20: float
    FLAG_DOCUMENT_21: float
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None
    credit_active_unique: float
    total_credit_sum: float
    prev_app_count: float
    avg_amt_app: float
    total_amt_credit: float
    pos_cash_count: float
    avg_months_balance: float
    installments_count: float
    total_amt_payment: float
    avg_days_entry_payment: float

# Charger le modèle
model = joblib.load(MODELS_DIR / "best_model.joblib")

@app.post("/predict")
async def predict(data: ClientData):
    try:
        # Convertir les données Pydantic en dictionnaire
        data_dict = data.dict()

        # Créer un DataFrame avec les 115 colonnes dans l'ordre attendu
        input_df = pd.DataFrame([data_dict])

        # Faire la prédiction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Interpréter la prédiction
        decision = "Approved" if prediction == 0 else "Refused"
        proba_approved = float(probabilities[0])  # Probabilité pour TARGET=0
        proba_refused = float(probabilities[1])   # Probabilité pour TARGET=1

        # Retourner la réponse
        return {
            "decision": decision,
            "probability_approved": proba_approved,
            "probability_refused": proba_refused
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Credit Scoring API. Use /predict endpoint to make predictions."}