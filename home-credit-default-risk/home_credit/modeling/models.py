from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from home_credit.config import RANDOM_SEED


def get_models(random_state=RANDOM_SEED):
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        "LightGBM": LGBMClassifier(is_unbalance=True, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state)
    }

def get_metrics():
    return {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "precision": precision_score,
        "recall": recall_score
    } 