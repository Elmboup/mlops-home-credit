import pandas as pd
from home_credit.modeling.predict import predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def test_predict_function():
    # Données factices
    X = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]}, index=[100001, 100002])
    y = [0, 1]
    
    # Pipeline minimal
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=10, random_state=42, probability=True))
    ])
    
    pipe.fit(X, y)
    preds_df = predict(pipe, X)
    
    assert preds_df.shape[0] == 2
    assert "default_proba" in preds_df.columns
