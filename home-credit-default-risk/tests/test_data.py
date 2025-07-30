import pandas as pd
from home_credit.modeling.predict import predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# def test_code_is_tested():
#     assert False
