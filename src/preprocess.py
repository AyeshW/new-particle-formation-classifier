import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def basic_preprocess(df, drop_cols=["id","date","class4","class2"]):
    X = df.copy()
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(c, axis=1)
    return X

def fit_imputer_scaler(X_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler

def transform_imputer_scaler(X, imputer, scaler):
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled
