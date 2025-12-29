import os, json, joblib, numpy as np, pandas as pd

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_oof(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_oof(path):
    import numpy as np
    return np.load(path)
