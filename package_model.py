"""Part D (packaging): rebuild Part C's final classifier and save it as a reusable artefact.

Writes models/delay_model.pkl, models/delay_threshold.csv, models/feature_order.csv,
models/impute_values.csv and outputs/part_d_reload_check.csv.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments, save_csv

ROOT = Path(__file__).resolve().parent
MODELS, OUT = ROOT / "models", ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

config = pd.read_csv(OUT / "part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
threshold = float(config["threshold"])

# Same stratified split as Part C, so the model sees exactly the same 840 training rows.
table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, n_estimators=int(config["n_estimators"]),
                                     max_depth=max_depth, min_samples_leaf=int(config["min_samples_leaf"]))),
]).fit(X_train, y_train)

# The fill-in medians come from the same clean_shipments call that built the modelling table.
_, medians = clean_shipments(pd.read_csv(ROOT / "data" / "raw_shipments.csv"))

joblib.dump(model, MODELS / "delay_model.pkl")
save_csv(pd.DataFrame({"threshold": [threshold]}), MODELS / "delay_threshold.csv")
save_csv(pd.DataFrame({"feature": FEATURES}), MODELS / "feature_order.csv")
save_csv(pd.DataFrame({"column": list(medians), "median": list(medians.values())}), MODELS / "impute_values.csv")

# Reload the pickle and prove it scores the test rows exactly like the in-memory model.
reloaded = joblib.load(MODELS / "delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
save_csv(pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
                       "value": pd.Series([len(X_test), round(float(diff), 4), threshold], dtype=object)}),
         OUT / "part_d_reload_check.csv")
print(f"saved models/delay_model.pkl; reload max |diff| on {len(X_test)} test rows = {diff}")
