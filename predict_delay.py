"""Score a batch of new parcels and turn it into an action list for operations.

Combines Part C's classifier (models/) with Part D's hub segments (outputs/hub_segments.csv).
Writes outputs/new_shipment_actions.csv, one row per new parcel, in file order.
"""
from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments, save_csv

ROOT = Path(__file__).resolve().parent
DATA, MODELS, OUT = ROOT / "data", ROOT / "models", ROOT / "outputs"

new = pd.read_csv(DATA / "new_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")
model = joblib.load(MODELS / "delay_model.pkl")
threshold = float(pd.read_csv(MODELS / "delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
impute = pd.read_csv(MODELS / "impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
segments = pd.read_csv(OUT / "hub_segments.csv")[["hub_id", "segment"]]

print(f"new parcels: {len(new)} rows; missing distance_km {new['distance_km'].isna().sum()}, "
      f"parcel_weight_kg {new['parcel_weight_kg'].isna().sum()}, declared_value {new['declared_value'].isna().sum()}, "
      f"payment_mode {new['payment_mode'].isna().sum()}; {new['service_type'].nunique()} service_type spellings")

# Same cleaning and features as training, with the stored historical medians as fill values.
clean, _ = clean_shipments(new, medians=medians)
features = build_features(clean, hubs)

scored = features[["shipment_id", "hub_id"]].copy()
scored["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
scored["flagged"] = (scored["delay_probability"] >= threshold).astype(int)
scored = scored.merge(segments, on="hub_id", how="left")      # left merge keeps file order


def action(row):
    if row["flagged"] == 0:
        return "No action"
    return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"


scored["action"] = scored.apply(action, axis=1)
scored = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
save_csv(scored, OUT / "new_shipment_actions.csv")

print(f"threshold {threshold}: {int(scored['flagged'].sum())} of {len(scored)} flagged")
print(scored["action"].value_counts().to_string())
