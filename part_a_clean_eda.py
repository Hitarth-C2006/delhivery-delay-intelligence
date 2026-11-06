"""Part A: turn the messy export into the modelling table and measure where delays come from.

Writes data/modelling_table.csv, outputs/part_a_checks.csv,
outputs/part_a_delay_by_group.csv and outputs/part_a_correlations.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import (FEATURES, WEIGHT_SENTINEL_ABOVE, build_features,
                               clean_shipments, save_csv)

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "raw_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")

# "Before" counts: measured after removing duplicates and before any filling.
dedup = raw.drop_duplicates()
sentinel_rows = int((dedup["parcel_weight_kg"] > WEIGHT_SENTINEL_ABOVE).sum())
spellings_before = int(dedup["service_type"].nunique())
spellings_after = int(dedup["service_type"].str.strip().str.title().nunique())

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv(DATA / "modelling_table.csv", index=False, lineterminator="\n")

# Spread of delivery hours, 1.5 x IQR rule on unrounded quartiles.
q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = int(((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum())

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", sentinel_rows),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", spellings_before),
    ("service_type_spellings_after", spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", float(table["is_delayed"].mean())),
    ("declared_value_mean_inr", float(table["declared_value_inr"].mean())),
    ("declared_value_median_inr", float(table["declared_value_inr"].median())),
    ("log_declared_value_mean", float(table["log_declared_value"].mean())),
    ("log_declared_value_median", float(table["log_declared_value"].median())),
    ("hours_q1", float(q1)),
    ("hours_q3", float(q3)),
    ("hours_lower_bound", float(lower)),
    ("hours_upper_bound", float(upper)),
    ("hours_outlier_count", outliers),
]
# object dtype keeps counts as integers (1224, not 1224.0) next to rounded decimals
values = pd.Series([round(v, 4) if isinstance(v, float) else v for _, v in checks], dtype=object)
save_csv(pd.DataFrame({"metric": [m for m, _ in checks], "value": values}), OUT / "part_a_checks.csv")

# Delay evidence by service type, hub tier and distance band.
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"],
                                right=True, include_lowest=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grouped = table.groupby(dim, observed=False).agg(
        shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for group, r in grouped.iterrows():
        rows.append({"dimension": dim, "group": str(group), "shipments": int(r["shipments"]),
                     "delayed": int(r["delayed"]), "delay_rate": r["delay_rate"],
                     "mean_hours": r["mean_hours"]})
save_csv(pd.DataFrame(rows), OUT / "part_a_delay_by_group.csv")

# Correlation matrix of the numeric columns (row labels kept as the first column).
corr = table.drop(columns=["distance_band"]).select_dtypes(include=np.number).corr()
corr.index.name = "column"
save_csv(corr, OUT / "part_a_correlations.csv", index=True)

print(f"modelling table: {len(table)} rows, {int(table['is_delayed'].sum())} delayed")
print("medians:", medians)
