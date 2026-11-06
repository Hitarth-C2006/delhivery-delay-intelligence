"""Shared constants and the two preparation functions used by every Part.

clean_shipments() fixes the messy export, build_features() adds the model inputs.
Both are imported by the training scripts and by predict_delay.py, so a parcel
is prepared the same way at training time and at scoring time.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_ABOVE = 100   # network max is 40 kg, so anything above 100 is a scanner fault
RUPEE = "₹"


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    Pass medians=None on the historical table to compute them; pass the stored
    dictionary when cleaning new parcels so they get the same fill values.
    """
    df = raw.drop_duplicates().reset_index(drop=True)                     # 1. exact duplicates

    df["service_type"] = df["service_type"].str.strip().str.title()         # 2. one spelling each

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")              # 3. blank -> Unknown

    text = df["declared_value"].astype("object")                           # 4. "₹1,290" -> 1290
    text = text.str.replace(RUPEE, "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(text, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])              # 5. text -> datetime

    sentinel = df["parcel_weight_kg"] > WEIGHT_SENTINEL_ABOVE              # 6. 999 kg -> missing
    df.loc[sentinel, "parcel_weight_kg"] = np.nan

    if medians is None:                                                    # 7. median fill
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive the model inputs. Never touches delivery_hours."""
    hub_cols = hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]]
    df = clean.merge(hub_cols, on="hub_id", how="left")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df


def save_csv(df, path, index=False):
    """Write a result file: decimals rounded to 4 places, LF line endings on every OS."""
    df.round(4).to_csv(path, index=index, lineterminator="\n")
