"""Part D (segments): cluster the 24 hubs by how they really behave and check the tier labels.

Writes outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv,
segment_profile.csv and tier_vs_segment_mismatch.csv.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import save_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")

# 1. One row per hub, ascending hub_id (HUB01 first) - the order the hubs enter the scaler and KMeans.
profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "size"),
                mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"),
                mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .sort_index()
           .reset_index())
save_csv(profile, OUT / "hub_profile.csv")

# 2. Scale the four behaviour columns so no single unit (km vs hours vs a rate) dominates the distance.
PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[PROFILE_COLS])


def fit_kmeans(k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)


# 3. Separation evidence: mean distance between hubs in different clusters / in the same cluster.
pairs = list(combinations(range(len(scaled)), 2))           # 276 unordered hub pairs
pair_dist = np.array([np.linalg.norm(scaled[i] - scaled[j]) for i, j in pairs])
sep_rows = []
for k in [2, 3, 4, 5]:
    labels = fit_kmeans(k).labels_
    same = np.array([labels[i] == labels[j] for i, j in pairs])
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({"K": k, "across_over_within": pair_dist[~same].mean() / pair_dist[same].mean(),
                     "cluster_sizes": "-".join(str(s) for s in sizes)})
save_csv(pd.DataFrame(sep_rows), OUT / "part_d_separation.csv")

# 4. K = 3 (three playbooks). Name clusters by their hubs' mean delay rate, not by KMeans' arbitrary numbers.
profile["cluster"] = fit_kmeans(3).labels_
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
save_csv(segments, OUT / "hub_segments.csv")

seg_profile = (profile.groupby("segment")
               .agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                    mean_n_stops=("mean_n_stops", "mean"))
               .reindex(["Standard", "Watch", "Escalate"])
               .reset_index())
save_csv(seg_profile, OUT / "segment_profile.csv")

TIER_IMPLIES = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_IMPLIES)]
save_csv(mismatch, OUT / "tier_vs_segment_mismatch.csv")

print(pd.DataFrame(sep_rows).round(4).to_string(index=False))
print(seg_profile.round(4).to_string(index=False))
print("tier mismatches:\n", mismatch.to_string(index=False))
