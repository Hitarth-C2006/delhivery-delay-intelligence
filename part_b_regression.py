"""Part B: predict delivery_hours and decide which regression model to trust.

Writes outputs/part_b_checks.csv, part_b_tree_depth.csv, part_b_metrics.csv,
part_b_gb_learning_rate.csv and part_b_cv.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES, save_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# 1. Split first: 70% train, then the remaining 30% halved into validation and test.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Scaler fitted on the 840 training rows only, then applied to the test rows.
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
train_max_abs_mean = float(np.abs(train_scaled.mean(axis=0)).max())   # ~1e-16, rounds to 0.0
test_max_abs_mean = float(np.abs(test_scaled.mean(axis=0)).max())     # clearly non-zero
checks = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    # object dtype keeps the row counts as integers; decimals rounded here to 4 places
    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                        round(train_max_abs_mean, 4), round(test_max_abs_mean, 4)], dtype=object),
})
save_csv(checks, OUT / "part_b_checks.csv")


def scores(name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# 3. Baseline: predict the training mean for every validation parcel.
metrics = [scores("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

# 4. Tree depth diagnostic: train vs validation error as the tree is allowed to grow.
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": rmse(y_train, tree.predict(X_train)),
                       "val_rmse": rmse(y_val, tree.predict(X_val)), "_depth": d})
depth_df = pd.DataFrame(depth_rows)
best_depth = depth_df.loc[depth_df["val_rmse"].idxmin(), "_depth"]   # idxmin keeps the first (smaller) depth on ties
best_depth = None if pd.isna(best_depth) else int(best_depth)
save_csv(depth_df.drop(columns="_depth"), OUT / "part_b_tree_depth.csv")

# 5. Four candidates, fitted on train and scored on validation.
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    row = scores(name, "val", y_val, model.predict(X_val))
    val_rmse[name] = row["RMSE"]
    metrics.append(row)

# 6. Learning rate experiment for gradient boosting.
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
save_csv(pd.DataFrame(lr_rows), OUT / "part_b_gb_learning_rate.csv")

# 7. 5-fold CV for the two best on validation, one shared KFold object, on train + validation.
ranking = sorted(val_rmse, key=val_rmse.get)
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranking[:2]:
    r2s = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": r2s.mean(), "std_r2": r2s.std()})
save_csv(pd.DataFrame(cv_rows), OUT / "part_b_cv.csv")

# Final model = lowest validation RMSE, already fitted on train. Test rows used once, here.
final_name = ranking[0]
metrics.append(scores(final_name, "test", y_test, candidates[final_name].predict(X_test)))
save_csv(pd.DataFrame(metrics), OUT / "part_b_metrics.csv")

print(f"scaled column means: train max |mean| = {train_max_abs_mean:.2e}, test max |mean| = {test_max_abs_mean:.4f}")
print("validation RMSE ranking:", [(n, round(val_rmse[n], 3)) for n in ranking])
print("best tree depth:", "unlimited" if best_depth is None else best_depth)
print("final model:", final_name)
