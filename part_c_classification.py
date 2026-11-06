"""Part C: will this parcel miss its promise? Classifier, noise check, tuning, alert threshold.

Writes outputs/part_c_models.csv, part_c_feature_scores.csv, part_c_selectk.csv,
part_c_tuning.csv, part_c_thresholds.csv, part_c_final_test.csv and part_c_final_config.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS, save_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# 1. Same two-step split as Part B, stratified on the target so every split keeps ~23% delayed.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"split sizes {len(y_train)}/{len(y_val)}/{len(y_test)}, "
      f"delayed {int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())}")


def report(y_true, y_pred):
    """Confusion-matrix cells plus the four scores, all computed from the same predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


# 2. Blind baseline and three classifiers at the default 0.5 rule, scored on validation.
model_rows = [{"model": "blind_all_on_time", **report(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **report(y_val, logreg.predict(X_val))})

best_tree, best_tree_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f1 = f1_score(y_val, tree.predict(X_val))
    if f1 > best_tree_f1:            # strict ">" keeps the smaller depth on ties
        best_tree, best_tree_f1 = d, f1
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_tree, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_tree}", **report(y_val, tree.predict(X_val))})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **report(y_val, forest.predict(X_val))})
save_csv(pd.DataFrame(model_rows), OUT / "part_c_models.csv")

# 3. Which columns carry signal? Forest importance and ANOVA F-score, both from training rows.
f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
feature_scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                               "rf_importance": forest.feature_importances_})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="first").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
save_csv(feature_scores.sort_values("f_rank"), OUT / "part_c_feature_scores.csv")
noise_share = feature_scores.loc[feature_scores["is_noise"], "rf_importance"].sum()
print(f"noise columns' share of forest importance: {noise_share:.4f}")

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
save_csv(pd.DataFrame(selectk_rows), OUT / "part_c_selectk.csv")

# 4. Tuning on the 7 best columns. Selector fitted on training rows only; test rows untouched.
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
X_train7, X_val7 = selector.transform(X_train), selector.transform(X_val)
print("7 selected columns:", list(np.array(FEATURES)[selector.get_support()]))

grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(X_train7, y_train)
rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                                 random_state=42).fit(X_train7, y_train)
default_7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train7, y_train)

PARAM_KEYS = ["n_estimators", "max_depth", "min_samples_leaf"]


def describe(params):
    return "; ".join(f"{k}={params[k]}" for k in PARAM_KEYS)


tuning = []
for name, search in [("grid_search", grid_search), ("randomized_search", rand_search)]:
    n_cand = len(search.cv_results_["params"])
    tuning.append({"search": name, "candidates": n_cand, "cv_folds": 5, "total_fits": n_cand * 5,
                   "best_params": describe(search.best_params_),
                   "best_cv_accuracy": round(search.best_score_, 4),
                   "val_f1": f1_score(y_val, search.best_estimator_.predict(X_val7)),
                   "_params": search.best_params_})
default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
tuning.append({"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
               "best_params": describe(default_params), "best_cv_accuracy": None,
               "val_f1": f1_score(y_val, default_7.predict(X_val7)), "_params": default_params})
tuning_df = pd.DataFrame(tuning)
save_csv(tuning_df.drop(columns="_params"), OUT / "part_c_tuning.csv")

# 5. Final configuration: highest validation F1; ties -> default, then grid, then randomized.
tie_order = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning_df["_tie"] = tuning_df["search"].map(tie_order)
winner = tuning_df.sort_values(["val_f1", "_tie"], ascending=[False, True]).iloc[0]
chosen = winner["_params"]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **chosen))]).fit(X_train, y_train)
print("final configuration chosen from:", winner["search"], chosen)

# 6. Alert threshold: highest t whose validation recall is still >= 0.80.
val_proba = final_model.predict_proba(X_val)[:, 1]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = [{"threshold": t, **report(y_val, (val_proba >= t).astype(int))} for t in THRESHOLDS]
threshold_df = pd.DataFrame(threshold_rows)
save_csv(threshold_df, OUT / "part_c_thresholds.csv")
meets_rule = threshold_df.loc[threshold_df["recall"] >= 0.80, "threshold"]
alert_threshold = float(meets_rule.max()) if len(meets_rule) else 0.1
print("alert threshold:", alert_threshold)

# 7. The one and only test evaluation, at the chosen threshold.
test_pred = (final_model.predict_proba(X_test)[:, 1] >= alert_threshold).astype(int)
save_csv(pd.DataFrame([{"threshold": alert_threshold, **report(y_test, test_pred)}]),
         OUT / "part_c_final_test.csv")
save_csv(pd.DataFrame([{
    "selector_k": K, "n_estimators": chosen["n_estimators"],
    "max_depth": "unlimited" if chosen["max_depth"] is None else chosen["max_depth"],
    "min_samples_leaf": chosen["min_samples_leaf"], "threshold": alert_threshold,
    "chosen_from": winner["search"],
}]), OUT / "part_c_final_config.csv")
