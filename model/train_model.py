"""
train_model.py  –  Supply Chain Delay Prediction
Trains 5 classifiers, saves comparison metrics to model/comparison_results.json,
and saves the best model (XGBoost) as model/model.pkl.
"""

import json
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed – skipping XGBClassifier")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed – skipping LGBMClassifier")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] catboost not installed – skipping CatBoostClassifier")

# ── Data loading ──────────────────────────────────────────────────────────────
df = pd.read_csv("../data/Data_supply.csv")

# ── Feature engineering ───────────────────────────────────────────────────────
df["quantity_per_day"] = df["Order.Item.Quantity"] - df["Days.for.shipping.scheduled"]
df["profit_margin"] = df["Order.Profit.Per.Order"] / df["Sales"]
df["discount_amount"] = df["Sales"] * df["Order.Item.Discount.Rate"]

shipping_delay_rate = df.groupby("Shipping.Mode")["Late_delivery_risk"].mean()
df["shipping_mode_delay_rate"] = df["Shipping.Mode"].map(shipping_delay_rate)

region_delay_rate = df.groupby("Order.Region")["Late_delivery_risk"].mean()
df["region_delay_rate"] = df["Order.Region"].map(region_delay_rate)

market_delay_rate = df.groupby("Market")["Late_delivery_risk"].mean()
df["market_delay_rate"] = df["Market"].map(market_delay_rate)

# ── Feature selection ─────────────────────────────────────────────────────────
features = [
    "Type",
    "Sales",
    "Order.Item.Quantity",
    "quantity_per_day",
    "profit_margin",
    "discount_amount",
    "Days.for.shipping.scheduled",
    "shipping_mode_delay_rate",
    "region_delay_rate",
    "market_delay_rate",
]
target = "Late_delivery_risk"

df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

X = df[features].copy()
y = df[target]
X = X.fillna(0)

le = LabelEncoder()
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = le.fit_transform(X[col])

# Save encoder so app.py can reuse it
with open("../model/encoders.pkl", "wb") as f:
    pickle.dump(le, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Class imbalance ratio (used by XGBoost)
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# ── Model registry ────────────────────────────────────────────────────────────
model_registry = [
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10, random_state=42
        ),
    ),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
    ),
]

if HAS_XGB:
    model_registry.append((
        "XGBoost",
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    ))

if HAS_LGB:
    model_registry.append((
        "LightGBM",
        LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    ))

if HAS_CAT:
    model_registry.append((
        "CatBoost",
        CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
    ))

# ── Training & evaluation loop ────────────────────────────────────────────────
results = []

for name, clf in model_registry:
    print(f"\n{'─'*50}")
    print(f"Training: {name}")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = (
        clf.predict_proba(X_test)[:, 1]
        if hasattr(clf, "predict_proba")
        else clf.decision_function(X_test)
    )

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    results.append({
        "name":      name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc":       round(auc,  4),
        "model_obj": clf,
    })

# ── Rank by AUC, save comparison JSON ─────────────────────────────────────────
results_sorted = sorted(results, key=lambda r: r["auc"], reverse=True)

comparison_output = [
    {
        "rank":      rank + 1,
        "name":      r["name"],
        "accuracy":  r["accuracy"],
        "precision": r["precision"],
        "recall":    r["recall"],
        "f1":        r["f1"],
        "auc":       r["auc"],
    }
    for rank, r in enumerate(results_sorted)
]

with open("../model/comparison_results.json", "w") as f:
    json.dump(comparison_output, f, indent=2)

print("\n\n Comparison results saved to model/comparison_results.json")
print(json.dumps(comparison_output, indent=2))

# ── Persist the best model ─────────────────────────────────────────────────────
best = results_sorted[0]
print(f"\n Best model: {best['name']}  (AUC {best['auc']})")

with open("../model/model.pkl", "wb") as f:
    pickle.dump(best["model_obj"], f)
print(" Best model saved to model/model.pkl")

# ── Feature importance plot (if the best model supports it) ───────────────────
best_clf = best["model_obj"]
if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(X.columns, importances)
    plt.title(f"Feature Importance – {best['name']}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("../reports/feature_importance_v4.png")
    plt.show()
    print(" Feature importance plot saved to reports/feature_importance_v4.png")
