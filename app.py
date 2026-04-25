import json
import os

import pandas as pd
import pickle
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
model = pickle.load(open("model/model.pkl", "rb"))

# ── Load dataset once for delay-rate look-ups ─────────────────────────────────
df_rates = pd.read_csv("data/Data_supply.csv")
shipping_delay_rate = df_rates.groupby("Shipping.Mode")["Late_delivery_risk"].mean()
region_delay_rate   = df_rates.groupby("Order.Region")["Late_delivery_risk"].mean()
market_delay_rate   = df_rates.groupby("Market")["Late_delivery_risk"].mean()

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

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Feature engineering
    df["quantity_per_day"] = df["Order.Item.Quantity"] - df["Days.for.shipping.scheduled"]
    df["profit_margin"]    = 0.25
    df["discount_amount"]  = df["Sales"] * 0.1

    # Delay-rate look-ups
    df["shipping_mode_delay_rate"] = df["Shipping.Mode"].map(shipping_delay_rate)
    df["region_delay_rate"]        = df["Order.Region"].map(region_delay_rate)
    df["market_delay_rate"]        = df["Market"].map(market_delay_rate)

    # Encode Type
    df["Type"] = df["Type"].astype("category").cat.codes

    df = df[features]

    prediction = model.predict(df)[0]
    result = "Late Delivery Expected" if prediction == 1 else "On Time Delivery"
    return jsonify({"prediction": result})


@app.route("/comparison", methods=["GET"])
def comparison():
    """Return real model comparison metrics produced by train_model.py."""
    results_path = "model/comparison_results.json"
    if not os.path.exists(results_path):
        return jsonify({"error": "comparison_results.json not found. Run train_model.py first."}), 404

    with open(results_path) as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
