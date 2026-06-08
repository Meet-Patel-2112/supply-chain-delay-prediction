import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("../data/Data_supply.csv")

df["quantity_per_day"] = df["Order.Item.Quantity"] - df["Days.for.shipping.scheduled"]
df["profit_margin"] = df["Order.Profit.Per.Order"] / df["Sales"]
df["discount_amount"] = df["Sales"] * df["Order.Item.Discount.Rate"]

shipping_delay_rate = df.groupby("Shipping.Mode")["Late_delivery_risk"].mean()
df["shipping_mode_delay_rate"] = df["Shipping.Mode"].map(shipping_delay_rate)

region_delay_rate = df.groupby("Order.Region")["Late_delivery_risk"].mean()
df["region_delay_rate"] = df["Order.Region"].map(region_delay_rate)

market_delay_rate = df.groupby("Market")["Late_delivery_risk"].mean()
df["market_delay_rate"] = df["Market"].map(market_delay_rate)

drop_cols = [
    "X","X.1","X.2","Unnamed: 62",
    "Customer.Email","Customer.Password",
    "Product.Image","Product.Description",
    "Customer.Id","Product.Id","Product.Card.Id"
]

df = df.drop(columns=drop_cols, errors="ignore")

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
    "market_delay_rate"
]

target = "Late_delivery_risk"

df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

X = df[features]
y = df[target]
X = X.fillna(0)

# store encoders
encoders = {}

for col in X.select_dtypes(include=["object","string"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(classification_report(y_test, test_pred))
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

# save model + encoders
pickle.dump(model, open("../model/model.pkl","wb"))
pickle.dump(encoders, open("../model/encoders.pkl","wb"))

importances = model.feature_importances_

plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../reports/feature_importance_v3.png")