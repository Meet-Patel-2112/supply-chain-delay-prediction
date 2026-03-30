import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("../data/Data_supply.csv")

drop_cols = [
    "X", "X.1", "X.2", "Unnamed: 62",
    "Customer.Email", "Customer.Password",
    "Product.Image", "Product.Description",
    "Customer.Id", "Product.Id", "Product.Card.Id"
]

df = df.drop(columns=drop_cols, errors="ignore")

features = [
    "Type",
    "Sales",
    "Order.Item.Quantity", 
    "Order.Item.Product.Price", 
    "Shipping.Mode", 
    "Shipping_Time", 
    "Order.Region", 
    "Market"
]

target = "Late_delivery_risk"

df = df.dropna(subset=["Late_delivery_risk"])

df["Late_delivery_risk"] = df["Late_delivery_risk"].astype(int)

X = df[features]
y = df["Late_delivery_risk"]

le = LabelEncoder()

for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = le.fit_transform(X[col])
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.dtypes)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, pred))

pickle.dump(model, open("../model/model.pkl", "wb"))