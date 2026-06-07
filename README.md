Supply Chain Delay Prediction
Overview

This project predicts whether an order in a supply chain will be delivered on time or late using machine learning.

It combines data analysis, feature engineering, model comparison, and a Flask web application to provide real-time predictions.

The system trains multiple models, selects the best performing one, and exposes it through a web interface.

Project Structure
supply-chain-delay-prediction
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│   └── Data_supply.csv
│
├── model
│   ├── train_model.py
│   ├── model.pkl
│   ├── encoders.pkl
│
├── templates
│   └── index.html
│
├── notebooks
│   └── eda.ipynb
│
└── reports
    └── feature importance visualizations
Dataset

The dataset contains historical supply chain order information.

Key attributes include:

Order Type
Sales
Quantity
Shipping Mode
Order Region
Market
Scheduled Shipping Days

Target variable: 

Late_delivery_risk
0 → On-time delivery
1 → Late delivery
Feature Engineering

Additional features are generated to improve prediction performance.

Quantity per day
Profit margin
Discount amount
Shipping mode delay rate
Region delay rate
Market delay rate

These features capture operational patterns that influence delivery delays.

Machine Learning Models

The training pipeline evaluates multiple models:

Random Forest
Logistic Regression
XGBoost
LightGBM
CatBoost

Each model is evaluated using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC

The best model is automatically selected and saved.

Training the Model

Run:

python model/train_model.py

This will:

Train multiple classifiers
Evaluate their performance
Save comparison metrics
Store the best model as model.pkl
Running the Application

Install dependencies:

pip install -r requirements.txt

Run the Flask server:

python app.py

Open:

http://localhost:5000
API Endpoint
Predict Delivery Delay

POST /predict

Example request:

{
  "Type": "Consumer",
  "Sales": 250,
  "Order.Item.Quantity": 5,
  "Days.for.shipping.scheduled": 3,
  "Shipping.Mode": "Standard Class",
  "Order.Region": "West",
  "Market": "US"
}

Response:

{
  "prediction": "Late Delivery Expected"
}
Results

The model learns patterns in historical supply chain data to predict delivery risks. Feature engineering and ensemble models significantly improve prediction accuracy.