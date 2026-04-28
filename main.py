import pandas as pd
import json

from src.preprocessing import preprocess_data
from src.clustering import train_clustering
from src.classification import train_classification
from src.regression import train_regression

print("📊 Loading dataset...")
df = pd.read_csv("/Users/apple/Downloads/Smart Retail Intelligence system (SRIS)/data/Customer_Transactions.csv")

print("🧹 Preprocessing...")
df = preprocess_data(df)

print("🔵 Clustering...")
sil_score = train_clustering(df)

print("🟢 Classification...")
acc = train_classification(df)

print("🟣 Regression...")
mae, rmse, r2 = train_regression(df)

# 🔥 SAVE METRICS
metrics = {
    "classification": {"accuracy": acc},
    "regression": {"mae": mae, "rmse": rmse, "r2": r2},
    "clustering": {"silhouette": sil_score}
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("🎉 ALL MODELS + METRICS SAVED!")