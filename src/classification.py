from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

from sklearn.metrics import accuracy_score, classification_report

def train_classification(df):

    features = [
        'age','annual_income','spending_score',
        'num_purchases','website_visits_per_month',
        'cart_abandon_rate','total_spend','engagement_score'
    ]

    X = df[features]
    y = df['churned']

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 🔥 Predictions
    y_pred = model.predict(X_test)

    # 🔥 Metrics
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/classification.pkl")
    print("✅ Classification model saved")
    
    return acc 