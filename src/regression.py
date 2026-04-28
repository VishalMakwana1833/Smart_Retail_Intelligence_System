from sklearn.linear_model import LinearRegression
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_regression(df):

    X = df[['annual_income','num_purchases','engagement_score']]
    y = df['total_spend']

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 🔥 Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ RMSE: {rmse:.2f}")
    print(f"✅ R2 Score: {r2:.2f}")

    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/regression.pkl")
    
    return mae, rmse, r2 