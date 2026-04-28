from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.metrics import silhouette_score

def train_clustering(df):

    X = df[['age','annual_income','spending_score','total_spend','engagement_score']]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)

    # 🔥 Silhouette Score
    score = silhouette_score(X_scaled, model.labels_)
    print(f"✅ Silhouette Score: {score:.2f}")

    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/clustering.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    return score  