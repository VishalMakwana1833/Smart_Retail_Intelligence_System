import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df = df.drop_duplicates()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['feedback_text'] = df['feedback_text'].fillna("No Feedback")

    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['country'] = le.fit_transform(df['country'])

    df['total_spend'] = df['num_purchases'] * df['avg_purchase_value']
    df['engagement_score'] = (
        df['website_visits_per_month'] * 0.7 +
        (1 - df['cart_abandon_rate']) * 0.3
    )

    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    df['recency_days'] = (pd.to_datetime("today") - df['last_purchase_date']).dt.days

    # 🔥 Drop original date column
    df.drop('last_purchase_date', axis=1, inplace=True)

    return df