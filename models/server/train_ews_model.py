import pandas as pd
import os
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder

def train_ews_model(data_path="../../datasets", output_path="results"):
    print("Training EWS LightGBM Regressor...")
    os.makedirs(output_path, exist_ok=True)
    
    products = pd.read_csv(f"{data_path}/Products.csv")
    scores = pd.read_csv(f"{data_path}/Product_scores.csv")
    
    latest_scores = scores.groupby('sku_id').last().reset_index()
    df = pd.merge(products, latest_scores, on='sku_id', how='inner')
    
    df['price'] = pd.to_numeric(df['price'] , errors='coerce').fillna(0)
    df['category'] = df['category'].fillna('Unknown')
    
    le = LabelEncoder()
    df['cat_encoded'] = le.fit_transform(df['category'])
    
    # Predict 30-day return rate based on product details and early sentiment proxy
    X = df[['price', 'cat_encoded']]
    y = df['return_rate'].fillna(df['return_rate'].mean())
    
    if len(X) > 0:
        model = lgb.LGBMRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, f"{output_path}/ews_lgbm_model.pkl")
        joblib.dump(le, f"{output_path}/category_label_encoder.pkl")
        
        print(f"Saved EWS LightGBM model and encoder to {output_path}/")
    else:
        print("Not enough data to train EWS.")

if __name__ == "__main__":
    train_ews_model()
