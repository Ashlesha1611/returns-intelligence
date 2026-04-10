import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def train_profile_embedder(data_path="../../datasets", output_path="results"):
    print("Training Profile Embedder...")
    os.makedirs(output_path, exist_ok=True)
    
    customers_df = pd.read_csv(f"{data_path}/Customers.csv")
    
    # Combine features into a single profile string
    customers_df['profile_text'] = (
        customers_df['lifestyle_tags'].fillna('') + " " +
        customers_df['preferred_styles'].fillna('') + " " +
        customers_df['Preferred_colors'].fillna('')
    ).str.replace("|", " ")
    
    # TF-IDF Vectorizer to exact 50 max features to match legacy dimensionality
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    embeddings = vectorizer.fit_transform(customers_df['profile_text']).toarray()
    
    # Format embeddings identically to old raw data
    results = []
    for i, row in customers_df.iterrows():
        vec_str = ",".join([f"{x:.6f}" for x in embeddings[i]])
        results.append({
            "customer_id": row["customer_id"],
            "embedding_vector": vec_str,
            "last_updated": pd.Timestamp.now()
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(f"{output_path}/User_Profile_Embeddings.csv", index=False)
    print(f"Saved {len(out_df)} profile embeddings to {output_path}/User_Profile_Embeddings.csv")

if __name__ == "__main__":
    train_profile_embedder()
