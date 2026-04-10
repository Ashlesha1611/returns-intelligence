import pandas as pd
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer

def train_dashboard_nlp_tagger(data_path="../../datasets", output_path="results"):
    print("Training Dashboard NLP Tagger...")
    os.makedirs(output_path, exist_ok=True)
    
    returns = pd.read_csv(f"{data_path}/Returns.csv")
    skus = returns['sku_id'].unique()
    
    results = {}
    for sku in skus:
        sku_returns = returns[returns['sku_id'] == sku]
        
        # 1. Group by standard dropdown reasons
        raw_counts = dict(sku_returns['return_reason'].value_counts())
        reason_counts = {k: int(v) for k, v in raw_counts.items()}
        
        # 2. Extract pain-point tags from free text notes
        notes = sku_returns['return_note'].dropna().tolist()
        tags = []
        if len(" ".join(notes)) > 10:
            try:
                vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
                vectorizer.fit([" ".join(notes)])
                tags = list(vectorizer.get_feature_names_out())
            except Exception:
                pass
                
        results[str(sku)] = {
            "top_reasons": reason_counts,
            "pain_point_tags": tags
        }
        
    with open(f"{output_path}/top_return_reasons.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved NLP tags for {len(results)} SKUs to {output_path}/top_return_reasons.json")

if __name__ == "__main__":
    train_dashboard_nlp_tagger()
