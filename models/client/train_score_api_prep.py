import pandas as pd
import os
import json

def train_score_api_prep(data_path="../../datasets", output_path="results"):
    print("Training Score API Prep...")
    os.makedirs(output_path, exist_ok=True)
    
    scores_df = pd.read_csv(f"{data_path}/Product_scores.csv")
    
    # Sort strictly by date to extract the chronological last row
    latest_scores = scores_df.sort_values(by='date').groupby('sku_id').last().reset_index()
    
    results = {}
    for _, row in latest_scores.iterrows():
        results[row['sku_id']] = {
            "quality_score": float(row.get('quality_score', 0)),
            "value_score": float(row.get('Value_score', 0)),
            "sentiment_score": float(row.get('sentiment_score', 0)),
            "return_rate": float(row.get('return_rate', 0)),
            "last_updated_date": str(row.get('date', ''))
        }
        
    with open(f"{output_path}/latest_product_scores.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved latest product scores for {len(results)} SKUs to {output_path}/latest_product_scores.json")

if __name__ == "__main__":
    train_score_api_prep()
