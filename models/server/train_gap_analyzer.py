import pandas as pd
import os
import json

def train_gap_analyzer(data_path="../../datasets", output_path="results"):
    print("Training Gap Analyzer...")
    os.makedirs(output_path, exist_ok=True)
    
    products = pd.read_csv(f"{data_path}/Products.csv")
    
    gaps = []
    for _, p in products.iterrows():
        desc = str(p['description_text']).lower()
        missing = []
        
        # Enhanced heuristic gap logic
        if p['category'] == 'Furniture' and 'dimension' not in desc and 'inch' not in desc:
            missing.append('Missing Dimensions in Copy')
        if pd.isna(p['material']) or 'material' not in desc:
            missing.append('Missing Material Details')
        if pd.isna(p['color']) and 'color' not in desc:
            missing.append('Color Info Missing')
        if 'care' not in desc and 'clean' not in desc:
            missing.append('Lacks Care Instructions')
            
        score = 100 - (len(missing) * 20)
        
        if len(missing) > 0:
            gaps.append({
                "sku_id": p['sku_id'],
                "product_name": p['product_name'],
                "missing_tags": missing,
                "listing_score": max(20, score)
            })
            
    with open(f"{output_path}/listing_gaps.json", "w") as f:
        json.dump(gaps, f, indent=4)
        
    print(f"Saved {len(gaps)} gap analyses to {output_path}/listing_gaps.json")

if __name__ == "__main__":
    train_gap_analyzer()
