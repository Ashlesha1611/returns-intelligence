import pandas as pd
import os
import json

def train_delivery_aggregator(data_path="../../datasets", output_path="results"):
    print("Training Delivery Aggregator...")
    os.makedirs(output_path, exist_ok=True)
    
    orders = pd.read_csv(f"{data_path}/Orders.csv")
    order_items = pd.read_csv(f"{data_path}/Order_items.csv")
    
    merged = pd.merge(order_items, orders, on="order_id", how="inner")
    
    # Group by SKU to calculate median shipping turnaround length
    delivery_stats = merged.groupby('sku_id')['shipping_time_days'].median().reset_index()
    
    # Key-Value mapping for simple frontend API ingestion
    results = {}
    for _, row in delivery_stats.iterrows():
        results[row['sku_id']] = int(row['shipping_time_days']) if pd.notna(row['shipping_time_days']) else 7
        
    with open(f"{output_path}/shipping_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved delivery stats for {len(results)} SKUs to {output_path}/shipping_stats.json")

if __name__ == "__main__":
    train_delivery_aggregator()
