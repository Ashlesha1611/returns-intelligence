import pandas as pd
import os

def train_return_aggregator(data_path="../../datasets", output_path="results"):
    print("Training Return Rate Aggregator...")
    os.makedirs(output_path, exist_ok=True)
    
    returns = pd.read_csv(f"{data_path}/Returns.csv")
    orders = pd.read_csv(f"{data_path}/Order_items.csv") 
    products = pd.read_csv(f"{data_path}/Products.csv")
    
    # Total sold per SKU
    sold_vol = orders.groupby('sku_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'total_sold'})
    
    # Total returned per SKU
    ret_vol = returns.groupby('sku_id').size().reset_index(name='total_returned')
    
    # Merge and calculate rate
    agg = pd.merge(sold_vol, ret_vol, on='sku_id', how='left').fillna(0)
    agg['return_rate'] = (agg['total_returned'] / agg['total_sold']).fillna(0)
    
    # Add product metadata
    agg = pd.merge(agg, products[['sku_id', 'category', 'product_name']], on='sku_id', how='left')
    
    # Sort
    agg = agg.sort_values(by=['total_returned', 'return_rate'], ascending=[False, False])
    
    agg.to_csv(f"{output_path}/return_rates.csv", index=False)
    print(f"Saved return aggregation for {len(agg)} SKUs to {output_path}/return_rates.csv")

if __name__ == "__main__":
    train_return_aggregator()
