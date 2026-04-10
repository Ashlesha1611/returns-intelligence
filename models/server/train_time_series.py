import pandas as pd
import os

def train_time_series(data_path="../../datasets", output_path="results"):
    print("Training Time-Series Aggregator...")
    os.makedirs(output_path, exist_ok=True)
    
    scores = pd.read_csv(f"{data_path}/Product_scores.csv")
    scores['date'] = pd.to_datetime(scores['date'])
    scores = scores.sort_values('date')
    
    results = []
    
    for sku, group in scores.groupby('sku_id'):
        group = group.set_index('date')
        
        # Calculate rolling means for windows
        rolling_7 = group['return_rate'].rolling('7D').mean()
        rolling_30 = group['return_rate'].rolling('30D').mean()
        rolling_90 = group['return_rate'].rolling('90D').mean()
        
        # Extract latest value of the windows
        if not group.empty:
            results.append({
                "sku_id": sku,
                "latest_date": str(group.index[-1].date()),
                "7d_trend": float(rolling_7.iloc[-1]) if not pd.isna(rolling_7.iloc[-1]) else 0,
                "30d_trend": float(rolling_30.iloc[-1]) if not pd.isna(rolling_30.iloc[-1]) else 0,
                "90d_trend": float(rolling_90.iloc[-1]) if not pd.isna(rolling_90.iloc[-1]) else 0
            })
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(f"{output_path}/rolling_scores.csv", index=False)
    print(f"Saved rolling windows to {output_path}/rolling_scores.csv")

if __name__ == "__main__":
    train_time_series()
