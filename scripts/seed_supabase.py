import os
import json
import math
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_dict(row_dict):
    """Recursively converts pandas NaN or math.nan to None for JSON compliance."""
    for key, val in row_dict.items():
        if isinstance(val, float) and math.isnan(val):
            row_dict[key] = None
        elif pd.isna(val):
             row_dict[key] = None
    return row_dict

def insert_csv_to_table(filepath, table_name, chunk_size=200):
    print(f"Uploading {filepath} to {table_name}...")
    if not os.path.exists(filepath):
        print(f"  -> File not found: {filepath}")
        return
        
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    records = df.to_dict(orient="records")
    
    # Needs explicit nan handling for JSON compatibility
    records = [clean_dict(r) for r in records]
    
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i+chunk_size]
        try:
            supabase.table(table_name).upsert(chunk).execute()
        except Exception as e:
            print(f"  -> Error inserting chunk at {i}: {e}")
            
    print(f"  -> Finished {table_name}. Uploaded {len(records)} rows.\n")

def insert_json_to_table(filepath, table_name):
    print(f"Uploading {filepath} to {table_name}...")
    if not os.path.exists(filepath):
        print(f"  -> File not found: {filepath}")
        return
        
    with open(filepath, 'r') as f:
        records = json.load(f)
        
    for i in range(0, len(records), 100):
        chunk = records[i:i+100]
        try:
            supabase.table(table_name).insert(chunk).execute()
        except Exception as e:
            print(f"  -> Error inserting chunk at {i}: {e}")
            
    print(f"  -> Finished {table_name}. Uploaded {len(records)} rows.\n")

def seed_database():
    dataset_dir = "datasets"
    
    # 1. Base Entities
    insert_csv_to_table(f"{dataset_dir}/Customers.csv", "customers")
    insert_csv_to_table(f"{dataset_dir}/Products.csv", "products")
    
    # 2. Orders pipeline
    insert_csv_to_table(f"{dataset_dir}/Orders.csv", "orders")
    insert_csv_to_table(f"{dataset_dir}/Order_items.csv", "order_items")
    
    # 3. Post-Purchase Loop
    insert_csv_to_table(f"{dataset_dir}/Returns.csv", "returns")
    insert_csv_to_table(f"{dataset_dir}/Reviews.csv", "reviews")
    
    # 4. ML Embeddings
    # Vector embedding strings like "0.5,0.4,..." become native Postgres arrays [0.5, 0.4, ...]
    print("Uploading user_embeddings...")
    emb_path = "models/client/results/User_Profile_Embeddings.csv"
    if os.path.exists(emb_path):
        emb_df = pd.read_csv(emb_path)
        emb_records = []
        for _, row in emb_df.iterrows():
            vec_str = row['embedding_vector']
            vec_list = [float(x) for x in str(vec_str).split(",")]
            emb_records.append({
                "customer_id": row['customer_id'],
                "embedding_vector": vec_list,
                "last_updated": row['last_updated']
            })
        
        for i in range(0, len(emb_records), 200):
            try:
                supabase.table("user_embeddings").upsert(emb_records[i:i+200]).execute()
            except Exception as e:
                print(f"  -> Error inserting embeddings at {i}: {e}")
        print(f"  -> Finished user_embeddings. Uploaded {len(emb_records)} rows.\n")
    
    # 5. ML Insights JSON
    insert_json_to_table("models/server/results/priority_actions.json", "priority_actions")
    insert_json_to_table("models/client/results/sku_summaries.json", "sku_summaries")
    
    print("Full Database Seeding Complete!")

if __name__ == "__main__":
    seed_database()
