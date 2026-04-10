import pandas as pd
import numpy as np
import os
import faiss

def train_faiss_indexer(data_path="../../datasets", output_path="results"):
    print("Training FAISS Indexer...")
    os.makedirs(output_path, exist_ok=True)
    
    # Check if a new embedding file is available, else fallback to raw dummy dataset
    embed_file = f"{output_path}/User_Profile_Embeddings.csv"
    if not os.path.exists(embed_file):
        embed_file = f"{data_path}/User_Profile_Embeddings.csv"
        
    embeddings_df = pd.read_csv(embed_file)
    
    def parse_vector(v_str):
        if pd.isna(v_str): return np.zeros(50)
        return np.array([float(x) for x in v_str.split(',')], dtype=np.float32)

    vectors = np.array([parse_vector(x) for x in embeddings_df['embedding_vector'].values])
    
    # 50 dimensions for User Profile embeddings
    if len(vectors) > 0:
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        faiss.write_index(index, f"{output_path}/faiss_index.bin")
        
        # Save map so engine knows which ID corresponds to which faiss index row
        pd.DataFrame({"customer_id": embeddings_df['customer_id']}).to_csv(f"{output_path}/faiss_customer_map.csv", index=False)
        print(f"Saved FAISS index for {len(vectors)} profiles to {output_path}/faiss_index.bin")
    else:
        print("No embeddings found to index.")

if __name__ == "__main__":
    train_faiss_indexer()
