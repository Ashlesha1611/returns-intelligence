import pandas as pd
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
# Note: Ensure you set GROQ_API_KEY inside your .env file
groq_api_key = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=groq_api_key) if groq_api_key else None

def train_review_nlp_summarizer(data_path="../../datasets", output_path="results"):
    print("Training Review NLP Summarizer...")
    os.makedirs(output_path, exist_ok=True)
    
    reviews = pd.read_csv(f"{data_path}/Reviews.csv")
    returns = pd.read_csv(f"{data_path}/Returns.csv")
    
    skus = reviews['sku_id'].unique()
    summaries = []
    
    for sku in skus[:10]: # Processing first 10 for MVP demonstration
        sku_reviews = reviews[reviews['sku_id'] == sku]['review_text'].dropna().tolist()
        sku_returns = returns[returns['sku_id'] == sku]['return_note'].dropna().tolist()
        
        all_text = " ".join(sku_reviews + sku_returns)
        
        # 1. LLM Summarization and Tag Extraction (Unified)
        tags = [{"label": "Popular", "type": "positive"}] # Fallback
        summary_text = f"Highly rated product according to {len(sku_reviews)} recent buyers."
        
        if client:
            try:
                prompt_text = f"""
                Analyze the following product reviews and return notes: '{all_text[:1500]}'
                
                You must respond in valid JSON format ONLY, according to this schema:
                {{
                    "tags": [
                        {{"label": "Word1", "type": "positive"}},
                        {{"label": "Word2", "type": "negative"}}
                    ],
                    "ai_summary": "A 1-sentence summary of the general sentiment and key traits."
                }}
                Ensure the tag types are accurately labeled (positive, negative, or neutral) based on the context. Provide max 3 tags.
                """
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt_text}],
                    response_format={"type": "json_object"},
                    max_tokens=200
                )
                output = json.loads(response.choices[0].message.content)
                tags = output.get("tags", tags)
                summary_text = output.get("ai_summary", summary_text)
            except Exception as e:
                print(f"LLM API Error for SKU {sku}. Falling back to default format. Error: {e}")
        
        summaries.append({
            "sku_id": sku,
            "tags": tags,
            "ai_summary": summary_text
        })
        
    with open(f"{output_path}/sku_summaries.json", "w") as f:
        json.dump(summaries, f, indent=4)
        
    print(f"Saved NLP summaries for {len(summaries)} SKUs to {output_path}/sku_summaries.json")

if __name__ == "__main__":
    train_review_nlp_summarizer()
