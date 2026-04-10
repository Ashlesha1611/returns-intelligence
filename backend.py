from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from engine import engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    returns_df = pd.read_csv("datasets/Returns.csv")
    products_df = pd.read_csv("datasets/Products.csv")
    reviews_df = pd.read_csv("datasets/Reviews.csv")
    scores_df = pd.read_csv("datasets/Product_scores.csv")
    costs_df = pd.read_csv("datasets/Return_costs.csv")
    orders_df = pd.read_csv("datasets/Orders.csv")
    profile_emb_df = pd.read_csv("datasets/User_Profile_Embeddings.csv")
    
    # Train ALL Models
    engine.train_all(products_df, scores_df, reviews_df, returns_df, costs_df, orders_df, profile_emb_df)
except Exception as e:
    print(f"Error initializing: {e}")

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    total_returns = len(returns_df)
    total_orders = total_returns * 12 
    return_rate = round((total_returns / total_orders) * 100, 1) if total_orders else 0
    margin_erosion = float(costs_df["Total_return_cost"].sum())
    return {"return_rate": return_rate, "margin_erosion": margin_erosion}

@app.get("/api/dashboard/categories")
def get_top_categories():
    merged = pd.merge(returns_df, products_df, on="sku_id", how="inner")
    category_counts = merged["category"].value_counts().head(5).reset_index()
    category_counts.columns = ["name", "returns"]
    res = []
    for idx, row in category_counts.iterrows():
        res.append({"name": str(row["name"]), "returns": int(row["returns"])})
    return res

@app.get("/api/dashboard/actions")
def get_top_actions():
    return engine.get_priority_action_queue()

@app.get("/api/dashboard/scores")
def get_scores_trend():
    if "date" in scores_df.columns:
        scores_df["month"] = pd.to_datetime(scores_df["date"]).dt.strftime('%b %Y')
        monthly = scores_df.groupby("month")["avg_rating"].mean().reset_index()
        monthly["date_obj"] = pd.to_datetime(monthly["month"], format='%b %Y')
        monthly = monthly.sort_values(by="date_obj")
        monthly["score"] = monthly["avg_rating"] * 20
        return monthly.to_dict("records")
    return []

@app.get("/api/dashboard/gaps")
def get_listing_gaps():
    return engine.get_listing_gap_analysis()

class ProductSpec(BaseModel):
    price: float
    category: str
    material: str

class CustomerPrefs(BaseModel):
    sku_id: str
    tags: list[str]

@app.post("/api/ml/predict_product")
def predict_product_endpoint(spec: ProductSpec):
    return engine.predict_eww_risk(spec.price, spec.category)

@app.post("/api/customer/product")
def get_customer_product_endpoint(prefs: CustomerPrefs):
    return engine.get_smart_shopping_view(prefs.sku_id, prefs.tags)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
