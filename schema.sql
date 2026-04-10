-- Drop existing tables to ensure a clean slate if re-running
DROP TABLE IF EXISTS sku_summaries CASCADE;
DROP TABLE IF EXISTS priority_actions CASCADE;
DROP TABLE IF EXISTS user_embeddings CASCADE;
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS returns CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Enable the pgvector extension to replace FAISS internally
CREATE EXTENSION IF NOT EXISTS vector;

-- Core Tables
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    age_group TEXT,
    lifestyle_tags TEXT,
    preferred_styles TEXT,
    preferred_colors TEXT,
    lighting_condition TEXT,
    avg_spend FLOAT,
    return_rate FLOAT
);

CREATE TABLE products (
    sku_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    sub_category TEXT,
    brand TEXT,
    price FLOAT,
    material TEXT,
    color TEXT,
    finish TEXT,
    dimensions TEXT,
    weight FLOAT,
    style TEXT,
    description_text TEXT,
    image_url TEXT,
    Launch_date TEXT
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    order_date TIMESTAMP,
    delivery_date TIMESTAMP,
    shipping_time_days FLOAT,
    total_amount FLOAT,
    Payment_type TEXT
);

CREATE TABLE order_items (
    order_item_id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES orders(order_id),
    sku_id TEXT REFERENCES products(sku_id),
    quantity INT,
    price_per_unit FLOAT,
    Discount FLOAT
);

CREATE TABLE returns (
    return_id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES orders(order_id),
    order_item_id TEXT REFERENCES order_items(order_item_id),
    sku_id TEXT REFERENCES products(sku_id),
    customer_id TEXT REFERENCES customers(customer_id),
    return_reason TEXT,
    return_note TEXT,
    return_date TIMESTAMP,
    return_status TEXT,
    refund_amount FLOAT,
    return_shipping_cost FLOAT,
    condition_received TEXT
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    sku_id TEXT REFERENCES products(sku_id),
    customer_id TEXT REFERENCES customers(customer_id),
    rating INT,
    review_text TEXT,
    review_title TEXT,
    sentiment_score FLOAT,
    review_date TIMESTAMP,
    verified_purchase BOOLEAN,
    image_uploaded BOOLEAN
);

-- ML Results Tables (Data Warehousing)
CREATE TABLE user_embeddings (
    customer_id TEXT PRIMARY KEY REFERENCES customers(customer_id),
    embedding_vector vector(50), 
    last_updated TIMESTAMP
);

CREATE TABLE priority_actions (
    sku_id TEXT PRIMARY KEY REFERENCES products(sku_id),
    product_name TEXT,
    urgency_score FLOAT,
    velocity INT,
    revenue_at_risk FLOAT,
    action TEXT,
    pain_points JSONB,
    original_description TEXT,
    suggested_rewrite TEXT
);

CREATE TABLE sku_summaries (
    sku_id TEXT PRIMARY KEY REFERENCES products(sku_id),
    tags JSONB,
    ai_summary TEXT
);
