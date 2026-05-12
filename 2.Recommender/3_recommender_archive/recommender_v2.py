import pandas as pd
import numpy as np
import joblib
import os
import logging
import faiss
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION (ENTERPRISE V2.0)
# ==========================================
CONFIG = {
    "paths": {
        "raw_data": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
        "artifacts_dir": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts_v2"
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    "rerank_candidates": 100, # Stage 1 retrieval size
    "epsilon": 1e-9
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2. UTILITY & GEO
# ==========================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def get_type_similarity(t1, t2):
    # (Simplified for v2, could be expanded to matrix)
    return 1.0 if t1 == t2 else 0.2 

# ==========================================
# 3. ADVANCED DATA PIPELINE
# ==========================================
def load_and_preprocess_v2():
    logger.info("Starting Advanced Preprocessing v2.0...")
    df = pd.read_csv(CONFIG['paths']['raw_data'], low_memory=False)
    
    # 1. Numeric Cleanup
    for col in ['price_egp', 'area_value', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['listing_id', 'price_egp'])
    
    # 2. Amenities Binary Encoding (Lifestyle Matrix)
    logger.info("Building Lifestyle Amenity Matrix...")
    for amenity in CONFIG["top_amenities"]:
        df[f"has_{amenity.lower().replace(' ', '_')}"] = df['amenities'].fillna('').apply(
            lambda x: 1 if amenity.lower() in str(x).lower() else 0
        )
    
    # 3. Imputation & Normalization
    df['amenities_count'] = df['amenities'].fillna('').apply(lambda x: len(str(x).split('|')) if x else 0)
    for col in ['lat', 'lon', 'area_value']:
        df[col] = df[col].fillna(df[col].median())
    
    df = df.reset_index(drop=True)
    return df

# ==========================================
# 4. STAGE 2 RANKER (LEARNING-TO-RANK)
# ==========================================
def train_xgboost_ranker(df, embeddings):
    """Trains a synthetic XGBoost ranker to replace static weights."""
    logger.info("Training Stage-2 XGBoost Ranker (Synthetic Teacher method)...")
    
    # Generate Synthetic Training Set
    # We pick 500 samples and find 20 neighbors for each
    train_indices = np.random.choice(len(df), 500, replace=False)
    X_train, y_train = [], []
    
    amenity_cols = [f"has_{a.lower().replace(' ', '_')}" for a in CONFIG["top_amenities"]]
    
    for i in tqdm(train_indices, desc="Synthesizing Ranker Data"):
        q = df.iloc[i]
        q_emb = embeddings[i]
        
        # Simple sample: just take some random entries to learn "what makes a good match"
        sample_idxs = np.random.choice(len(df), 50)
        c_df = df.iloc[sample_idxs]
        c_embs = embeddings[sample_idxs]
        
        # Calculate features for ranker
        price_diff = np.abs(c_df['price_egp'].values - q['price_egp']) / (q['price_egp'] + 1e-9)
        geo_dist = haversine_vectorized(q['lat'], q['lon'], c_df['lat'].values, c_df['lon'].values)
        
        # Semantic Sim
        sims = np.dot(c_embs, q_emb) / (np.linalg.norm(c_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        
        # Amenity overlap
        q_amenities = q[amenity_cols].values
        c_amenities = c_df[amenity_cols].values
        amenity_overlap = np.sum(q_amenities * c_amenities, axis=1) / (np.maximum(1, np.sum(q_amenities)))
        
        for idx in range(len(c_df)):
            # Target Score: Heuristic-based synthetic "Ground Truth"
            # In a real v2.0, this would come from user clicks/bookings.
            target_score = (
                (1 - np.clip(price_diff[idx], 0, 1)) * 0.4 + 
                (1 - np.clip(geo_dist[idx]/20, 0, 1)) * 0.2 + 
                sims[idx] * 0.2 + 
                amenity_overlap[idx] * 0.2
            )
            
            X_train.append([price_diff[idx], geo_dist[idx], sims[idx], amenity_overlap[idx]])
            y_train.append(target_score)
            
    ranker = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    ranker.fit(np.array(X_train), np.array(y_train))
    return ranker

# ==========================================
# 5. RETRIEVAL & VERIFICATION
# ==========================================
def evaluate_v2(df, faiss_index, embeddings, ranker):
    logger.info("Verifying v2.0 Performance Signal...")
    # Select 200 random test queries
    test_idxs = np.random.choice(len(df), 200, replace=False)
    
    latencies = []
    amenity_recalls = []
    
    amenity_cols = [f"has_{a.lower().replace(' ', '_')}" for a in CONFIG["top_amenities"]]
    
    for i in test_idxs:
        q = df.iloc[i]
        q_emb = np.ascontiguousarray(embeddings[i:i+1]).astype('float32')
        
        # Latency Phase
        import time
        start = time.time()
        D, I = faiss_index.search(q_emb, 50)
        latencies.append((time.time() - start) * 1000)
        
        # Ranking check
        top_recs = df.iloc[I[0][:10]]
        q_lifestyle = q[amenity_cols].values
        if np.sum(q_lifestyle) > 0:
            rec_lifestyle = top_recs[amenity_cols].values
            # Check overlap for the top 1 rec
            overlap = np.sum(q_lifestyle * rec_lifestyle[0]) / np.sum(q_lifestyle)
            amenity_recalls.append(overlap)
            
    logger.info(f"--- Verification Dashboard v2.0 ---")
    logger.info(f"Retrieval Latency (FAISS): {np.mean(latencies):.2f} ms")
    logger.info(f"Lifestyle Recall@1 (Amenity match): {np.mean(amenity_recalls)*100:.2f}%")
    logger.info(f"-----------------------------------")

def main():
    if not os.path.exists(CONFIG["paths"]["artifacts_dir"]): os.makedirs(CONFIG["paths"]["artifacts_dir"])
    
    df = load_and_preprocess_v2()
    
    # 1. Semantic Embeddings
    logger.info("Building Semantic Vectors...")
    model = SentenceTransformer(CONFIG["model_id"])
    texts = (df['title'] + " " + df['description']).tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # 2. FAISS Index (Stage 1 retrieval)
    logger.info("Building FAISS Index...")
    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) # Inner Product for Cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    # 3. Stage 2 Ranker (LTR)
    ranker = train_xgboost_ranker(df, embeddings)
    
    # 4. Verification
    evaluate_v2(df, index, embeddings, ranker)
    
    # 5. Export
    logger.info("Exporting Enterprise v2.0 Artifacts...")
    faiss.write_index(index, os.path.join(CONFIG["paths"]["artifacts_dir"], "faiss_v2.index"))
    ranker.save_model(os.path.join(CONFIG["paths"]["artifacts_dir"], "ranker_v2.json"))
    np.save(os.path.join(CONFIG["paths"]["artifacts_dir"], "embeddings_v2.npy"), embeddings)
    df.to_pickle(os.path.join(CONFIG["paths"]["artifacts_dir"], "processed_df_v2.pkl"))
    
    logger.info("DONE! Enterprise v2.0 Engine is ready.")

if __name__ == "__main__":
    main()
