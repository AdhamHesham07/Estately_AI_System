import pandas as pd
import numpy as np
import joblib
import os
import logging
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION & GLOBAL SETTINGS
# ==========================================
CONFIG = {
    "paths": {
        "raw_data": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
        "artifacts_dir": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts"
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "structural_weights": [2.5, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5], # Price, Area, Beds, Baths, Lat, Lon, Amenities
    "hybrid_weights": {
        "structural": 0.30,
        "price_boost": 0.20,
        "semantic": 0.20,
        "geo": 0.15,
        "type": 0.15
    },
    "filters": {
        "price_relax_L1": 0.30,  # ±30%
        "price_relax_L2": 0.50   # ±50%
    },
    "geo_decay": {
        "town": 3.0,   # 3km decay factor
        "city": 8.0    # 8km decay factor
    },
    "epsilon": 1e-9
}

# Graded Similarity Matrix: Captures "Discovery" matches across similar types
PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7, 'iVilla': 0.6},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9, 'iVilla': 0.7},
    'Penthouse': {'Apartment': 0.8, 'Duplex': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9, 'iVilla': 0.7, 'Bungalow': 0.8},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9, 'iVilla': 0.6},
    'Twin House': {'Villa': 0.8, 'Townhouse': 0.9},
    'Chalet': {'Cabin': 0.9},
    'Cabin': {'Chalet': 0.9}
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculates distances between two sets of coordinates in kilometers."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def get_type_similarity(type1, type2):
    if type1 == type2: return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(type1, {}).get(type2, 0.0)

# ==========================================
# 3. DATA PIPELINE (END-TO-END)
# ==========================================
def load_and_prepare_data():
    logger.info(f"Loading raw dataset from {CONFIG['paths']['raw_data']}...")
    df = pd.read_csv(CONFIG['paths']['raw_data'], low_memory=False)
    original_count = len(df)
    
    # Ported Cleaning Logic
    def clean_numeric_feature(val, is_studio_zero=False):
        if pd.isna(val): return np.nan
        val = str(val).lower().strip()
        if is_studio_zero and val == 'studio': return 0
        if val == '7+': return 7
        try: return float(val)
        except: return np.nan

    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].apply(lambda x: clean_numeric_feature(x, True))
    if 'bathrooms' in df.columns:
        df['bathrooms'] = df['bathrooms'].apply(clean_numeric_feature)
        
    for col in ['price_egp', 'area_value', 'lat', 'lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=['listing_id', 'price_egp'])
    
    # Impute missing minor numeric features
    for col in ['lat', 'lon', 'bedrooms', 'bathrooms', 'area_value']:
        df[col] = df[col].fillna(df[col].median())
            
    # Outlier Removal (IQR)
    def remove_outliers(data, col):
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]
    
    logger.info("Applying Outlier Filtering...")
    df = remove_outliers(df, 'price_egp')
    df = remove_outliers(df, 'area_value')
    
    # Feature Engineering
    df['amenities_count'] = df['amenities'].fillna('').apply(lambda x: len(str(x).split('|')) if x and str(x).strip() else 0)
    df['description'] = df['description'].fillna('No description available')
    df['title'] = df['title'].fillna('No title available')
    
    # Index Consistency (CRITICAL for Positional Alignment)
    df = df.reset_index(drop=True)
    
    logger.info(f"Consolidated Preprocessing complete: {original_count} -> {len(df)}")
    return df

# ==========================================
# 4. THE RERANKING ENGINE (PRODUCTION LOGIC)
# ==========================================
def _rank_candidates(query_item, candidates_df, query_embed, candidate_embeds, knn_dists):
    """Core ranking math with progressive logic."""
    # 1. Structural (KNN Distance)
    structural_scores = np.exp(-knn_dists)
    
    # 2. Semantic (Embeddings)
    norm_q = query_embed / (np.linalg.norm(query_embed) + CONFIG["epsilon"])
    norm_c = candidate_embeds / (np.expand_dims(np.linalg.norm(candidate_embeds, axis=1), 1) + CONFIG["epsilon"])
    semantic_scores = np.dot(norm_c, norm_q)
    
    # 3. Price Proximity Boost
    q_price = query_item['price_egp']
    c_prices = candidates_df['price_egp'].values
    price_scores = np.exp(-np.abs(c_prices - q_price) / (q_price + CONFIG["epsilon"]))
    
    # 4. Adaptive Geo Score
    geo_distances = haversine_vectorized(query_item['lat'], query_item['lon'], 
                                         candidates_df['lat'].values, candidates_df['lon'].values)
    decay = CONFIG["geo_decay"]["town"] if pd.notna(query_item.get('town')) else CONFIG["geo_decay"]["city"]
    geo_scores = np.exp(-geo_distances / decay)
    
    # 5. Type Graded Similarity
    type_scores = np.array([get_type_similarity(query_item['property_type'], t) for t in candidates_df['property_type']])
    
    # Final Balanced Score
    w = CONFIG["hybrid_weights"]
    final_scores = (
        w["structural"] * structural_scores +
        w["price_boost"] * price_scores +
        w["semantic"]    * semantic_scores +
        w["geo"]         * geo_scores +
        w["type"]        * type_scores
    )
    return np.argsort(final_scores)[::-1]

# ==========================================
# 5. CORE PIPELINE OPERATIONS
# ==========================================
def evaluate_pipeline(df_train, df_test, knn, scaler, embeddings_train, embeddings_test):
    logger.info("Evaluating Pipeline Accuracy (Air-Gapped Hybrid MRR)...")
    
    cols = ['price_egp', 'area_value', 'bedrooms', 'bathrooms', 'lat', 'lon', 'amenities_count']
    X_test = df_test[cols].copy()
    X_test['price_egp'], X_test['area_value'] = np.log1p(X_test['price_egp']), np.log1p(X_test['area_value'])
    X_test_scaled = scaler.transform(X_test) * CONFIG["structural_weights"]
    
    distances, indices = knn.kneighbors(X_test_scaled)
    price_errors, mrr_scores = [], []
    
    for i in range(len(df_test)):
        q_item, q_embed = df_test.iloc[i], embeddings_test[i]
        c_idxs, c_dists = indices[i], distances[i]
        
        candidates = df_train.iloc[c_idxs]
        c_embeds = embeddings_train[c_idxs]
        
        # Call Production Ranking
        sorted_offsets = _rank_candidates(q_item, candidates, q_embed, c_embeds, c_dists)
        top_recs = candidates.iloc[sorted_offsets[:5]]
        
        # Metrics
        price_errors.append(np.mean([abs(p - q_item['price_egp']) / (q_item['price_egp'] + CONFIG["epsilon"]) for p in top_recs['price_egp']]))
        ranks = np.where(top_recs['property_type'].values == q_item['property_type'])[0]
        mrr_scores.append(1 / (ranks[0] + 1) if len(ranks) > 0 else 0)
        
    logger.info(f"Summary: Price Deviation={np.mean(price_errors)*100:.2f}%, MRR={np.mean(mrr_scores):.4f}")

def precompute_recs(df, knn, X_scaled, embeddings):
    logger.info("Precomputing All Recommendations...")
    distances, indices = knn.kneighbors(X_scaled)
    results = []
    
    for i in tqdm(range(len(df)), desc="End-to-End Ranking"):
        q_item, q_price, q_city = df.iloc[i], df.iloc[i]['price_egp'], df.iloc[i]['city']
        
        # Progressive Relaxation (Level 1: Strict -> Level 2: Wide -> Level 3: KNN)
        neighbor_indices, knn_dists = indices[i][1:], distances[i][1:]
        
        mask = (df.iloc[neighbor_indices]['price_egp'] >= q_price * (1 - CONFIG["filters"]["price_relax_L1"])) & \
               (df.iloc[neighbor_indices]['price_egp'] <= q_price * (1 + CONFIG["filters"]["price_relax_L1"])) & \
               (df.iloc[neighbor_indices]['city'] == q_city)
        
        offsets = np.where(mask)[0]
        if len(offsets) < 5: # Level 2
             mask = (df.iloc[neighbor_indices]['price_egp'] >= q_price * (1 - CONFIG["filters"]["price_relax_L2"])) & \
                    (df.iloc[neighbor_indices]['price_egp'] <= q_price * (1 + CONFIG["filters"]["price_relax_L2"])) & \
                    (df.iloc[neighbor_indices]['city'] == q_city)
             offsets = np.where(mask)[0]
        if len(offsets) < 5: # Level 3 fallback
            offsets = np.arange(len(neighbor_indices))
            
        c_idxs, final_dists = neighbor_indices[offsets], knn_dists[offsets]
        c_df, c_embeds = df.iloc[c_idxs], embeddings[c_idxs]
        
        sorted_offsets = _rank_candidates(q_item, c_df, embeddings[i], c_embeds, final_dists)
        top_recs = c_df.iloc[sorted_offsets[:10]]
        
        results.append({"listing_id": q_item['listing_id'], "recommendations": ",".join(top_recs['listing_id'].tolist())})
        
    return pd.DataFrame(results)

# ==========================================
# 6. MAIN ORCHESTRATION
# ==========================================
def main():
    if not os.path.exists(CONFIG["paths"]["artifacts_dir"]): os.makedirs(CONFIG["paths"]["artifacts_dir"])
    
    df = load_and_prepare_data()
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    # 1. Structural KNN
    logger.info("Initializing Structural KNN...")
    cols = ['price_egp', 'area_value', 'bedrooms', 'bathrooms', 'lat', 'lon', 'amenities_count']
    X_train = df_train[cols].copy()
    X_train['price_egp'], X_train['area_value'] = np.log1p(X_train['price_egp']), np.log1p(X_train['area_value'])
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train) * CONFIG["structural_weights"]
    
    knn = NearestNeighbors(n_neighbors=50, metric='euclidean', n_jobs=-1).fit(X_train_scaled)
    
    # 2. Semantic Vectors (AIR-GAPPED SPLIT)
    logger.info("Generating Semantic Embeddings... (Segregated for Evaluation)")
    st_model = SentenceTransformer(CONFIG["model_id"])
    
    train_texts = (df_train['title'] + " " + df_train['description']).tolist()
    embeddings_train = st_model.encode(train_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    test_texts = (df_test['title'] + " " + df_test['description']).tolist()
    embeddings_test = st_model.encode(test_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Evaluation
    evaluate_pipeline(df_train, df_test, knn, scaler, embeddings_train, embeddings_test)
    
    # 4. Production Re-assembly
    logger.info("Reconstructing full embedding matrix for production...")
    # Map back to original indices to keep final export aligned with df
    embeddings_full = np.zeros((len(df), embeddings_train.shape[1]))
    embeddings_full[df_train.index] = embeddings_train
    embeddings_full[df_test.index] = embeddings_test
    
    X_full = df[cols].copy()
    X_full['price_egp'], X_full['area_value'] = np.log1p(X_full['price_egp']), np.log1p(X_full['area_value'])
    X_full_scaled = scaler.transform(X_full) * CONFIG["structural_weights"]
    
    recs_df = precompute_recs(df, knn, X_full_scaled, embeddings_full)
    
    joblib.dump(knn, os.path.join(CONFIG["paths"]["artifacts_dir"], 'knn_v1.joblib'))
    joblib.dump(scaler, os.path.join(CONFIG["paths"]["artifacts_dir"], 'scaler_v1.joblib'))
    np.save(os.path.join(CONFIG["paths"]["artifacts_dir"], 'embeddings_v1.npy'), embeddings_full)
    recs_df.to_csv(os.path.join(CONFIG["paths"]["artifacts_dir"], 'recs_v1.csv'), index=False)
    
    logger.info("Success! Production Artifacts Saved.")

if __name__ == "__main__":
    main()
