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

# Set Global Reproducibility (v3.4)
np.random.seed(42)

# ==========================================
# 1. CONFIGURATION (PRODUCTION V3.0)
# ==========================================
CONFIG = {
    "paths": {
        "raw_data": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
        "artifacts_dir": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts_v3"
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    "retrieval": {
        "k_candidates": 100,
        "nprobe": 10,
        "recall_target": 0.95
    },
    "ranking": {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10"
    },
    "epsilon": 1e-9
}

# Graded Similarity Matrix (Re-integrated from v1)
PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9},
    'Chalet': {'Cabin': 0.9}
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2. CORE UTILITIES
# ==========================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def normalize_query(q):
    """Ensures query vectors are L2-normalized for stable cosine search."""
    q_copy = np.ascontiguousarray(q.copy()).astype('float32')
    if q_copy.ndim == 1:
        q_copy = q_copy.reshape(1, -1)
    faiss.normalize_L2(q_copy)
    return q_copy

def get_type_sim(t1, t2):
    if t1 == t2: return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(t1, {}).get(t2, 0.05)

def jaccard_similarity(q_vec, c_matrix):
    """Robust 1D query vs 2D candidate matrix broadcasting."""
    # Ensure q_vec is expanded to (1, N) for safe broadcasting
    q_vec = q_vec.reshape(1, -1)
    intersection = np.sum(q_vec * c_matrix, axis=1)
    union = np.sum(np.maximum(q_vec, c_matrix), axis=1)
    return intersection / (union + 1e-9)

# Feature Order Registry (Master Protection)
FEATURE_COLS = ['price_prox', 'geo_log', 'semantic_sim', 'type_sim', 'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff']

# ==========================================
# 3. ADVANCED DATA PIPELINE
# ==========================================
def load_and_preprocess_v3():
    logger.info("Ingesting Raw Data for v3.0 Baseline...")
    df = pd.read_csv(CONFIG['paths']['raw_data'], low_memory=False)
    
    # 1. Feature Sanitization
    for col in ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['listing_id', 'price_egp'])
    
    # 2. Binary Lifestyle Matrix
    for amenity in CONFIG["top_amenities"]:
        df[f"amenity_{amenity.lower().replace(' ', '_')}"] = df['amenities'].fillna('').apply(
            lambda x: 1 if amenity.lower() in str(x).lower() else 0
        )
    
    # 3. Outlier Clipping & Imputation (v3.3)
    p_q1, p_q3 = df['price_egp'].quantile([0.25, 0.75])
    p_limit = p_q3 + 1.5 * (p_q3 - p_q1)
    df = df[df['price_egp'] <= p_limit]
    
    # Impute Structural Features (NaN Protection)
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median()).astype(int)
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median()).astype(int)
    
    # 4. De-biasing feature (v3.1) - Log transformed for stability
    df['desc_len'] = np.log1p(df['description'].fillna('').apply(len))
    
    df = df.reset_index(drop=True)
    return df

# ==========================================
# 4. SYNTHETIC PREFERENCE MODEL (SPM)
# ==========================================
def calculate_spm_relevance(query, candidates_df):
    """
    Calculates an OBJECTIVE relevance proxy (0-4) based on hard constraints.
    This is our ground-truth teacher for the LTR model.
    """
    q_price = query['price_egp']
    c_prices = candidates_df['price_egp'].values
    price_ratio = np.abs(c_prices - q_price) / (q_price + 1e-9)
    
    geo_dist = haversine_vectorized(query['lat'], query['lon'], candidates_df['lat'].values, candidates_df['lon'].values)
    type_sims = np.array([get_type_sim(query['property_type'], t) for t in candidates_df['property_type'].values])
    same_city = (candidates_df['city'].values == query['city']).astype(int)
    
    # Relevance Logic
    relevance = np.zeros(len(candidates_df))
    
    # Score 4: Perfect match
    relevance[(same_city == 1) & (type_sims >= 1.0) & (price_ratio <= 0.1)] = 4
    # Score 3: Great match
    relevance[(relevance == 0) & (same_city == 1) & (type_sims >= 0.8) & (price_ratio <= 0.25)] = 3
    # Score 2: Good match
    relevance[(relevance == 0) & (same_city == 1) & (price_ratio <= 0.4)] = 2
    # Score 1: Weak match
    relevance[(relevance == 0) & (price_ratio <= 0.6)] = 1
    
    return relevance.astype(int)

# ==========================================
# 5. LTR TRAINING (TRIPLE-NEGATIVE SAMPLING)
# ==========================================
def generate_ltr_training_data(df, embeddings, faiss_index, id_to_idx):
    logger.info("Generating Training Dataset with Hard-Negative Sampling (v3.4 Mapping)...")
    train_indices = np.random.choice(len(df), 400, replace=False)
    
    X_train, y_train, qids = [], [], []
    amenity_cols = [f"amenity_{a.lower().replace(' ', '_')}" for a in CONFIG["top_amenities"]]
    
    for qid, i in enumerate(tqdm(train_indices)):
        query = df.iloc[i]
        q_emb = normalize_query(embeddings[i])
        
        # STAGE 1 RETRIEVAL
        D, I = faiss_index.search(q_emb, 100)
        
        # ID-to-Index Translation (The Master Fix)
        retrieved_ids = I[0]
        row_indices = [id_to_idx[lid] for lid in retrieved_ids if lid in id_to_idx]
        
        candidates_df = df.iloc[row_indices]
        c_embs = embeddings[row_indices]
        
        # Features (Stabilized)
        price_prox = np.log1p(np.abs(candidates_df['price_egp'].values - query['price_egp']) / (query['price_egp'] + 1e-9))
        geo_log = np.log1p(haversine_vectorized(query['lat'], query['lon'], candidates_df['lat'].values, candidates_df['lon'].values))
        semantic_sim = np.dot(c_embs, embeddings[i]) 
        type_sim = np.array([get_type_sim(query['property_type'], t) for t in candidates_df['property_type'].values])
        amenity_jaccard = jaccard_similarity(query[amenity_cols].values.astype(float), candidates_df[amenity_cols].values.astype(float))
        desc_log_diff = np.abs(candidates_df['desc_len'].values - query['desc_len'])
        beds_diff = np.abs(candidates_df['bedrooms'].values - query['bedrooms'])
        baths_diff = np.abs(candidates_df['bathrooms'].values - query['bathrooms'])
        
        # Labels from SPM
        relevance_labels = calculate_spm_relevance(query, candidates_df)
        
        for idx in range(len(candidates_df)):
            features = [
                price_prox[idx], geo_log[idx], semantic_sim[idx], 
                type_sim[idx], amenity_jaccard[idx], desc_log_diff[idx],
                beds_diff[idx], baths_diff[idx]
            ]
            X_train.append(features)
            y_train.append(relevance_labels[idx])
            qids.append(qid)
            
    return np.array(X_train), np.array(y_train), np.array(qids)

# ==========================================
# 6. DIVERSITY & VERIFICATION
# ==========================================
def apply_diversity_rerank(df_recs, top_n=10):
    """Prevents clustering by deduplicating properties from the same building/district."""
    final_recs = []
    seen_buildings = set()
    seen_districts = {} # Limit max 3 per district
    
    for _, row in df_recs.iterrows():
        b_id = row.get('building_name', row['listing_id'])
        d_id = row.get('district', 'unknown')
        
        dist_count = seen_districts.get(d_id, 0)
        
        # Diversity Condition
        if b_id not in seen_buildings and dist_count < 3:
            final_recs.append(row)
            seen_buildings.add(b_id)
            seen_districts[d_id] = dist_count + 1
        
        if len(final_recs) >= top_n: break
    return pd.DataFrame(final_recs)

def evaluate_v3(df_train, df_test, emb_train, emb_test, faiss_index, ranker, id_to_idx_train):
    logger.info("Conducting Air-Gapped Verification (v3.4 Safe Mapping)...")
    test_idxs = np.random.choice(len(df_test), 100, replace=False)
    
    n_dcgs = []
    amenity_cols = [f"amenity_{a.lower().replace(' ', '_')}" for a in CONFIG["top_amenities"]]
    
    for idx_in_test in tqdm(test_idxs, desc="Calculating nDCG@10"):
        query = df_test.iloc[idx_in_test]
        q_emb = normalize_query(emb_test[idx_in_test])
        
        # 1. Retrieval (Safe ID Mapping)
        D, I = faiss_index.search(q_emb, 100)
        row_indices = [id_to_idx_train[lid] for lid in I[0] if lid in id_to_idx_train]
        
        candidates = df_train.iloc[row_indices]
        c_embs = emb_train[row_indices]
        
        # 2. Ranking Features
        price_prox = np.log1p(np.abs(candidates['price_egp'].values - query['price_egp']) / (query['price_egp'] + 1e-9))
        geo_log = np.log1p(haversine_vectorized(query['lat'], query['lon'], candidates['lat'].values, candidates['lon'].values))
        semantic_sim = np.dot(c_embs, emb_test[idx_in_test])
        type_sim = np.array([get_type_sim(query['property_type'], t) for t in candidates['property_type'].values])
        amenity_jaccard = jaccard_similarity(query[amenity_cols].values.astype(float), candidates[amenity_cols].values.astype(float))
        desc_log_diff = np.abs(candidates['desc_len'].values - query['desc_len'])
        beds_diff = np.abs(candidates['bedrooms'].values - query['bedrooms'])
        baths_diff = np.abs(candidates['bathrooms'].values - query['bathrooms'])
        
        X_eval = np.stack([price_prox, geo_log, semantic_sim, type_sim, amenity_jaccard, desc_log_diff, beds_diff, baths_diff], axis=1)
        
        # 3. Predict & Rank
        scores = ranker.predict(X_eval)
        final_top_idx = np.argsort(scores)[::-1]
        
        # 4. Apply Diversity (v3.1)
        ranked_candidates = candidates.iloc[final_top_idx]
        diverse_top = apply_diversity_rerank(ranked_candidates, top_n=10)
        
        # 5. Metrics (against SPM)
        true_relevance = calculate_spm_relevance(query, candidates)
        rel_map = dict(zip(candidates['listing_id'], true_relevance))
        diverse_rel = [rel_map[lid] for lid in diverse_top['listing_id']]
        
        def dcg(rel):
            return np.sum((2**np.array(rel) - 1) / np.log2(np.arange(2, len(rel) + 2)))
            
        actual_dcg = dcg(diverse_rel[:10])
        ideal_dcg = dcg(np.sort(true_relevance)[::-1][:10])
        
        n_dcgs.append(actual_dcg / (ideal_dcg + 1e-9))
        
    logger.info(f"--- Production Integrity Dashboard v3.3 Master ---")
    logger.info(f"nDCG@10 (Total Integrity): {np.mean(n_dcgs):.4f}")
    logger.info(f"Feature Importance (v3.3): {dict(zip(FEATURE_COLS, ranker.feature_importances_))}")
    logger.info(f"-------------------------------------------")

# ==========================================
# 7. PRODUCTION PRECOMPUTE
# ==========================================
def precompute_recs_v3(df, embeddings, faiss_index, ranker, id_to_idx):
    logger.info("Precomputing Production Recommendations (v3.4 Safe Mapping)...")
    results = []
    amenity_cols = [f"amenity_{a.lower().replace(' ', '_')}" for a in CONFIG["top_amenities"]]
    
    for i in tqdm(range(len(df)), desc="Reranking Market"):
        query = df.iloc[i]
        q_emb = normalize_query(embeddings[i])
        
        # 1. FAISS Stage 1 (Safe mapping)
        D, I = faiss_index.search(q_emb, 101) # 101 to get 100 after excluding self
        row_indices = [id_to_idx[lid] for lid in I[0] if lid in id_to_idx]
        
        candidates = df.iloc[row_indices]
        # Filter self
        candidates = candidates[candidates['listing_id'] != query['listing_id']].iloc[:100]
        c_embs = embeddings[candidates.index]
        
        # 2. XGBoost Stage 2
        price_prox = np.log1p(np.abs(candidates['price_egp'].values - query['price_egp']) / (query['price_egp'] + 1e-9))
        geo_log = np.log1p(haversine_vectorized(query['lat'], query['lon'], candidates['lat'].values, candidates['lon'].values))
        semantic_sim = np.dot(c_embs, embeddings[i])
        type_sim = np.array([get_type_sim(query['property_type'], t) for t in candidates['property_type'].values])
        amenity_jaccard = jaccard_similarity(query[amenity_cols].values.astype(float), candidates[amenity_cols].values.astype(float))
        desc_log_diff = np.abs(candidates['desc_len'].values - query['desc_len'])
        beds_diff = np.abs(candidates['bedrooms'].values - query['bedrooms'])
        baths_diff = np.abs(candidates['bathrooms'].values - query['bathrooms'])
        
        X_prod = np.stack([price_prox, geo_log, semantic_sim, type_sim, amenity_jaccard, desc_log_diff, beds_diff, baths_diff], axis=1)
        
        scores = ranker.predict(X_prod)
        ranked_candidates = candidates.iloc[np.argsort(scores)[::-1]]
        
        # 3. Diversity Stage 3
        diverse_top = apply_diversity_rerank(ranked_candidates, top_n=10)
        
        results.append({
            "listing_id": query['listing_id'],
            "recommendations": ",".join(diverse_top['listing_id'].astype(str))
        })
        
    return pd.DataFrame(results)

# ==========================================
# 7. MAIN ORCHESTRATION
# ==========================================
def main():
    if not os.path.exists(CONFIG["paths"]["artifacts_dir"]): os.makedirs(CONFIG["paths"]["artifacts_dir"])
    
    df = load_and_preprocess_v3()
    
    # 0. Air-Gapped Split (v3.4 Fixed Seeds)
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    
    # 0.1 Reverse Mapping (The "Master Link")
    id_to_idx_train = {lid: i for i, lid in enumerate(df_train['listing_id'])}
    
    # 1. Embeddings
    model = SentenceTransformer(CONFIG["model_id"])
    emb_train = model.encode((df_train['title'] + " " + df_train['description']).tolist(), batch_size=32, show_progress_bar=True)
    emb_test = model.encode((df_test['title'] + " " + df_test['description']).tolist(), batch_size=32, show_progress_bar=True)
    
    # 2. FAISS ID-Mapped Index
    listing_ids = df_train['listing_id'].values.astype('int64')
    sub_index = faiss.IndexFlatIP(emb_train.shape[1])
    index = faiss.IndexIDMap2(sub_index)
    
    faiss.normalize_L2(emb_train)
    faiss.normalize_L2(emb_test)
    index.add_with_ids(emb_train.astype('float32'), listing_ids)
    
    # 3. LTR Training
    X, y, qids = generate_ltr_training_data(df_train, emb_train, index, id_to_idx_train)
    
    unique_qids = np.unique(qids)
    train_q, val_q = train_test_split(unique_qids, test_size=0.15, random_state=42)
    
    train_mask = np.isin(qids, train_q)
    val_mask = np.isin(qids, val_q)
    
    X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
    X_val, y_val, qids_val = X[val_mask], y[val_mask], qids[val_mask]
    
    # Strict QID Sorting (Master Fix v3.2)
    def sort_by_qid(X_arr, y_arr, q_arr):
        order = np.argsort(q_arr)
        return X_arr[order], y_arr[order], q_arr[order]
        
    X_train, y_train, qids_train = sort_by_qid(X_train, y_train, qids_train)
    X_val, y_val, qids_val = sort_by_qid(X_val, y_val, qids_val)
    
    ranker = xgb.XGBRanker(objective='rank:ndcg', n_estimators=500, learning_rate=0.05, max_depth=5, early_stopping_rounds=20)
    ranker.fit(X_train, y_train, group=np.unique(qids_train, return_counts=True)[1], eval_set=[(X_val, y_val)], eval_group=[np.unique(qids_val, return_counts=True)[1]], verbose=False)
    
    # 4. Evaluation (PURE TEST SET)
    evaluate_v3(df_train, df_test, emb_train, emb_test, index, ranker, id_to_idx_train)
    
    # 5. Production Re-assembly (Full ID-Mapped Index)
    logger.info("Rebuilding Full ID-Mapped Index for Production Deployment...")
    df_full = pd.concat([df_train, df_test]).reset_index(drop=True)
    emb_full = np.vstack([emb_train, emb_test])
    ids_full = df_full['listing_id'].values.astype('int64')
    
    # Final production mapping
    id_to_idx_full = {lid: i for i, lid in enumerate(df_full['listing_id'])}
    
    full_sub_index = faiss.IndexFlatIP(emb_full.shape[1])
    full_index = faiss.IndexIDMap2(full_sub_index)
    full_index.add_with_ids(emb_full.astype('float32'), ids_full)
    
    recs_df = precompute_recs_v3(df_full, emb_full, full_index, ranker, id_to_idx_full)
    
    # 6. Export
    logger.info("Saving v3.4 Production Gold Artifacts...")
    faiss.write_index(full_index, os.path.join(CONFIG["paths"]["artifacts_dir"], "faiss_v3.index"))
    ranker.save_model(os.path.join(CONFIG["paths"]["artifacts_dir"], "ranker_v3.json"))
    joblib.dump(df_full, os.path.join(CONFIG["paths"]["artifacts_dir"], "df_v3.joblib"))
    recs_df.to_csv(os.path.join(CONFIG["paths"]["artifacts_dir"], "recs_v3.csv"), index=False)
    
    logger.info("FINISH: v3.4 Production Gold Hybrid Engine Initialized.")

if __name__ == "__main__":
    main()
