import os
import logging
import numpy as np
import pandas as pd
import faiss
import joblib
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer

# ==================================================
# 1. CORE CONFIGURATION & REPRODUCIBILITY (v8.3)
# ==================================================
np.random.seed(42)

CONFIG = {
    "paths": {
        "data": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
        "artifacts": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\1_model_artifacts",
        "cache": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\1_model_artifacts\cache"
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    "features": ['price_prox', 'geo_log', 'semantic_sim', 'type_sim', 'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff', 'price_sqft_diff', 'sqft_diff']
}

PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9},
    'Chalet': {'Cabin': 0.9}
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================
# 2. CACHING UTILITIES
# ==================================================
def save_cache(obj, name):
    path = os.path.join(CONFIG["paths"]["cache"], f"v8_3_{name}")
    joblib.dump(obj, path)
    logger.info(f"Checkpoint saved: v8_3_{name}")

def load_cache(name):
    path = os.path.join(CONFIG["paths"]["cache"], f"v8_3_{name}")
    if os.path.exists(path):
        logger.info(f"Checkpoint loaded: v8_3_{name}")
        return joblib.load(path)
    return None

# ==================================================
# 3. UTILITY CLASSES
# ==================================================
class FeatureStore:
    def __init__(self, df, amenity_cols):
        self.store = df.set_index('listing_id').to_dict('index')
        self.amenity_cols = amenity_cols

    def get(self, lid):
        return self.store.get(lid)

    def get_amenity_vec(self, lid):
        item = self.get(lid)
        if not item: return np.zeros(len(self.amenity_cols), dtype=np.float32)
        return np.array([item.get(c, 0) for c in self.amenity_cols], dtype=np.float32)

# ==================================================
# 4. MATHEMATICAL ENGINES
# ==================================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def normalize_query(q):
    q_copy = np.ascontiguousarray(q.copy()).astype('float32')
    if q_copy.ndim == 1: q_copy = q_copy.reshape(1, -1)
    faiss.normalize_L2(q_copy)
    return q_copy

def get_type_sim(t1, t2):
    if t1 == t2: return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(str(t1), {}).get(str(t2), 0.05)

def calculate_jaccard(q_vec, c_matrix):
    q_vec = np.asarray(q_vec).astype(np.float32).reshape(1, -1)
    c_matrix = np.asarray(c_matrix).astype(np.float32)
    intersection = np.sum(q_vec * c_matrix, axis=1)
    union = np.sum(np.maximum(q_vec, c_matrix), axis=1)
    return np.nan_to_num(intersection / (union + 1e-9))

# ==================================================
# 5. DATA PIPELINE (v8.3 HONEST DATA)
# ==================================================
def preprocess_v8_3():
    logger.info("Step 1: Ingesting & De-duplicating Data (Honest Foundation)...")
    df = pd.read_csv(CONFIG["paths"]["data"], low_memory=False)
    
    # [v8.3] Strict De-duplication to prevent leakage
    initial_len = len(df)
    df.drop_duplicates(subset=['title', 'description', 'price_egp', 'lat', 'lon'], inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} duplicate/leaked rows.")
    
    cols = ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms']
    for col in cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['listing_id', 'price_egp', 'lat', 'lon', 'area_value'])
    
    # [v8.3] Feature: Price per SQFT
    df['price_sqft'] = df['price_egp'] / (df['area_value'] + 1e-9)
    
    for col in ['bedrooms', 'bathrooms']:
        df[col] = df[col].fillna(df[col].median()).astype(int)
    
    q1, q3 = df['price_egp'].quantile([0.25, 0.75])
    limit = q3 + 1.5 * (q3 - q1)
    df = df[df['price_egp'] <= limit]
    
    amenity_cols = []
    for a in CONFIG["top_amenities"]:
        c = f"amen__{a.lower().replace(' ', '_')}"
        df[c] = df['amenities'].fillna('').astype(str).apply(lambda x: 1 if a.lower() in x.lower() else 0)
        amenity_cols.append(c)
        
    df['desc_log'] = np.log1p(df['description'].fillna('').astype(str).apply(len))
    df['listing_id'] = df['listing_id'].astype(str)
    
    # [v8.3] Ensure district has no NaNs to prevent GroupShuffleSplit crash
    df['district'] = df['district'].fillna('Unknown')
    
    return df.reset_index(drop=True), amenity_cols

# ==================================================
# 6. RELEVANCE TEACHER (v8.3 NOISY REALITY)
# ==================================================
def get_spm_labels(query, candidates_df, g_dist, a_jac):
    """v8.3 Teacher: Adds gaussian noise to thresholds to simulate human variance."""
    if len(candidates_df) == 0: return np.array([])
    
    q_p, c_p = query['price_egp'], candidates_df['price_egp'].values
    p_err = np.abs(c_p - q_p) / (q_p + 1e-9)
    
    # [v8.3] Add 5% Noise to price error to prevent trivial formula exploitation
    noise = np.random.normal(1.0, 0.05, size=len(p_err))
    p_err_noisy = p_err * noise
    
    t_sim = np.array([get_type_sim(query['property_type'], t) for t in candidates_df['property_type'].values])
    same_city = (candidates_df['city'].values == query['city']).astype(int)
    
    rel = np.zeros(len(candidates_df))
    # Thresholds are now "Soft" because of the p_err_noisy
    rel[(same_city == 1) & (t_sim >= 0.8) & (p_err_noisy <= 0.15) & (a_jac >= 0.4) & (g_dist <= 3.0)] = 4
    rel[(rel == 0) & (same_city == 1) & (p_err_noisy <= 0.25) & (a_jac >= 0.2) & (g_dist <= 7.0)] = 3
    rel[(rel == 0) & (same_city == 1) & (p_err_noisy <= 0.4)] = 2
    rel[(rel == 0) & (p_err_noisy <= 0.6)] = 1
    return rel.astype(int)

# ==================================================
# 7. MAIN ORCHESTRATION (v8.3 HONEST SPLIT)
# ==================================================
def main():
    os.makedirs(CONFIG["paths"]["cache"], exist_ok=True)
    
    # --- STAGE 1: PREPROCESS & SPLIT ---
    cache_1 = load_cache("stage1_data.joblib")
    if cache_1:
        df_train, df_test, emb_train, emb_test, amen_cols = cache_1
    else:
        df, amen_cols = preprocess_v8_3()
        
        # [v8.3] District-Based Group Split (Honest Evaluation)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['district']))
        df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
        
        logger.info(f"Split Summary: Train Districts={df_train['district'].nunique()}, Test Districts={df_test['district'].nunique()}")
        
        model = SentenceTransformer(CONFIG["model_id"])
        emb_train = model.encode((df_train['title'] + " " + df_train['description']).tolist(), normalize_embeddings=True, show_progress_bar=True)
        emb_test = model.encode((df_test['title'] + " " + df_test['description']).tolist(), normalize_embeddings=True, show_progress_bar=True)
        save_cache((df_train, df_test, emb_train, emb_test, amen_cols), "stage1_data.joblib")

    # --- STAGE 2: FAISS ---
    index_path = os.path.join(CONFIG["paths"]["cache"], "v8_3_stage2.index")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_train.shape[1]))
        index.add_with_ids(emb_train.astype('float32'), np.arange(len(df_train)).astype('int64'))
        faiss.write_index(index, index_path)
    
    store_train = FeatureStore(df_train, amen_cols)
    
    # --- STAGE 3: TRAINING DATA ---
    cache_3 = load_cache("stage3_vectors.joblib")
    if cache_3:
        X, y, qids = cache_3
    else:
        logger.info("Step 2: Building LTR Training Dataset (Honest Reality)...")
        X, y, qids = [], [], []
        # Sample queries from each district proportionally
        q_idxs = np.random.choice(len(df_train), min(1000, len(df_train)), replace=False)
        for qid, q_idx in enumerate(tqdm(q_idxs)):
            query = df_train.iloc[q_idx]
            _, ids = index.search(normalize_query(emb_train[q_idx]), 150)
            valid_row_idxs = [int(lid) for lid in ids[0] if lid >= 0 and df_train.iloc[int(lid)]['listing_id'] != query['listing_id']]
            if not valid_row_idxs: continue
            cands, c_embs = df_train.iloc[valid_row_idxs], emb_train[valid_row_idxs]
            
            # Features
            p_p = np.log1p(np.abs(cands['price_egp'] - query['price_egp']) / (query['price_egp'] + 1e-9))
            g_l = np.log1p(haversine_vectorized(query['lat'], query['lon'], cands['lat'].values, cands['lon'].values))
            s_s = np.dot(c_embs, emb_train[q_idx])
            t_s = np.array([get_type_sim(query['property_type'], t) for t in cands['property_type'].values])
            a_j = calculate_jaccard(store_train.get_amenity_vec(query['listing_id']), np.stack([store_train.get_amenity_vec(lid) for lid in cands['listing_id']]))
            d_l = np.abs(cands['desc_log'] - query['desc_log'])
            b1, b2 = np.abs(cands['bedrooms'] - query['bedrooms']), np.abs(cands['bathrooms'] - query['bathrooms'])
            
            # [v8.3] Enriched Features
            diff_sqft = np.log1p(np.abs(cands['area_value'] - query['area_value']) / (query['area_value'] + 1e-9))
            diff_price_sqft = np.log1p(np.abs(cands['price_sqft'] - query['price_sqft']) / (query['price_sqft'] + 1e-9))
            
            X_batch = np.stack([p_p, g_l, s_s, t_s, a_j, d_l, b1, b2, diff_price_sqft, diff_sqft], axis=1)
            y_batch = get_spm_labels(query, cands, g_dist=np.expm1(g_l), a_jac=a_j)
            
            X.extend(X_batch); y.extend(y_batch); qids.extend([qid]*len(y_batch))
        X, y, qids = np.array(X), np.array(y), np.array(qids)
        save_cache((X, y, qids), "stage3_vectors.joblib")

    # --- STAGE 4: TRAINING ---
    logger.info("Step 3: Training Honest XGBRanker...")
    uq = np.unique(qids)
    tq, vq = train_test_split(uq, test_size=0.15, random_state=42)
    m_t, m_v = np.isin(qids, tq), np.isin(qids, vq)
    X_t, y_t, q_t = X[m_t], y[m_t], qids[m_t]
    X_v, y_v, q_v = X[m_v], y[m_v], qids[m_v]
    o_t, o_v = np.argsort(q_t), np.argsort(q_v)
    X_t, y_t, q_t, X_v, y_v, q_v = X_t[o_t], y_t[o_t], q_t[o_t], X_v[o_v], y_v[o_v], q_v[o_v]
    
    ranker = xgb.XGBRanker(objective='rank:ndcg', n_estimators=500, learning_rate=0.05, early_stopping_rounds=20, tree_method="hist")
    ranker.fit(X_t, y_t, group=np.unique(q_t, return_counts=True)[1], eval_set=[(X_v, y_v)], eval_group=[np.unique(q_v, return_counts=True)[1]], verbose=10)
    
    # --- STAGE 5: nDCG EVALUATION ---
    logger.info("Step 4: nDCG Evaluation (District Cross-Validation)...")
    eval_n = []
    for test_idx in np.random.choice(len(df_test), 100, replace=False):
        q_row = df_test.iloc[test_idx]
        _, ids = index.search(normalize_query(emb_test[test_idx]), 150)
        v_rows = [int(lid) for lid in ids[0] if lid >= 0]
        if not v_rows: continue
        cands, c_embs = df_train.iloc[v_rows], emb_train[v_rows]
        
        p_p = np.log1p(np.abs(cands['price_egp'] - q_row['price_egp'])/(q_row['price_egp']+1e-9))
        g_l = np.log1p(haversine_vectorized(q_row['lat'], q_row['lon'], cands['lat'].values, cands['lon'].values))
        s_s = np.dot(c_embs, emb_test[test_idx])
        t_s = np.array([get_type_sim(q_row['property_type'], t) for t in cands['property_type'].values])
        a_j = calculate_jaccard(np.array([q_row[c] for c in amen_cols], dtype=np.float32), np.stack([store_train.get_amenity_vec(lid) for lid in cands['listing_id']]))
        d_l = np.abs(cands['desc_log'] - q_row['desc_log'])
        b1, b2 = np.abs(cands['bedrooms'] - q_row['bedrooms']), np.abs(cands['bathrooms'] - q_row['bathrooms'])
        diff_sqft = np.log1p(np.abs(cands['area_value'] - q_row['area_value']) / (q_row['area_value'] + 1e-9))
        diff_p_sqft = np.log1p(np.abs(cands['price_sqft'] - q_row['price_sqft']) / (q_row['price_sqft'] + 1e-9))
        
        preds = ranker.predict(np.stack([p_p, g_l, s_s, t_s, a_j, d_l, b1, b2, diff_p_sqft, diff_sqft], axis=1))
        rel_map = dict(zip(cands['listing_id'], get_spm_labels(q_row, cands, g_dist=np.expm1(g_l), a_jac=a_j)))
        div_rel = [rel_map[lid] for lid in cands.iloc[np.argsort(preds)[::-1]]['listing_id'][:10]]
        dcg = lambda r: np.sum((2**np.array(r)-1)/np.log2(np.arange(2, len(r)+2)))
        eval_n.append(dcg(div_rel)/(dcg(np.sort(list(rel_map.values()))[::-1][:10]) + 1e-9))
        
    logger.info(f"Honest nDCG@10 (Cross-District): {np.mean(eval_n):.4f}")
    logger.info(f"Feature Importances: {dict(zip(CONFIG['features'], ranker.feature_importances_))}")
    
    # --- STAGE 6: EXPORT ---
    logger.info("Step 5: Exporting Final v8.3 Master Artifacts...")
    joblib.dump(pd.concat([df_train, df_test]).reset_index(drop=True), os.path.join(CONFIG["paths"]["artifacts"], "final_v8_3_df.joblib"))
    ranker.save_model(os.path.join(CONFIG["paths"]["artifacts"], "final_v8_3_ranker.json"))
    logger.info("PROJECT COMPLETE: Honest Reality Protocol (v8.3) Active.")

if __name__ == "__main__":
    main()
