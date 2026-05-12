import os
import logging
import numpy as np
import pandas as pd
import faiss
import joblib
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# ==================================================
# 1. CORE CONFIGURATION & REPRODUCIBILITY
# ==================================================
np.random.seed(42)

CONFIG = {
    "paths": {
        "data": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
        "artifacts": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts_final"
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    "features": ['price_prox', 'geo_log', 'semantic_sim', 'type_sim', 'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff']
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
# 2. UTILITY CLASSES (FROM v4 SAFETY)
# ==================================================
class FeatureStore:
    """Enterprise feature vault: ID-safe attribute retrieval."""
    def __init__(self, df, amenity_cols):
        self.store = df.set_index('listing_id').to_dict('index')
        self.amenity_cols = amenity_cols

    def get(self, lid):
        return self.store.get(int(lid))

    def get_amenity_vec(self, lid):
        item = self.get(lid)
        if not item: return np.zeros(len(self.amenity_cols))
        return np.array([item[c] for c in self.amenity_cols])

# ==================================================
# 3. MATHEMATICAL ENGINES (FROM v3.4 SUPERIORITY)
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
    """Broadcasting-safe multi-candidate Jaccard."""
    q_vec = q_vec.reshape(1, -1)
    intersection = np.sum(q_vec * c_matrix, axis=1)
    union = np.sum(np.maximum(q_vec, c_matrix), axis=1)
    return intersection / (union + 1e-9)

# ==================================================
# 4. DATA PIPELINE
# ==================================================
def preprocess_v5():
    logger.info("Step 1: Ingesting & Sanitizing Data (v5.0 Union)...")
    df = pd.read_csv(CONFIG["paths"]["data"], low_memory=False)
    
    # Numeric & Drop
    cols = ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms']
    for col in cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['listing_id', 'price_egp', 'lat', 'lon'])
    
    # Imputation
    for col in ['bedrooms', 'bathrooms']:
        df[col] = df[col].fillna(df[col].median()).astype(int)
    
    # Outlier Clipping
    upper = df['price_egp'].quantile(0.75) + 1.5 * (df['price_egp'].quantile(0.75) - df['price_egp'].quantile(0.25))
    df = df[df['price_egp'] <= upper]
    
    # Amenities Matrix
    amenity_cols = []
    for a in CONFIG["top_amenities"]:
        c = f"amen__{a.lower().replace(' ', '_')}"
        df[c] = df['amenities'].fillna('').apply(lambda x: 1 if a.lower() in str(x).lower() else 0)
        amenity_cols.append(c)
        
    # Log-scaling (Essential Signal Stability)
    df['desc_log'] = np.log1p(df['description'].fillna('').apply(len))
    df['listing_id'] = df['listing_id'].astype(int)
    
    return df.reset_index(drop=True), amenity_cols

# ==================================================
# 5. RELEVANCE TEACHER (SPM)
# ==================================================
def get_spm_labels(query, candidates_df):
    q_p = query['price_egp']
    c_p = candidates_df['price_egp'].values
    p_err = np.abs(c_p - q_p) / (q_p + 1e-9)
    
    t_sim = np.array([get_type_sim(query['property_type'], t) for t in candidates_df['property_type'].values])
    same_city = (candidates_df['city'].values == query['city']).astype(int)
    
    rel = np.zeros(len(candidates_df))
    rel[(same_city == 1) & (t_sim >= 1.0) & (p_err <= 0.1)] = 4
    rel[(rel == 0) & (same_city == 1) & (t_sim >= 0.8) & (p_err <= 0.25)] = 3
    rel[(rel == 0) & (same_city == 1) & (p_err <= 0.4)] = 2
    rel[(rel == 0) & (p_err <= 0.6)] = 1
    return rel.astype(int)

# ==================================================
# 6. HYBRID SEARCH & DIVERSITY
# ==================================================
def apply_mmr_rerank(df_recs, top_n=10):
    """Diverse results: deduplicate buildings, cap districts."""
    final = []
    seen_b, seen_d = set(), {}
    for _, row in df_recs.iterrows():
        bid, did = row.get('building_name', row['listing_id']), row.get('district', 'unknown')
        d_count = seen_d.get(did, 0)
        if bid not in seen_b and d_count < 3:
            final.append(row)
            seen_b.add(bid)
            seen_d[did] = d_count + 1
        if len(final) >= top_n: break
    return pd.DataFrame(final)

# ==================================================
# 7. MODEL TRAINING
# ==================================================
def main():
    os.makedirs(CONFIG["paths"]["artifacts"], exist_ok=True)
    df, amen_cols = preprocess_v5()
    
    # 1. Split & Encode
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    
    model = SentenceTransformer(CONFIG["model_id"])
    emb_train = model.encode((df_train['title'] + " " + df_train['description']).tolist(), normalize_embeddings=True, show_progress_bar=True)
    emb_test = model.encode((df_test['title'] + " " + df_test['description']).tolist(), normalize_embeddings=True, show_progress_bar=True)
    
    # 2. Build ID-Mapped Index
    sub_idx = faiss.IndexFlatIP(emb_train.shape[1])
    index = faiss.IndexIDMap2(sub_idx)
    index.add_with_ids(emb_train.astype('float32'), df_train['listing_id'].values.astype('int64'))
    
    store_train = FeatureStore(df_train, amen_cols)
    id_to_row_train = {lid: i for i, lid in enumerate(df_train['listing_id'])}
    
    # 3. Training Data Gen
    logger.info("Step 2: Building LTR Training Dataset (Triple-Negative)...")
    X, y, qids = [], [], []
    q_idxs = np.random.choice(len(df_train), 400, replace=False)
    
    for qid, q_idx in enumerate(tqdm(q_idxs)):
        query = df_train.iloc[q_idx]
        q_emb = normalize_query(emb_train[q_idx])
        
        _, ids = index.search(q_emb, 100)
        valid_row_idxs = [id_to_row_train[lid] for lid in ids[0] if lid in id_to_row_train]
        
        cands = df_train.iloc[valid_row_idxs]
        c_embs = emb_train[valid_row_idxs]
        
        # Features
        p_prox = np.log1p(np.abs(cands['price_egp'].values - query['price_egp']) / (query['price_egp'] + 1e-9))
        g_log = np.log1p(haversine_vectorized(query['lat'], query['lon'], cands['lat'].values, cands['lon'].values))
        s_sim = np.dot(c_embs, emb_train[q_idx])
        t_sim = np.array([get_type_sim(query['property_type'], t) for t in cands['property_type'].values])
        a_jac = calculate_jaccard(store_train.get_amenity_vec(query['listing_id']), np.stack([store_train.get_amenity_vec(lid) for lid in cands['listing_id']]))
        d_log_diff = np.abs(cands['desc_log'].values - query['desc_log'])
        b1_diff = np.abs(cands['bedrooms'].values - query['bedrooms'])
        b2_diff = np.abs(cands['bathrooms'].values - query['bathrooms'])
        
        X_batch = np.stack([p_prox, g_log, s_sim, t_sim, a_jac, d_log_diff, b1_diff, b2_diff], axis=1)
        y_batch = get_spm_labels(query, cands)
        
        X.extend(X_batch); y.extend(y_batch); qids.extend([qid]*len(y_batch))
        
    X, y, qids = np.array(X), np.array(y), np.array(qids)
    
    # 4. Train Ranker (Strict QID Sorting)
    logger.info("Step 3: Training XGBRanker (Group-Aware)...")
    uq = np.unique(qids)
    tq, vq = train_test_split(uq, test_size=0.15, random_state=42)
    m_t, m_v = np.isin(qids, tq), np.isin(qids, vq)
    
    X_t, y_t, q_t = X[m_t], y[m_t], qids[m_t]
    X_v, y_v, q_v = X[m_v], y[m_v], qids[m_v]
    
    # Sort
    o_t, o_v = np.argsort(q_t), np.argsort(q_v)
    X_t, y_t, q_t = X_t[o_t], y_t[o_t], q_t[o_t]
    X_v, y_v, q_v = X_v[o_v], y_v[o_v], q_v[o_v]
    
    ranker = xgb.XGBRanker(objective='rank:ndcg', n_estimators=500, learning_rate=0.05, early_stopping_rounds=20)
    ranker.fit(X_t, y_t, group=np.unique(q_t, return_counts=True)[1], eval_set=[(X_v, y_v)], eval_group=[np.unique(q_v, return_counts=True)[1]], verbose=False)
    
    # 5. Evaluation
    logger.info("Step 4: nDCG Evaluation Phase...")
    eval_n = []
    for test_idx in np.random.choice(len(df_test), 50, replace=False):
        q_row = df_test.iloc[test_idx]
        q_vec = normalize_query(emb_test[test_idx])
        _, ids = index.search(q_vec, 100)
        v_rows = [id_to_row_train[lid] for lid in ids[0] if lid in id_to_row_train]
        cands = df_train.iloc[v_rows]
        c_embs = emb_train[v_rows]
        
        # Features
        p_p = np.log1p(np.abs(cands['price_egp'].values - q_row['price_egp'])/(q_row['price_egp']+1e-9))
        g_l = np.log1p(haversine_vectorized(q_row['lat'], q_row['lon'], cands['lat'].values, cands['lon'].values))
        s_s = np.dot(c_embs, emb_test[test_idx])
        t_s = np.array([get_type_sim(q_row['property_type'], t) for t in cands['property_type'].values])
        a_j = calculate_jaccard(np.array([q_row[c] for c in amen_cols]), np.stack([store_train.get_amenity_vec(lid) for lid in cands['listing_id']]))
        d_l = np.abs(cands['desc_log'].values - q_row['desc_log'])
        b1 = np.abs(cands['bedrooms'].values - q_row['bedrooms'])
        b2 = np.abs(cands['bathrooms'].values - q_row['bathrooms'])
        
        X_e = np.stack([p_p, g_l, s_s, t_s, a_j, d_l, b1, b2], axis=1)
        preds = ranker.predict(X_e)
        sorted_cands = cands.iloc[np.argsort(preds)[::-1]]
        div_top = apply_mmr_rerank(sorted_cands, 10)
        
        rel_map = dict(zip(cands['listing_id'], get_spm_labels(q_row, cands)))
        div_rel = [rel_map[lid] for lid in div_top['listing_id']]
        
        dcg = lambda r: np.sum((2**np.array(r)-1)/np.log2(np.arange(2, len(r)+2)))
        eval_n.append(dcg(div_rel)/ (dcg(np.sort(list(rel_map.values()))[::-1][:10]) + 1e-9))
    
    logger.info(f"--- Enterprise v5.0 Final Certification ---")
    logger.info(f"nDCG@10 (Diverse): {np.mean(eval_n):.4f}")
    logger.info(f"Feature Importance: {dict(zip(CONFIG['features'], ranker.feature_importances_))}")
    
    # 6. Final Production Export
    logger.info("Step 5: Exporting Production Artifacts...")
    full_df = pd.concat([df_train, df_test]).reset_index(drop=True)
    full_emb = np.vstack([emb_train, emb_test])
    full_sub = faiss.IndexFlatIP(full_emb.shape[1])
    full_idx = faiss.IndexIDMap2(full_sub)
    full_idx.add_with_ids(full_emb.astype('float32'), full_df['listing_id'].values.astype('int64'))
    
    faiss.write_index(full_idx, os.path.join(CONFIG["paths"]["artifacts"], "final_index.faiss"))
    ranker.save_model(os.path.join(CONFIG["paths"]["artifacts"], "final_ranker.json"))
    joblib.dump(full_df, os.path.join(CONFIG["paths"]["artifacts"], "final_df.joblib"))
    
    logger.info("PROJECT COMPLETE: v5.0 Final Union Ready for Deployment.")

if __name__ == "__main__":
    main()
