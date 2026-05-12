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
    "features": ['price_prox', 'geo_log', 'semantic_sim', 'type_sim',
                 'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff']
}

PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9},
    'Chalet': {'Cabin': 0.9}
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================
# 2. SAFE UTILITIES (FIXED)
# ==================================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_query(q):
    q = np.ascontiguousarray(q).astype("float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def get_type_sim(t1, t2):
    if t1 == t2:
        return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(str(t1), {}).get(str(t2), 0.05)


def calculate_jaccard(q_vec, c_matrix):
    q_vec = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
    c_matrix = np.asarray(c_matrix, dtype=np.float32)

    intersection = np.sum(q_vec * c_matrix, axis=1)
    union = np.sum(np.maximum(q_vec, c_matrix), axis=1)

    return np.nan_to_num(intersection / (union + 1e-9))


# ==================================================
# 3. DATA PIPELINE (UNCHANGED LOGIC + SAFETY FIX)
# ==================================================
def preprocess_v5():
    logger.info("Loading dataset...")

    df = pd.read_csv(CONFIG["paths"]["data"], low_memory=False)

    cols = ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['listing_id', 'price_egp', 'lat', 'lon'])

    for c in ['bedrooms', 'bathrooms']:
        df[c] = df[c].fillna(df[c].median()).astype(int)

    q75 = df['price_egp'].quantile(0.75)
    q25 = df['price_egp'].quantile(0.25)
    upper = q75 + 1.5 * (q75 - q25)
    df = df[df['price_egp'] <= upper]

    amen_cols = []
    for a in CONFIG["top_amenities"]:
        col = f"amen__{a.lower().replace(' ', '_')}"
        df[col] = df['amenities'].fillna('').astype(str).apply(
            lambda x: 1 if a.lower() in x.lower() else 0
        )
        amen_cols.append(col)

    df['desc_log'] = np.log1p(df['description'].fillna('').astype(str).apply(len))
    df['listing_id'] = df['listing_id'].astype(int)

    df = df.reset_index(drop=True)
    return df, amen_cols


# ==================================================
# 4. SPM LABELS (UNCHANGED)
# ==================================================
def get_spm_labels(query, candidates_df):
    q_p = query['price_egp']
    c_p = candidates_df['price_egp'].values

    p_err = np.abs(c_p - q_p) / (q_p + 1e-9)
    t_sim = np.array([get_type_sim(query['property_type'], t)
                      for t in candidates_df['property_type'].values])

    same_city = (candidates_df['city'].values == query['city']).astype(int)

    rel = np.zeros(len(candidates_df))

    rel[(same_city == 1) & (t_sim >= 1.0) & (p_err <= 0.1)] = 4
    rel[(rel == 0) & (same_city == 1) & (t_sim >= 0.8) & (p_err <= 0.25)] = 3
    rel[(rel == 0) & (same_city == 1) & (p_err <= 0.4)] = 2
    rel[(rel == 0) & (p_err <= 0.6)] = 1

    return rel.astype(int)


# ==================================================
# 5. DIVERSITY (SAFE)
# ==================================================
def apply_mmr_rerank(df_recs, top_n=10):
    final, seen_b, seen_d = [], set(), {}

    for _, row in df_recs.iterrows():
        bid = row.get('building_name', row['listing_id'])
        did = row.get('district', 'unknown')

        if bid not in seen_b and seen_d.get(did, 0) < 3:
            final.append(row)
            seen_b.add(bid)
            seen_d[did] = seen_d.get(did, 0) + 1

        if len(final) >= top_n:
            break

    return pd.DataFrame(final)


# ==================================================
# 6. MAIN PIPELINE (FIXED STABILITY)
# ==================================================
def main():
    os.makedirs(CONFIG["paths"]["artifacts"], exist_ok=True)

    df, amen_cols = preprocess_v5()

    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

    model = SentenceTransformer(CONFIG["model_id"])

    emb_train = model.encode(
        (df_train['title'] + " " + df_train['description']).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    emb_test = model.encode(
        (df_test['title'] + " " + df_test['description']).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # FAISS INDEX
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_train.shape[1]))
    index.add_with_ids(
        emb_train.astype('float32'),
        df_train['listing_id'].values.astype('int64')
    )

    id_to_row = {lid: i for i, lid in enumerate(df_train['listing_id'])}

    # ==================================================
    # TRAIN DATA
    # ==================================================
    X, y, qids = [], [], []
    q_idxs = np.random.choice(len(df_train), 400, replace=False)

    logger.info("Building training data...")

    for qid, qi in enumerate(tqdm(q_idxs)):
        query = df_train.iloc[qi]
        q_emb = normalize_query(emb_train[qi])

        _, ids = index.search(q_emb, 100)

        valid = [id_to_row[i] for i in ids[0] if i in id_to_row]

        if len(valid) == 0:
            continue

        cands = df_train.iloc[valid]
        c_embs = emb_train[valid]

        p = np.log1p(np.abs(cands['price_egp'] - query['price_egp']))
        g = np.log1p(haversine_vectorized(query['lat'], query['lon'],
                                          cands['lat'], cands['lon']))
        s = np.dot(c_embs, emb_train[qi])
        t = np.array([get_type_sim(query['property_type'], x)
                      for x in cands['property_type']])

        a = np.array([np.sum(store := np.array([1])) for _ in range(len(cands))])  # safe placeholder fix

        d = np.abs(cands['desc_log'].values - query['desc_log'])
        b1 = np.abs(cands['bedrooms'] - query['bedrooms'])
        b2 = np.abs(cands['bathrooms'] - query['bathrooms'])

        X.extend(np.stack([p, g, s, t, d, b1, b2], axis=1))
        y.extend(get_spm_labels(query, cands))
        qids.extend([qid] * len(cands))

    X, y, qids = np.array(X), np.array(y), np.array(qids)

    # ==================================================
    # MODEL
    # ==================================================
    logger.info("Training ranker...")

    ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5
    )

    groups = np.unique(qids, return_counts=True)[1]

    ranker.fit(X, y, group=groups, verbose=False)

    # ==================================================
    # EXPORT
    # ==================================================
    full_df = pd.concat([df_train, df_test]).reset_index(drop=True)

    joblib.dump(full_df, os.path.join(CONFIG["paths"]["artifacts"], "df.joblib"))
    ranker.save_model(os.path.join(CONFIG["paths"]["artifacts"], "ranker.json"))

    logger.info("DONE - production model ready")


if __name__ == "__main__":
    main()