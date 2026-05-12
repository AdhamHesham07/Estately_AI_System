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
# CONFIG
# ==================================================
np.random.seed(42)

CONFIG = {
    "data_path": r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv",
    "artifacts_dir": r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts_v4",
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_k": 100
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# LOAD RAW DATA
# ==================================================
def load_data():
    return pd.read_csv(CONFIG["data_path"], low_memory=False)


# ==================================================
# PREPROCESSING (INSERTED HERE - PRODUCTION SAFE)
# ==================================================
def preprocess_data(df):
    df = df.copy()

    # 1. Numeric conversion
    numeric_cols = ["price_egp", "area_value", "lat", "lon", "bedrooms", "bathrooms"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Drop invalid rows
    df = df.dropna(subset=["listing_id", "price_egp", "lat", "lon"])

    # 3. Imputation
    for col in ["bedrooms", "bathrooms"]:
        df[col] = df[col].fillna(df[col].median())

    # 4. Outlier clipping (price)
    q1 = df["price_egp"].quantile(0.25)
    q3 = df["price_egp"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["price_egp"] <= upper]

    # 5. Text safety
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    # 6. Derived features
    df["desc_len"] = np.log1p(df["description"].str.len())

    # 7. ID safety
    df["listing_id"] = df["listing_id"].astype(int)

    return df.reset_index(drop=True)


# ==================================================
# EMBEDDINGS
# ==================================================
def build_embeddings(df, model):
    texts = (df["title"] + " " + df["description"]).tolist()

    emb = model.encode(texts, batch_size=32, normalize_embeddings=True)

    return {
        int(lid): emb[i]
        for i, lid in enumerate(df["listing_id"])
    }


# ==================================================
# FAISS INDEX
# ==================================================
def build_faiss_index(embedding_dict):
    ids = np.array(list(embedding_dict.keys())).astype("int64")
    vectors = np.array(list(embedding_dict.values())).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    faiss.normalize_L2(vectors)
    index.add_with_ids(vectors, ids)

    return index


# ==================================================
# FEATURE STORE
# ==================================================
class FeatureStore:
    def __init__(self, df):
        self.store = {
            int(row.listing_id): row
            for _, row in df.iterrows()
        }

    def get(self, lid):
        return self.store.get(int(lid))


# ==================================================
# DISTANCE
# ==================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ==================================================
# RETRIEVAL
# ==================================================
def retrieve(index, query_vec, k=100):
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, ids = index.search(query_vec, k)

    return [i for i in ids[0] if i != -1]


# ==================================================
# FEATURE BUILDING
# ==================================================
def build_features(query, candidates, feature_store, embeddings):
    X, valid_ids = [], []

    for cid in candidates:
        item = feature_store.get(cid)
        if item is None:
            continue

        X.append([
            abs(item.price_egp - query.price_egp),
            haversine(query.lat, query.lon, item.lat, item.lon),
            item.desc_len,
            abs(item.bedrooms - query.bedrooms),
            abs(item.bathrooms - query.bathrooms),
        ])
        valid_ids.append(cid)

    return np.array(X), valid_ids


# ==================================================
# LABELS
# ==================================================
def generate_labels(query, candidates, feature_store):
    labels = []

    for cid in candidates:
        item = feature_store.get(cid)
        if item is None:
            labels.append(0)
            continue

        diff = abs(item.price_egp - query.price_egp)

        if diff < query.price_egp * 0.1:
            labels.append(3)
        elif diff < query.price_egp * 0.25:
            labels.append(2)
        elif diff < query.price_egp * 0.5:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


# ==================================================
# TRAIN DATA
# ==================================================
def build_training_data(df, index, feature_store):
    model = SentenceTransformer(CONFIG["embedding_model"])
    embeddings = build_embeddings(df, model)

    X, y, qids = [], [], []

    samples = df.sample(300, random_state=42)

    for qid, row in tqdm(enumerate(samples.itertuples()), total=len(samples)):
        q_emb = embeddings[row.listing_id]

        candidates = retrieve(index, q_emb, CONFIG["top_k"])

        X_batch, valid_ids = build_features(row, candidates, feature_store, embeddings)
        y_batch = generate_labels(row, valid_ids, feature_store)

        X.extend(X_batch)
        y.extend(y_batch)
        qids.extend([qid] * len(valid_ids))

    return np.array(X), np.array(y), np.array(qids), embeddings


# ==================================================
# TRAIN RANKER
# ==================================================
def train_ranker(X, y, qids):
    unique_qids = np.unique(qids)

    train_q, val_q = train_test_split(unique_qids, test_size=0.2, random_state=42)

    train_mask = np.isin(qids, train_q)
    val_mask = np.isin(qids, val_q)

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(
        X[train_mask],
        y[train_mask],
        group=np.unique(qids[train_mask], return_counts=True)[1],
        eval_set=[(X[val_mask], y[val_mask])],
        eval_group=[np.unique(qids[val_mask], return_counts=True)[1]],
        verbose=True
    )

    return model


# ==================================================
# EVALUATION
# ==================================================
def evaluate(model, df, index, feature_store, embeddings):
    sample = df.sample(50, random_state=1)

    for row in sample.itertuples():
        q_emb = embeddings[row.listing_id]

        candidates = retrieve(index, q_emb, 50)

        X, valid_ids = build_features(row, candidates, feature_store, embeddings)

        if len(X) == 0:
            continue

        preds = model.predict(X)
        order = np.argsort(preds)[::-1]

        top = [valid_ids[i] for i in order[:10]]

        print("Top-10:", top)


# ==================================================
# MAIN PIPELINE
# ==================================================
def main():
    os.makedirs(CONFIG["artifacts_dir"], exist_ok=True)

    # STEP 1: LOAD
    df = load_data()

    # STEP 2: PREPROCESS (INSERTED HERE)
    df = preprocess_data(df)

    # STEP 3: MODEL + INDEX
    model = SentenceTransformer(CONFIG["embedding_model"])
    embedding_dict = build_embeddings(df, model)

    index = build_faiss_index(embedding_dict)

    feature_store = FeatureStore(df)

    # STEP 4: TRAINING DATA
    X, y, qids, embeddings = build_training_data(df, index, feature_store)

    # STEP 5: TRAIN
    ranker = train_ranker(X, y, qids)

    # STEP 6: EVAL
    evaluate(ranker, df, index, feature_store, embedding_dict)

    # STEP 7: SAVE
    faiss.write_index(index, os.path.join(CONFIG["artifacts_dir"], "faiss.index"))
    joblib.dump(ranker, os.path.join(CONFIG["artifacts_dir"], "ranker.pkl"))
    joblib.dump(df, os.path.join(CONFIG["artifacts_dir"], "df.pkl"))

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()