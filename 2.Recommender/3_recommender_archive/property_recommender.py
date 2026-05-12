import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib
import os
import logging
from tqdm import tqdm

# ================= CONFIG ================= #
CONFIG = {
    "weights": {
        "structural": 0.30,
        "price": 0.20,
        "semantic": 0.20,
        "geo": 0.15,
        "type": 0.15
    },
    "feature_weights": np.array([2.5, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
    "knn_neighbors": 50,
    "epsilon": 1e-9
}

RAW_DATA_PATH = r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv"
ARTIFACTS_DIR = r"C:\Users\Adham\Desktop\AI_System\2.Recommender\model_artifacts"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= UTIL ================= #
def normalize(v):
    return v / (np.linalg.norm(v) + CONFIG["epsilon"])

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ================= TYPE SIMILARITY ================= #
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

def get_type_similarity(a, b):
    if a == b:
        return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(a, {}).get(b, 0.0)

# ================= RANKING ENGINE ================= #
def rank_candidates(query, candidates, q_emb, c_embs, knn_dists):

    structural = np.exp(-knn_dists)

    semantic = np.dot(
        np.array([normalize(e) for e in c_embs]),
        normalize(q_emb)
    )

    geo = np.exp(
        -haversine_vectorized(
            query["lat"], query["lon"],
            candidates["lat"].values, candidates["lon"].values
        ) / 5.0
    )

    price = np.exp(
        -np.abs(candidates["price_egp"].values - query["price_egp"]) /
        (query["price_egp"] + CONFIG["epsilon"])
    )

    type_score = np.array([
        get_type_similarity(query["property_type"], t)
        for t in candidates["property_type"]
    ])

    final = (
        CONFIG["weights"]["structural"] * structural +
        CONFIG["weights"]["price"] * price +
        CONFIG["weights"]["semantic"] * semantic +
        CONFIG["weights"]["geo"] * geo +
        CONFIG["weights"]["type"] * type_score
    )

    return np.argsort(final)[::-1]

# ================= EMBEDDINGS ================= #
def generate_embeddings(df):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = (df["title"] + " " + df["description"]).tolist()
    return model.encode(texts, batch_size=32, convert_to_numpy=True)

# ================= MODEL ================= #
def train_knn(X):
    knn = NearestNeighbors(
        n_neighbors=CONFIG["knn_neighbors"],
        metric="euclidean",
        n_jobs=-1
    )
    knn.fit(X)
    return knn

# ================= FEATURE PREP ================= #
def prepare_features(df, scaler=None, fit=False):
    cols = [
        "price_egp", "area_value", "bedrooms",
        "bathrooms", "lat", "lon", "amenities_count"
    ]

    X = df[cols].copy()
    X["price_egp"] = np.log1p(X["price_egp"])
    X["area_value"] = np.log1p(X["area_value"])

    if fit:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X = X * CONFIG["feature_weights"]
    return X, scaler

# =========================================================
#           DATA PREPROCESSING (UNCHANGED BLOCK)
# =========================================================
def load_and_prepare_data():
    logger.info(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    original_count = len(df)
    
    def clean_beds(val):
        if pd.isna(val): return np.nan
        val = str(val).lower().strip()
        if val == 'studio': return 0
        if val == '7+': return 7
        try: return float(val)
        except: return np.nan

    def clean_baths(val):
        if pd.isna(val): return np.nan
        val = str(val).lower().strip()
        if val == '7+': return 7
        try: return float(val)
        except: return np.nan

    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].apply(clean_beds)
    if 'bathrooms' in df.columns:
        df['bathrooms'] = df['bathrooms'].apply(clean_baths)

    for col in ['price_egp', 'area_value', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['listing_id', 'price_egp'])

    for col in ['lat', 'lon', 'bedrooms', 'bathrooms', 'area_value']:
        df[col] = df[col].fillna(df[col].median())

    def remove_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= (Q1 - 1.5 * IQR)) &
                    (data[col] <= (Q3 + 1.5 * IQR))]

    df = remove_outliers(df, "price_egp")
    df = remove_outliers(df, "area_value")

    df["amenities"] = df["amenities"].fillna("")
    df["amenities_count"] = df["amenities"].apply(
        lambda x: len(str(x).split("|")) if x else 0
    )

    df["description"] = df["description"].fillna("No description available")
    df["title"] = df["title"].fillna("No title available")

    df = df.reset_index(drop=True)

    logger.info(f"Preprocessing complete: {original_count} → {len(df)}")
    return df

# ================= PRECOMPUTE ================= #
def precompute(df, knn, X, embeddings):
    distances, indices = knn.kneighbors(X)
    results = []

    for i in tqdm(range(len(df))):
        query = df.iloc[i]

        neighbors = indices[i][1:]
        dists = distances[i][1:]

        candidates = df.iloc[neighbors]
        c_embs = embeddings[neighbors]

        order = rank_candidates(
            query,
            candidates,
            embeddings[i],
            c_embs,
            dists
        )

        top = candidates.iloc[order[:10]]

        results.append({
            "listing_id": query["listing_id"],
            "recommendations": ",".join(top["listing_id"].astype(str))
        })

    return pd.DataFrame(results)

# ================= MAIN ================= #
def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_and_prepare_data()
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    X_train, scaler = prepare_features(train_df, fit=True)
    knn = train_knn(X_train)

    embeddings = generate_embeddings(df)

    X_full, _ = prepare_features(df, scaler=scaler)
    recs = precompute(df, knn, X_full, embeddings)

    joblib.dump(knn, os.path.join(ARTIFACTS_DIR, "knn.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    np.save(os.path.join(ARTIFACTS_DIR, "embeddings.npy"), embeddings)
    recs.to_csv(os.path.join(ARTIFACTS_DIR, "recs.csv"), index=False)

    logger.info("DONE")

if __name__ == "__main__":
    main()