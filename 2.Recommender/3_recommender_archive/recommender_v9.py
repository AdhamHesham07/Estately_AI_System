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

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & SEEDING
# ---------------------------------------------------------
# Set the random seed to ensure that our results are reproducible
np.random.seed(42)

BASE_DIR = r"C:\Users\Adham\Desktop\AI_System"

# Configuration dictionary to easily manage file paths, model versions, and features
CONFIG = {
    "paths": {
        "data": os.path.join(BASE_DIR, "5.Data", "propertyfinder.csv"),
        "artifacts": os.path.join(BASE_DIR, "2.Recommender", "1_model_artifacts"),
        "cache": os.path.join(BASE_DIR, "2.Recommender", "1_model_artifacts", "cache")
    },
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    "features": [
        'price_prox', 'geo_log', 'semantic_sim', 'type_sim', 
        'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff'
    ]
}

# A heuristic matrix defining the similarity score between different types of properties.
PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9},
    'Chalet': {'Cabin': 0.9}
}

# Initialize logging to track pipeline progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# CACHING UTILITIES
# =========================================================
# These functions allow us to save and load intermediate stages (like embeddings) 
# so we don't have to rerun expensive calculations if the script crashes.

def save_cache(obj, name):
    """Saves a python object to disk using joblib."""
    path = os.path.join(CONFIG["paths"]["cache"], name)
    joblib.dump(obj, path)

def load_cache(name):
    """Loads a python object from disk if it exists, otherwise returns None."""
    path = os.path.join(CONFIG["paths"]["cache"], name)
    return joblib.load(path) if os.path.exists(path) else None

# =========================================================
# FEATURE STORE
# =========================================================
class FeatureStore:
    """
    A lightning-fast dictionary-based lookup for property features.
    Instead of using Pandas .loc inside loops (which is slow), we convert the dataframe 
    to a dictionary for O(1) attribute retrieval.
    """
    def __init__(self, dataframe, amenity_columns):
        self.store = dataframe.set_index('listing_id').to_dict('index')
        self.amenity_columns = amenity_columns

    def get(self, listing_id):
        """Retrieves all attributes of a specific listing ID."""
        return self.store.get(listing_id)

    def get_amenity_vec(self, listing_id):
        """Builds a binary numpy array representing the presence of amenities for a listing."""
        item = self.get(listing_id)
        if not item:
            return np.zeros(len(self.amenity_columns), dtype=np.float32)
        return np.array([item.get(col, 0) for col in self.amenity_columns], dtype=np.float32)

# =========================================================
# MATHEMATICAL AND DISTANCE ENGINES
# =========================================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculates the physical distance (in kilometers) between GPS coordinates."""
    earth_radius = 6371 # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * earth_radius * np.arcsin(np.sqrt(a))

def normalize_query(query_vector):
    """Ensures that query vectors are L2 normalized, required for correct FAISS Cosine distance."""
    query_vector = np.ascontiguousarray(query_vector.astype('float32').reshape(1, -1))
    faiss.normalize_L2(query_vector)
    return query_vector

def get_type_sim(type1, type2):
    """Looks up the similarity between two property types using the heuristic matrix."""
    if type1 == type2:
        return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(str(type1), {}).get(str(type2), 0.05)

def calculate_jaccard(query_vector, candidate_matrix):
    """
    Calculates the Jaccard similarity (Intersection over Union) 
    between a query's amenities and a matrix of candidate amenities.
    """
    query_vector = query_vector.astype(np.float32).reshape(1, -1)
    candidate_matrix = candidate_matrix.astype(np.float32)
    
    intersection = np.sum(query_vector * candidate_matrix, axis=1)
    union = np.sum(np.maximum(query_vector, candidate_matrix), axis=1)
    
    # Avoid division by zero
    return np.nan_to_num(intersection / (union + 1e-9))

# =========================================================
# DATA PREPROCESSING PIPELINE
# =========================================================
def preprocess():
    """
    Loads raw data, cleans columns, handles missing values, removes extreme outliers,
    and engineers basic features necessary for the model.
    """
    df = pd.read_csv(CONFIG["paths"]["data"], low_memory=False)

    numeric_columns = ['price_egp', 'area_value', 'lat', 'lon', 'bedrooms', 'bathrooms']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are missing critical geographic or identity information
    df = df.dropna(subset=['listing_id', 'price_egp', 'lat', 'lon'])

    # Fill missing bedroom/bathroom values with the median value of the valid dataset
    for col in ['bedrooms', 'bathrooms']:
        df[col] = df[col].fillna(df[col].median()).astype(int)

    # Remove price outliers (properties sitting in the ultra-luxury fringe that skew metrics)
    price_q1, price_q3 = df['price_egp'].quantile([0.25, 0.75])
    df = df[df['price_egp'] <= price_q3 + 1.5 * (price_q3 - price_q1)]

    # One-hot encode the selected top amenities
    amenity_cols = []
    for amenity_name in CONFIG["top_amenities"]:
        safe_col_name = f"amen__{amenity_name.lower().replace(' ', '_')}"
        # Mark 1 if the amenity string is found, otherwise 0
        df[safe_col_name] = df['amenities'].fillna('').astype(str).apply(
            lambda raw_str: int(amenity_name.lower() in raw_str.lower())
        )
        amenity_cols.append(safe_col_name)

    # Create a feature tracking how detailed the description is (log scaled)
    df['desc_log'] = np.log1p(df['description'].fillna('').astype(str).apply(len))

    # [NOTE]: Replaced astype(int) with astype(str) to prevent ValueError crashes on alphanumeric IDs
    df['listing_id'] = df['listing_id'].astype(str)  

    return df.reset_index(drop=True), amenity_cols

# =========================================================
# RELEVANCE LABEL GENERATOR (THE TEACHER)
# =========================================================
def get_spm_labels(query_row, candidates_df, geo_distance, amenity_jaccard):
    """
    Generates deterministic target labels ('Relevance Scores' 1 to 4) 
    that the XGBoost model will try to learn and predict.
    """
    if len(candidates_df) == 0:
        return np.array([])

    query_price = query_row['price_egp']
    candidate_prices = candidates_df['price_egp'].values

    # Calculate absolute relative price error
    price_error = np.abs(candidate_prices - query_price) / (query_price + 1e-9)
    
    # Calculate property type similarity via matrix lookup
    type_similarity = np.array([get_type_sim(query_row['property_type'], t) for t in candidates_df['property_type']])
    
    # Determine if they are in the exact same city
    same_city = (candidates_df['city'].values == query_row['city']).astype(int)

    relevance_scores = np.zeros(len(candidates_df))

    # Level 4: Elite Match (Price within 15%, highly similar Type & Amenities, within 3km)
    relevance_scores[
        (same_city == 1) & (type_similarity >= 0.8) & (price_error <= 0.15) & 
        (amenity_jaccard >= 0.4) & (geo_distance <= 3)
    ] = 4
    
    # Level 3: Strong Match (Price within 25%, decent amenities, within 7km)
    relevance_scores[
        (relevance_scores == 0) & (same_city == 1) & (price_error <= 0.25) & 
        (amenity_jaccard >= 0.2) & (geo_distance <= 7)
    ] = 3
    
    # Level 2: Relevant Match (Price within 40%, same city)
    relevance_scores[
        (relevance_scores == 0) & (same_city == 1) & (price_error <= 0.4)
    ] = 2
    
    # Level 1: Discovery/Weak Match (Price within 60%)
    relevance_scores[
        (relevance_scores == 0) & (price_error <= 0.6)
    ] = 1

    return relevance_scores.astype(int)

# =========================================================
# MAIN PIPELINE ORCHESTRATION
# =========================================================
def main():
    # Ensure cache directory exists
    os.makedirs(CONFIG["paths"]["cache"], exist_ok=True)

    # 1. Prepare and Split Data
    df, amenity_cols = preprocess()
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)

    # 2. Encode Property Descriptions using HuggingFace Language Models
    model = SentenceTransformer(CONFIG["model_id"])
    
    # Combine title and description to capture full semantic context
    train_text = (df_train['title'] + " " + df_train['description']).tolist()
    test_text = (df_test['title'] + " " + df_test['description']).tolist()
    
    embedding_train = model.encode(train_text, normalize_embeddings=True)
    embedding_test = model.encode(test_text, normalize_embeddings=True)

    # 3. Build FAISS Physical Search Index (For fast candidate retrieval)
    faiss_index = faiss.IndexFlatIP(embedding_train.shape[1])
    faiss_index = faiss.IndexIDMap2(faiss_index)
    # Map the vector directly to its position (index) in the df_train dataframe
    faiss_index.add_with_ids(embedding_train.astype('float32'), np.arange(len(df_train)))

    feature_store = FeatureStore(df_train, amenity_cols)

    # 4. Generate Machine Learning Training Dataset (Query -> Candidates)
    X_features, y_labels, query_ids = [], [], []

    # Sample 400 random queries from the training set
    sampled_query_indices = np.random.choice(len(df_train), 400, replace=False)
    
    logger.info("Generating Training Data Groups...")
    for qid, query_idx in enumerate(tqdm(sampled_query_indices)):
        query_row = df_train.iloc[query_idx]

        # Retrieve top 100 semantically similar properties via FAISS
        query_vector = normalize_query(embedding_train[query_idx])
        _, returned_faiss_ids = faiss_index.search(query_vector, 100)

        # Remove the query property itself from its own candidates (Anti-Leakage)
        valid_candidate_indices = [int(i) for i in returned_faiss_ids[0] if i >= 0 and i != query_idx]
        
        if not valid_candidate_indices:
            continue

        candidates_df = df_train.iloc[valid_candidate_indices]
        candidate_embeddings = embedding_train[valid_candidate_indices]

        # --- Calculate Ranking Features ---
        
        # 1. Price Proximity: Clipped log difference
        price_proximity = np.clip(np.log1p(np.abs(candidates_df['price_egp'].values - query_row['price_egp']) / (query_row['price_egp'] + 1e-9)), 0, 5)
        
        # 2. Geography: Log Haversine distance
        raw_geo_dist = haversine_vectorized(query_row['lat'], query_row['lon'], candidates_df['lat'], candidates_df['lon'])
        geo_distance_log = np.log1p(raw_geo_dist)
        
        # 3. Semantics: Dot product of embeddings
        semantic_similarity = np.dot(candidate_embeddings, embedding_train[query_idx])
        
        # 4. Property Type: Matrix lookup similarity
        type_similarity = np.array([get_type_sim(query_row['property_type'], pt) for pt in candidates_df['property_type']])
        
        # 5. Amenities: Jaccard intersection over union
        amenity_jaccard_sim = calculate_jaccard(
            feature_store.get_amenity_vec(query_row['listing_id']),
            np.stack([feature_store.get_amenity_vec(lid) for lid in candidates_df['listing_id']])
        )
        
        # 6-8. Structural Differences
        description_length_diff = np.abs(candidates_df['desc_log'] - query_row['desc_log'])
        bedrooms_diff = np.abs(candidates_df['bedrooms'] - query_row['bedrooms'])
        bathrooms_diff = np.abs(candidates_df['bathrooms'] - query_row['bathrooms'])

        # Stack features into an array of shape: (Num Candidates, 8 Features)
        X_batch = np.stack([
            price_proximity, geo_distance_log, semantic_similarity, type_similarity, 
            amenity_jaccard_sim, description_length_diff, bedrooms_diff, bathrooms_diff
        ], axis=1)

        # Generate the Teacher's relevance labels for this specific query-candidate group
        y_batch = get_spm_labels(query_row, candidates_df, np.expm1(geo_distance_log), amenity_jaccard_sim)

        if len(y_batch) == 0:
            continue

        X_features.extend(X_batch)
        y_labels.extend(y_batch)
        query_ids.extend([qid] * len(y_batch))

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    query_ids = np.array(query_ids)

    # 5. Split queries into Validation Strategy (85% Train / 15% Eval)
    unique_queries = np.unique(query_ids)
    tq_ids, vq_ids = train_test_split(unique_queries, test_size=0.15, random_state=42)

    # Create boolean masks to segment X and y
    mask_train = np.isin(query_ids, tq_ids)
    mask_val = np.isin(query_ids, vq_ids)

    X_train_fold, y_train_fold, qids_train_fold = X_features[mask_train], y_labels[mask_train], query_ids[mask_train]
    X_val_fold, y_val_fold, qids_val_fold = X_features[mask_val], y_labels[mask_val], query_ids[mask_val]

    # XGBRanker REQUIRES the query IDs to be strictly sorted
    order_train = np.argsort(qids_train_fold)
    order_val = np.argsort(qids_val_fold)

    X_train_fold, y_train_fold, qids_train_fold = X_train_fold[order_train], y_train_fold[order_train], qids_train_fold[order_train]
    X_val_fold, y_val_fold, qids_val_fold = X_val_fold[order_val], y_val_fold[order_val], qids_val_fold[order_val]

    logger.info("Training XGBoost Ranker...")
    ranker = xgb.XGBRanker(objective='rank:ndcg', n_estimators=500, learning_rate=0.05, early_stopping_rounds=20)

    # Train model using XGBoost's grouping mechanism
    ranker.fit(
        X_train_fold, y_train_fold,
        group=np.unique(qids_train_fold, return_counts=True)[1],
        eval_set=[(X_val_fold, y_val_fold)],
        eval_group=[np.unique(qids_val_fold, return_counts=True)[1]],
        verbose=10
    )

    # =========================================================
    # SYSTEM EVALUATION (nDCG METRIC)
    # =========================================================
    logger.info("Evaluating Final System nDCG...")
    ndcg_scores = []

    # Sample 50 entirely unseen test queries
    for test_idx in np.random.choice(len(df_test), 50, replace=False):
        test_query = df_test.iloc[test_idx]

        # Retrieve candidates via FAISS
        query_vector = normalize_query(embedding_test[test_idx])
        _, returned_faiss_ids = faiss_index.search(query_vector, 100)
        
        valid_candidate_indices = [int(x) for x in returned_faiss_ids[0] if x >= 0]
        if not valid_candidate_indices:
            continue

        cands = df_train.iloc[valid_candidate_indices]
        cands_emb = embedding_train[valid_candidate_indices]

        # Calculate features identical to training step
        price_prox = np.log1p(np.abs(cands['price_egp'] - test_query['price_egp']) / (test_query['price_egp'] + 1e-9))
        geo_dist = np.log1p(haversine_vectorized(test_query['lat'], test_query['lon'], cands['lat'], cands['lon']))
        sem_sim = np.dot(cands_emb, embedding_test[test_idx])
        type_sim = np.array([get_type_sim(test_query['property_type'], x) for x in cands['property_type']])
        
        query_amenities = np.array([test_query[col] for col in amenity_cols], dtype=np.float32)
        candidate_amenities_matrix = np.stack([feature_store.get_amenity_vec(l) for l in cands['listing_id']])
        amen_jaccard = calculate_jaccard(query_amenities, candidate_amenities_matrix)
        
        desc_length_diff = np.abs(cands['desc_log'] - test_query['desc_log'])
        beds_diff = np.abs(cands['bedrooms'] - test_query['bedrooms'])
        baths_diff = np.abs(cands['bathrooms'] - test_query['bathrooms'])

        # Predict relevance using trained XGBoost Ranker
        model_predictions = ranker.predict(np.stack([
            price_prox, geo_dist, sem_sim, type_sim, amen_jaccard, desc_length_diff, beds_diff, baths_diff
        ], axis=1))

        # We inject slight gaussian noise to the Teacher during test to simulate human behavioral variance
        simulated_human_relevance = get_spm_labels(
            test_query, cands,
            g_dist=np.expm1(geo_dist) + np.random.normal(0, 0.1, len(geo_dist)),
            a_jac=np.clip(amen_jaccard + np.random.normal(0, 0.05, len(amen_jaccard)), 0, 1)
        )

        # Sort the candidates by the model's prediction
        model_ranked_order_indices = np.argsort(model_predictions)[::-1][:10]
        
        # The relevance of the items in the order the model put them
        model_ordered_relevance = simulated_human_relevance[model_ranked_order_indices]

        # Calculate Discounted Cumulative Gain
        dcg = np.sum((2**model_ordered_relevance - 1) / np.log2(np.arange(2, len(model_ordered_relevance) + 2)))
        
        # Calculate Ideal Discounted Cumulative Gain (Perfect Sorting)
        ideal_relevance_sort = np.sort(simulated_human_relevance)[::-1][:10]
        idcg = np.sum((2**ideal_relevance_sort - 1) / np.log2(np.arange(2, 11 + 2)))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    logger.info(f"FINAL SYSTEM nDCG@10: {np.mean(ndcg_scores):.4f}")

if __name__ == "__main__":
    main()