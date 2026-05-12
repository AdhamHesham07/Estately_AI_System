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
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = r"C:\Users\Adham\Desktop\AI_System\5.Data\propertyfinder.csv"
ARTIFACTS_DIR = r"C:\Users\Adham\Desktop\AI_System\2.Recommender\1_model_artifacts"

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculates distances between two sets of coordinates in kilometers."""
    R = 6371  # Earth's radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# PHASE 2.5: Graded Property Similarity Matrix
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

def get_type_similarity(type1, type2):
    if type1 == type2: return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(type1, {}).get(type2, 0.0)

def _rank_candidates(query_item, candidates_df, query_embed, candidate_embeds, knn_dists):
    """Shared ranking logic used by both evaluation and production."""
    # 1. Structural Score (Point 1 Recovery)
    # Re-integrating the structural distance from the initial search
    structural_scores = np.exp(-knn_dists)
    
    # 2. Semantic Similarity
    norm_query = query_embed / np.linalg.norm(query_embed)
    norm_candidates = candidate_embeds / np.expand_dims(np.linalg.norm(candidate_embeds, axis=1), 1)
    semantic_scores = np.dot(norm_candidates, norm_query)
    
    # 3. Geo Scoring (Point 3 & 4 Adaptive)
    geo_distances = haversine_vectorized(
        query_item['lat'], query_item['lon'],
        candidates_df['lat'].values, candidates_df['lon'].values
    )
    # Adaptive decay: tighter in towns, broader in cities
    decay_val = 3.0 if pd.notna(query_item.get('town')) else 8.0
    geo_scores = np.exp(-geo_distances / decay_val)
    
    # 4. Graded Type Similarity
    type_scores = np.array([get_type_similarity(query_item['property_type'], t) for t in candidates_df['property_type']])
    
    # 5. Dedicated Price Proximity Boost
    query_price = query_item['price_egp']
    candidate_prices = candidates_df['price_egp'].values
    # exp(-% deviation) - rewards properties closer to the actual budget
    price_scores = np.exp(-np.abs(candidate_prices - query_price) / query_price)
    
    # 6. Combined Scoring (Weight Tuning)
    # Balanced blend: 50% on math/budget, 20% on vibe/text, 30% on geo/type
    final_scores = (
        0.30 * structural_scores + 
        0.20 * price_scores + 
        0.20 * semantic_scores + 
        0.15 * geo_scores + 
        0.15 * type_scores
    )
    
    # Return sorted internal positional indices
    return np.argsort(final_scores)[::-1]

def load_and_prepare_data():
    logger.info(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    original_count = len(df)
    
    # 1. Clean Numeric Strings (Ported from notebook)
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
        
    # Ensure numeric types
    for col in ['price_egp', 'area_value', 'lat', 'lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 2. Basic Filtering (Drop rows missing critical recommendation data)
    df = df.dropna(subset=['listing_id', 'price_egp'])
    
    # 3. Handle Missing Values (Ported from notebook)
    num_cols = ['lat', 'lon', 'bedrooms', 'bathrooms', 'area_value']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # 4. Outlier Removal (Ported from notebook IQR method)
    def remove_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]
    
    logger.info("Removing outliers in price and area...")
    df = remove_outliers(df, 'price_egp')
    df = remove_outliers(df, 'area_value')
    
    # 5. Feature Engineering (Amenities Count)
    if 'amenities' in df.columns:
        df['amenities'] = df['amenities'].fillna('')
        df['amenities_count'] = df['amenities'].apply(lambda x: len(str(x).split('|')) if x and str(x).strip() else 0)
    else:
        df['amenities_count'] = 0
    
    # 6. NLP Cleanup
    df['description'] = df['description'].fillna('No description available')
    df['title'] = df['title'].fillna('No title available')
    
    logger.info(f"Preprocessing complete. Rows: {original_count} -> {len(df)}")
    
    # PHASE 2: Type Similarity Mapping
    # (Existing type mapping logic for property categories)
    type_groups = {
        'Residential': [0, 5, 6, 7, 10, 11, 17], # These will map after we encoding categories later if needed,
        # but the raw file has STRINGS for property_type. We need to handle that.
    }
    # Wait, the property_type in RAW is strings like 'Villa', 'Apartment'.
    # We should encode these strings to match our grouped logic.
    group_mapping = {
        'Apartment': 'Residential', 'Duplex': 'Residential', 'Penthouse': 'Residential', 
        'Hotel Apartment': 'Residential', 'iVilla': 'Residential', 'Half Floor': 'Residential', 'Roof': 'Residential',
        'Villa': 'Villa', 'Townhouse': 'Villa', 'Twin House': 'Villa', 'Palace': 'Villa', 'Bungalow': 'Villa',
        'Chalet': 'Vacation', 'Cabin': 'Vacation',
        'Land': 'Other', 'Whole Building': 'Other', 'Bulk Sale Unit': 'Other'
    }
    df['type_group'] = df['property_type'].map(group_mapping).fillna('Other')
    
    # For evaluate_recommendations, we still need numeric codes for 'property_type'
    # We'll use a simple factorize/LabelEncoder approach internally
    if 'property_type' in df.columns:
        df['property_type_label'] = pd.factorize(df['property_type'])[0]
    else:
        df['property_type_label'] = 0
        
    # CRITICAL: Reset index after all drops/filters to ensure index matches numpy array positions
    df = df.reset_index(drop=True)
    
    return df

def evaluate_recommendations(df_train, df_test, knn, scaler, embeddings_train, embeddings_test):
    logger.info("Evaluating full hybrid pipeline on Test Set (10%)...")
    
    structural_features = ['price_egp', 'area_value', 'bedrooms', 'bathrooms', 'lat', 'lon', 'amenities_count']
    X_test = df_test[structural_features].copy()
    X_test['price_egp'] = np.log1p(X_test['price_egp'])
    X_test['area_value'] = np.log1p(X_test['area_value'])
    X_test_scaled = scaler.transform(X_test) * [2.5, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    
    distances, indices = knn.kneighbors(X_test_scaled)
    
    price_errors = []
    strict_type_matches = []
    mrr_scores = []
    
    for i in range(len(df_test)):
        query_item = df_test.iloc[i]
        query_embed = embeddings_test[i]
        
        neighbor_pos_indices = indices[i]
        candidates = df_train.iloc[neighbor_pos_indices]
        candidate_embeds = embeddings_train[neighbor_pos_indices]
        knn_dists = distances[i]
        
        # Point 5: Using the same ranking logic as production
        sorted_offsets = _rank_candidates(query_item, candidates, query_embed, candidate_embeds, knn_dists)
        top_recs = candidates.iloc[sorted_offsets[:5]]
        
        # Metrics
        query_price = query_item['price_egp']
        rec_prices = top_recs['price_egp'].values
        # Individual price errors per recommendation
        individual_errors = [abs(p - query_price) / query_price for p in rec_prices]
        price_errors.append(np.mean(individual_errors))
        
        query_type = query_item['property_type']
        rec_types = top_recs['property_type'].values
        strict_type_matches.append(np.sum(rec_types == query_type) / 5)
        
        # MRR
        ranks = np.where(rec_types == query_type)[0]
        mrr_scores.append(1 / (ranks[0] + 1) if len(ranks) > 0 else 0)
        
    mean_price_error = np.mean(price_errors) * 100
    avg_strict_match = np.mean(strict_type_matches) * 100
    mrr = np.mean(mrr_scores)
    
    logger.info(f"--- Pipeline Evaluation (Hybrid) ---")
    logger.info(f"Mean Price Deviation: {mean_price_error:.2f}%")
    logger.info(f"Strict Type Accuracy: {avg_strict_match:.2f}%")
    logger.info(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    logger.info(f"------------------------------------")
    
    return mean_price_error, mrr

def train_recommender(df):
    logger.info("Preparing features for KNN...")
    
    # Features to use for structural similarity
    # price_egp and area_value are the most important
    structural_features = [
        'price_egp', 'area_value', 'bedrooms', 'bathrooms', 
        'lat', 'lon', 'amenities_count'
    ]
    
    # PHASE 1: Transform features
    X = df[structural_features].copy()
    X['price_egp'] = np.log1p(X['price_egp'])
    X['area_value'] = np.log1p(X['area_value'])
    
    # Outlier-robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=structural_features)
    
    # Apply weights: Price and Area are prioritized
    weights = {
        'price_egp': 2.5,
        'area_value': 2.0,
        'bedrooms': 1.0,
        'bathrooms': 1.0,
        'lat': 0.5,
        'lon': 0.5,
        'amenities_count': 0.5
    }
    
    for feature, weight in weights.items():
        X_scaled[feature] *= weight
        
    logger.info("Training KNN model (Structural retrieval)...")
    knn = NearestNeighbors(n_neighbors=50, metric='euclidean', n_jobs=-1)
    knn.fit(X_scaled)
    
    return knn, X_scaled, scaler

def generate_embeddings(df):
    model_id = 'paraphrase-multilingual-MiniLM-L12-v2'
    logger.info(f"Initializing standard Transformer ({model_id}) on CPU...")
    
    # Load original SentenceTransformer
    model = SentenceTransformer(model_id)
    
    # Combine title and description for richer context
    texts = (df['title'] + " " + df['description']).tolist()
    
    logger.info(f"Generating embeddings for {len(texts)} listings on CPU...")
    # Using batches and tqdm for progress tracking
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings

def precompute_recommendations(df, knn, X_scaled, embeddings):
    logger.info("Pre-computing Top 10 recommendations for all listings...")
    
    recommendations_list = []
    
    distances, indices = knn.kneighbors(X_scaled)
    
    for i in tqdm(range(len(df)), desc="Reranking End-to-End"):
        query_item = df.iloc[i]
        query_price = query_item['price_egp']
        query_city = query_item['city']
        query_embed = embeddings[i]
        
        neighbor_indices = indices[i][1:]
        knn_dists = distances[i][1:]
        
        # Point 3: Progressive Relaxation
        # Level 1: Strict (±30% price, Same City)
        mask = (df.iloc[neighbor_indices]['price_egp'] >= query_price * 0.7) & \
               (df.iloc[neighbor_indices]['price_egp'] <= query_price * 1.3) & \
               (df.iloc[neighbor_indices]['city'] == query_city)
        
        filtered_offsets = np.where(mask)[0]
        
        # Level 2: Relax Price (±50%)
        if len(filtered_offsets) < 5:
            mask = (df.iloc[neighbor_indices]['price_egp'] >= query_price * 0.5) & \
                   (df.iloc[neighbor_indices]['price_egp'] <= query_price * 1.5) & \
                   (df.iloc[neighbor_indices]['city'] == query_city)
            filtered_offsets = np.where(mask)[0]
            
        # Level 3: Relax City (All KNN neighbors)
        if len(filtered_offsets) < 5:
            filtered_offsets = np.arange(len(neighbor_indices))
            
        # Prepare for ranking
        final_candidate_idxs = neighbor_indices[filtered_offsets]
        candidates_df = df.iloc[final_candidate_idxs]
        candidate_embeds = embeddings[final_candidate_idxs]
        final_knn_dists = knn_dists[filtered_offsets]
        
        # Shared Reranking Engine (Points 1, 2, 4)
        sorted_candidate_offsets = _rank_candidates(query_item, candidates_df, query_embed, candidate_embeds, final_knn_dists)
        
        # Map back to listing IDs
        top_recs = candidates_df.iloc[sorted_candidate_offsets[:10]]
        
        recommendations_list.append({
            'listing_id': query_item['listing_id'],
            'recommendations': ",".join(top_recs['listing_id'].tolist())
        })
        
    return pd.DataFrame(recommendations_list)

def main():
    try:
        if not os.path.exists(ARTIFACTS_DIR):
            os.makedirs(ARTIFACTS_DIR)
            
        # 1. Load Data
        df = load_and_prepare_data()
        
        # 2. Train-Test Split (90/10)
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
        logger.info(f"Split data: Train={len(df_train)}, Test={len(df_test)}")
        
        # 3. Train Structural KNN on Training Data
        knn, X_train_scaled, scaler = train_recommender(df_train)
        
        # 4. Generate Semantic Embeddings (Full Dataset)
        embeddings_full = generate_embeddings(df)
        
        # Split embeddings for evaluation
        train_idxs = df_train.index
        test_idxs = df_test.index
        embeddings_train = embeddings_full[train_idxs]
        embeddings_test = embeddings_full[test_idxs]
        
        # 5. Evaluate
        evaluate_recommendations(df_train, df_test, knn, scaler, embeddings_train, embeddings_test)
        
        # 6. Precompute Recommendations
        structural_features = ['price_egp', 'area_value', 'bedrooms', 'bathrooms', 'lat', 'lon', 'amenities_count']
        weights = [2.5, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        
        # TRANSFORM for full dataset
        X_full = df[structural_features].copy()
        X_full['price_egp'] = np.log1p(X_full['price_egp'])
        X_full['area_value'] = np.log1p(X_full['area_value'])
        
        X_full_scaled = scaler.transform(X_full) * weights
        
        recs_df = precompute_recommendations(df, knn, X_full_scaled, embeddings_full)
        
        # 7. Save Artifacts
        logger.info("Saving model artifacts...")
        joblib.dump(knn, os.path.join(ARTIFACTS_DIR, 'knn_model.joblib'))
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
        np.save(os.path.join(ARTIFACTS_DIR, 'embeddings.npy'), embeddings_full)
        recs_df.to_csv(os.path.join(ARTIFACTS_DIR, 'top_recommendations.csv'), index=False)
        
        logger.info("Training complete. Artifacts saved to 2.Recommender/model_artifacts/")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
