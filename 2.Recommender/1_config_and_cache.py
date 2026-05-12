import os
import joblib

# ---------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------
# Sets up the baseline directory structure for Recommender artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = {
    "paths": {
        "artifacts": os.path.join(BASE_DIR, "2.Recommender", "1_model_artifacts"),
        "cache": os.path.join(BASE_DIR, "2.Recommender", "1_model_artifacts", "cache")
    },
    # The pre-trained sentence transformer model used for semantic text embeddings
    "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
    
    # Core amenities we actively check for feature generation
    "top_amenities": [
        'Balcony', 'Security', 'Covered Parking', 'Central A/C', 'Shared Gym',
        'Kitchen Appliances', 'Built in Wardrobes', 'Shared Spa', 'View of Landmark',
        'Shared Pool', 'Walk-in Closet', 'Study', 'Lobby in Building', 'View of Water',
        'Private Garden'
    ],
    
    # The exact order of features that the XGBoost model expects during inference
    "features": [
        'price_prox', 'semantic_sim', 'type_sim', 
        'amenity_jaccard', 'desc_log_diff', 'beds_diff', 'baths_diff',
        'price_sqft_diff', 'sqft_diff', 'category_match',
        'c_dom', 'c_lqs', 'c_completed', 'budget_penalty', 'market_premium'
    ],
    
    # Operational thresholds for the recommendation pipeline
    "thresholds": {
        "exploration_n": 50,         # FAISS nearest neighbors to retrieve before re-ranking
        "semantic_top_k": 300,       # Initial pool size for broad semantic searching
        "budget_buffer": 0.30,       # Allowable budget stretch percentage
        "mmr_lambda": 0.85           # Relevance vs Diversity tuning parameter (1.0 = All Relevance)
    }
}

# Heuristic matrix mapping how similar two property types are (0 to 1 scale)
PROPERTY_SIMILARITY_MATRIX = {
    'Apartment': {'Duplex': 0.8, 'Penthouse': 0.8, 'Hotel Apartment': 0.9, 'Studio': 0.7},
    'Duplex': {'Apartment': 0.8, 'Penthouse': 0.9},
    'Villa': {'Twin House': 0.8, 'Townhouse': 0.8, 'Palace': 0.9},
    'Townhouse': {'Villa': 0.8, 'Twin House': 0.9},
    'Chalet': {'Cabin': 0.9}
}

# =========================================================
# CACHING UTILITIES
# =========================================================
def save_cache(python_object_to_save, target_filename):
    """
    Serializes and saves a python object to the disk cache using joblib.
    This prevents recalculating heavy matrices on every query.
    """
    full_save_path = os.path.join(CONFIG["paths"]["cache"], target_filename)
    joblib.dump(python_object_to_save, full_save_path)

def load_cache(target_filename):
    """
    Attempts to load a serialized python object from the disk cache.
    Returns None if the file does not exist.
    """
    full_load_path = os.path.join(CONFIG["paths"]["cache"], target_filename)
    if os.path.exists(full_load_path):
        return joblib.load(full_load_path)
    return None
