import os
import sys
import json
import numpy as np
import pandas as pd
import importlib
import logging
import faiss
import xgboost as xgb
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))

config_cache_module = importlib.import_module("1_config_and_cache")
features_module = importlib.import_module("2_math_and_features")

CONFIG = config_cache_module.CONFIG
load_cache = config_cache_module.load_cache
FeatureStore = features_module.FeatureStore
FeatureEngine = features_module.FeatureEngine
mmr_rerank = features_module.mmr_rerank
euclidean_distance_vectorized = features_module.euclidean_distance_vectorized
normalize_query = features_module.normalize_query
get_type_sim = features_module.get_type_sim
calculate_jaccard = features_module.calculate_jaccard

CONTRACTS_DIR = os.path.join(BASE_DIR, "4.Data", "1_Contracts")
CANDIDATES_CONTRACT_FILE = os.path.join(CONTRACTS_DIR, "Rec_To_Ana_Candidates.json")
DESC_FAISS_INDEX_FILE = os.path.join(CONFIG["paths"]["artifacts"], "description_faiss.index")
DESC_IDS_FILE = os.path.join(CONFIG["paths"]["cache"], "description_ids.joblib")

logger = logging.getLogger(__name__)

# Global cache for models to avoid reloading on every query
_LOADED_MODELS_CACHE = {}
_QUERY_EMBEDDING_CACHE = TTLCache(maxsize=512, ttl=3600)

# --- Semantic Pre-Filter Index (real property descriptions) ---
# Loaded lazily once and reused for all subsequent queries.
_DESC_FAISS_INDEX = None
_DESC_INDEX_IDS = None  # Parallel list: position i -> listing_id


def _get_cached_query_embedding(model: SentenceTransformer, query_text: str) -> np.ndarray:
    """
    Embeds the synthetic query text once per unique query for a short-lived process cache.
    """
    cache_key = query_text.strip().lower()
    if cache_key in _QUERY_EMBEDDING_CACHE:
        return _QUERY_EMBEDDING_CACHE[cache_key]

    embedding = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")
    _QUERY_EMBEDDING_CACHE[cache_key] = embedding
    return embedding


def _get_description_faiss_index():
    """
    Lazily loads a prebuilt FAISS index on real property description text.
    This provides a semantic pre-filter channel that understands natural language
    like 'cozy garden apartment' or 'modern sea-view villa'.

    Important: this function never embeds all descriptions during a live request.
    Build the optional index offline with build_description_index.py.
    """
    global _DESC_FAISS_INDEX, _DESC_INDEX_IDS
    if _DESC_FAISS_INDEX is not None:
        return _DESC_FAISS_INDEX, _DESC_INDEX_IDS

    if not (os.path.exists(DESC_FAISS_INDEX_FILE) and os.path.exists(DESC_IDS_FILE)):
        logger.info(
            "[SEMANTIC PRE-FILTER] Prebuilt description index not found. "
            "Skipping optional description channel."
        )
        return None, None

    logger.info("[SEMANTIC PRE-FILTER] Loading prebuilt description FAISS index...")
    _DESC_FAISS_INDEX = faiss.read_index(DESC_FAISS_INDEX_FILE)
    _DESC_INDEX_IDS = load_cache("description_ids.joblib")
    logger.info(f"[SEMANTIC PRE-FILTER] Loaded {len(_DESC_INDEX_IDS or [])} description ids.")
    return _DESC_FAISS_INDEX, _DESC_INDEX_IDS

class QueryAdapter:
    """
    Adapter to bridge external queries (e.g., from a Chatbot) to the TRUE Recommender system,
    using the trained FAISS index and XGBoost model.
    """

    @classmethod
    def _load_models(cls):
        """Lazy load the models and caches into memory."""
        global _LOADED_MODELS_CACHE
        if _LOADED_MODELS_CACHE:
            return _LOADED_MODELS_CACHE

        logger.info("Loading production models and caches for QueryAdapter...")
        stage1_cache = load_cache("stage1_data.joblib")
        if not stage1_cache:
            raise FileNotFoundError("Recommender cache 'stage1_data.joblib' not found. Train Recommender first.")
        
        training_data, _, training_embeddings, _, amenity_columns = stage1_cache
        training_data = training_data.reset_index(drop=True)
        
        faiss_index_path = os.path.join(CONFIG["paths"]["artifacts"], "faiss_index.bin")
        faiss_search_index = faiss.read_index(faiss_index_path)
        
        ranker_path = os.path.join(CONFIG["paths"]["artifacts"], "xgb_ranker.json")
        xgboost_ranker = xgb.XGBRanker()
        xgboost_ranker.load_model(ranker_path)
        
        sentence_transformer_model = SentenceTransformer(CONFIG["model_id"])
        listing_feature_store = FeatureStore(training_data, amenity_columns)
        
        listing_id_to_index_mapping = {lid: idx for idx, lid in enumerate(training_data['listing_id'])}
        
        _LOADED_MODELS_CACHE = {
            "training_data": training_data,
            "training_embeddings": training_embeddings,
            "amenity_columns": amenity_columns,
            "faiss_search_index": faiss_search_index,
            "xgboost_ranker": xgboost_ranker,
            "sentence_transformer_model": sentence_transformer_model,
            "listing_feature_store": listing_feature_store,
            "listing_id_to_index_mapping": listing_id_to_index_mapping
        }
        return _LOADED_MODELS_CACHE

    @staticmethod
    def _primary_location_token(query: dict) -> str:
        """Prefer finest-grained location for filtering (town / district / subdistrict columns)."""
        for key in ("subdistrict", "district", "town"):
            val = query.get(key)
            if val:
                return str(val).strip().lower()
        return ""

    @classmethod
    def _create_synthetic_query_row(cls, query: dict, training_data: pd.DataFrame) -> pd.Series:
        """
        Converts the user's high-level filter dict into a Pandas Series that looks
        like a real property listing, so we can calculate feature differences.
        Uses Smart Imputation based on context, rather than global medians.
        """
        location_token = QueryAdapter._primary_location_token(query)
        target_city = str(query.get('city', '')).lower()
        target_types = [t.strip().lower() for t in str(query.get('property_type', 'Apartment')).split(',')]
        
        # Smart Imputation: Filter training data contextually
        location_filtered_data = pd.DataFrame()
        if location_token:
            location_filtered_data = training_data[
                (training_data['town'].fillna('').astype(str).str.lower() == location_token) |
                (training_data['district'].fillna('').astype(str).str.lower() == location_token) |
                (training_data['subdistrict'].fillna('').astype(str).str.lower() == location_token)
            ].copy()
            
        if location_filtered_data.empty and target_city:
            location_filtered_data = training_data[training_data['city'].fillna('').astype(str).str.lower() == target_city].copy()
            
        if location_filtered_data.empty:
            location_filtered_data = training_data.copy() # Global fallback
            
        # Further refine by property type to get accurate lookalikes
        type_filtered_data = location_filtered_data[location_filtered_data['property_type'].fillna('').astype(str).str.lower().isin(target_types)]
        imputation_source = type_filtered_data if not type_filtered_data.empty else location_filtered_data
            
        imputation_source.loc[:, 'price_egp'] = pd.to_numeric(imputation_source['price_egp'], errors='coerce')
        if query.get('price_min') is not None and query.get('price_max') is not None:
            try:
                target_price = (float(query['price_min']) + float(query['price_max'])) / 2.0
            except (TypeError, ValueError):
                target_price = query.get('price_egp') or query.get('price_max') or imputation_source['price_egp'].median()
        else:
            target_price = query.get('price_egp') or query.get('price_max') or imputation_source['price_egp'].median()
        
        # Financial Anchor Imputation for Area
        if query.get('area_value'):
            target_area = query.get('area_value')
        else:
            median_price_sqft = imputation_source['price_sqft'].median()
            if target_price and pd.notnull(median_price_sqft) and median_price_sqft > 0:
                target_area = target_price / median_price_sqft
            else:
                target_area = imputation_source['area_value'].median()
                
        # Structural Imputation for Bedrooms
        if query.get('bedrooms'):
            target_bedrooms = query.get('bedrooms')
        else:
            target_bedrooms = max(1, int(round(target_area / 55.0))) if pd.notnull(target_area) else 3
        
        synthetic_listing_data = {
            'category': query.get('category', 'buy').lower(),
            'city': query.get('city', ''),
            'town': query.get('town', ''),
            'property_type': target_types[0] if target_types else 'Apartment',
            'bedrooms': target_bedrooms,
            'bathrooms': query.get('bathrooms', max(1, target_bedrooms - 1)),
            'price_egp': target_price,
            'area_value': target_area,
            'price_sqft': target_price / (target_area + 1e-9) if target_area else imputation_source['price_sqft'].median(),
            'lat': imputation_source['lat'].median(),
            'lon': imputation_source['lon'].median(),
            'completion_status': str(query.get('completion_status', 'completed')).lower(),
            'payment_method': str(query.get('payment_method', 'cash')).lower()
        }
        
        # Context-Aware Meta-Features
        if target_price >= 10000000:
            synthetic_listing_data['listing_quality_score'] = 0.9
            synthetic_listing_data['desc_log'] = 6.5
        else:
            synthetic_listing_data['listing_quality_score'] = imputation_source['listing_quality_score'].median()
            synthetic_listing_data['desc_log'] = imputation_source['desc_log'].median()

        # Build description without poisoning missing locations
        luxury_prefix = "Exclusive Luxury " if target_price >= 10000000 else ""
        amenities_str = ", ".join(query.get('amenities', []))
        location_bits = [query.get('subdistrict'), query.get('district'), query.get('town'), query.get('city')]
        location_string = ", ".join([str(x) for x in location_bits if x])
        location_clause = f" in {location_string}" if location_string else ""
        
        generated_description = (
            f"{luxury_prefix}Stunning {synthetic_listing_data['property_type']} for {synthetic_listing_data['category']}{location_clause}. "
            f"Features {synthetic_listing_data['bedrooms']} spacious bedrooms and {synthetic_listing_data['bathrooms']} modern bathrooms. "
            f"Total area of {synthetic_listing_data['area_value']} sqm. Perfect location with premium finishes. "
            f"Includes amenities like: {amenities_str if amenities_str else 'modern facilities and high-end security'}."
        )
        synthetic_listing_data['synthetic_description'] = generated_description
            
        return pd.Series(synthetic_listing_data)

    @staticmethod
    def get_ranked_candidates(query: dict, top_k: int = 50) -> dict:
        """
        Processes a user query and returns the ranked results as a dictionary.
        This is thread-safe as it returns data in memory.
        """
        primary_location = QueryAdapter._primary_location_token(query)
        logger.info(
            "Processing query with keys=%s, primary_location=%s, has_city=%s",
            sorted(list(query.keys())),
            primary_location or "",
            bool(query.get("city")),
        )
        loaded_models = QueryAdapter._load_models()
        
        # Strict Core-Feature Guardrails (The Rejection Policy)
        has_location = bool(primary_location or query.get('city'))
        has_budget = bool(query.get('price_egp') or query.get('price_max') or query.get('price_min'))
        has_intent = bool(query.get('category'))
        
        if not (has_location and has_budget and has_intent):
            logger.warning("Query rejected: Missing core features (Intent, Location, or Budget).")
            return QueryAdapter._format_output([], is_vague=True, message="The input is not informative enough. Please ask the user for their preferred location, budget, and whether they want to buy or rent.")
        
        training_data = loaded_models["training_data"]
        training_embeddings = loaded_models["training_embeddings"]
        listing_id_to_index_mapping = loaded_models["listing_id_to_index_mapping"]
        
        # 1. Prepare Synthetic Row and Text Query
        query_row = QueryAdapter._create_synthetic_query_row(query, training_data)
        query_text = query_row['synthetic_description']
        
        # 1. RETRIEVAL LAYER (Semantic + Geographic Exploration)
        query_embedding_vector = _get_cached_query_embedding(
            loaded_models["sentence_transformer_model"], query_text
        )
        normalized_query_vector = normalize_query(query_embedding_vector)
        
        # 1a. Synthetic-description FAISS Search (300 candidates)
        _, retrieved_candidate_ids = loaded_models["faiss_search_index"].search(normalized_query_vector, 300)
        semantic_candidate_indices = [int(x) for x in retrieved_candidate_ids[0] if x >= 0]

        # ✅ NEW — 1b. Real-description Semantic Pre-Filter (natural language understanding)
        # Searches the actual property descriptions (not synthetic text) to surface listings
        # that match intent keywords like 'cozy', 'garden view', 'modern finishes', etc.
        try:
            desc_index, desc_ids = _get_description_faiss_index()
            if desc_index is not None and desc_ids:
                raw_query_vec = query_embedding_vector.astype("float32").reshape(1, -1)
                faiss.normalize_L2(raw_query_vec)
                _, desc_hits = desc_index.search(raw_query_vec, 150)
                desc_hit_ids = {str(desc_ids[int(i)]) for i in desc_hits[0] if i >= 0}
                # Map desc hit IDs back to training_data integer indices
                desc_hit_indices = training_data.index[
                    training_data["listing_id"].astype(str).isin(desc_hit_ids)
                ].tolist()
                logger.info(f"--- [SEMANTIC PRE-FILTER] Found {len(desc_hit_indices)} description matches ---")
            else:
                desc_hit_indices = []
        except Exception as e:
            logger.warning(f"[SEMANTIC PRE-FILTER] Skipped due to error: {e}")
            desc_hit_indices = []
        
        # 1b. Geographic Exploration
        target_location = primary_location
        target_city = str(query.get('city', '')).lower()
        
        geographic_candidate_pool = pd.DataFrame()
        if target_location:
            exact_geo_matches = training_data[
                (training_data['town'].fillna('').astype(str).str.lower() == target_location) |
                (training_data['district'].fillna('').astype(str).str.lower() == target_location) |
                (training_data['subdistrict'].fillna('').astype(str).str.lower() == target_location)
            ]
            
            # Soft Geographic Radius Expansion (5km Fallback)
            if len(exact_geo_matches) < 20 and target_city:
                logger.info(f"Geographic pool for '{target_location}' is too small ({len(exact_geo_matches)}). Activating Soft Radius Expansion.")
                target_lat = query_row.get('lat')
                target_lon = query_row.get('lon')
                
                if pd.notnull(target_lat) and pd.notnull(target_lon):
                    city_mask = training_data['city'].fillna('').astype(str).str.lower() == target_city
                    city_data = training_data[city_mask]
                    
                    distances = euclidean_distance_vectorized(
                        target_lat, target_lon, 
                        city_data['lat'].values, city_data['lon'].values
                    )
                    
                    # 0.05 degrees is roughly 5km
                    radius_mask = distances <= 0.05
                    radius_candidates = city_data[radius_mask]
                    geographic_candidate_pool = pd.concat([exact_geo_matches, radius_candidates]).drop_duplicates(subset=['listing_id'])
                else:
                    geographic_candidate_pool = exact_geo_matches
            else:
                geographic_candidate_pool = exact_geo_matches
                
        elif target_city:
            geographic_candidate_pool = training_data[
                training_data['city'].fillna('').astype(str).str.lower() == target_city
            ]
        
        if len(geographic_candidate_pool) > 0:
            # Smart Exploration: Prioritize properties that match the type and have high quality
            target_property_types = [t.strip().lower() for t in str(query.get('property_type', 'Apartment')).split(',')]
            property_type_matches = geographic_candidate_pool[geographic_candidate_pool['property_type'].fillna('').astype(str).str.lower().isin(target_property_types)]
            
            if not property_type_matches.empty:
                # Top quality matches for the requested type
                geographic_exploration_indices = property_type_matches.sort_values(by='listing_quality_score', ascending=False).head(100).index.tolist()
            else:
                # Fallback to general high-quality properties in that town
                geographic_exploration_indices = geographic_candidate_pool.sort_values(by='listing_quality_score', ascending=False).head(100).index.tolist()
        else:
            geographic_exploration_indices = []
            
        # Combine and deduplicate all three retrieval channels
        combined_candidate_indices = list(set(
            semantic_candidate_indices + geographic_exploration_indices + desc_hit_indices
        ))
        
        if not combined_candidate_indices:
            logger.warning("No candidates found.")
            return QueryAdapter._format_output(pd.DataFrame())
            
        candidate_properties = training_data.iloc[combined_candidate_indices].drop_duplicates(subset=['listing_id']).copy()
        
        # --- HARD FILTERING LAYER ---
        # 1. Location Filter (City/Town/District) - Extremely important for multi-region
        # If the user specified a location, we filter the pool to stay within that context.
        if target_location:
            location_filter_mask = (
                (candidate_properties['town'].fillna('').astype(str).str.lower() == target_location) |
                (candidate_properties['district'].fillna('').astype(str).str.lower() == target_location) |
                (candidate_properties['subdistrict'].fillna('').astype(str).str.lower() == target_location) |
                (candidate_properties['city'].fillna('').astype(str).str.lower() == target_location) |
                (candidate_properties['city'].fillna('').astype(str).str.lower() == target_city)
            )
            # Only apply if we actually have matches in that location to avoid empty results
            if location_filter_mask.any():
                candidate_properties = candidate_properties[location_filter_mask]
        
        # 2. Category Filter (Buy vs Rent) - Non-negotiable
        target_category = str(query.get('category', 'buy')).lower()
        candidate_properties = candidate_properties[candidate_properties['category'].fillna('unknown').astype(str).str.lower() == target_category]

        # 2b. Property Type Filter - Non-negotiable when the user specified types
        target_property_types = [
            t.strip().lower()
            for t in str(query.get('property_type', '')).split(',')
            if t and t.strip()
        ]
        if target_property_types:
            type_mask = candidate_properties['property_type'].fillna('').astype(str).str.lower().isin(target_property_types)
            if type_mask.any():
                candidate_properties = candidate_properties[type_mask]
        
        # 3. Budget Consideration
        candidate_properties['price_egp'] = pd.to_numeric(candidate_properties['price_egp'], errors='coerce')
        if query.get('price_min') is not None and query.get('price_max') is not None:
            try:
                budget_lo = float(query['price_min'])
                budget_hi = float(query['price_max'])
                candidate_properties = candidate_properties[
                    (candidate_properties['price_egp'] >= budget_lo)
                    & (candidate_properties['price_egp'] <= budget_hi)
                ]
            except Exception:
                pass
        elif query.get('price_max'):
            try:
                maximum_budget = float(query['price_max'])
                candidate_properties = candidate_properties[candidate_properties['price_egp'] <= maximum_budget]
            except Exception:
                pass
            
        if candidate_properties.empty:
            logger.warning(f"No candidates left after strict price filtering for {target_location}")
            return QueryAdapter._format_output(pd.DataFrame())

        # 4. Feature Extraction using Centralized Engine
        candidate_embedding_indices = [listing_id_to_index_mapping[lid] for lid in candidate_properties['listing_id'] if lid in listing_id_to_index_mapping]
        if not candidate_embedding_indices:
            logger.warning("No embedding indices resolved for filtered candidates.")
            return QueryAdapter._format_output(pd.DataFrame())
        candidate_embeddings = training_embeddings[candidate_embedding_indices]
        
        inference_features, _ = FeatureEngine.build_feature_stack(
            query_row, candidate_properties, candidate_embeddings, query_embedding_vector, loaded_models["listing_feature_store"]
        )
        relevance_predictions = loaded_models["xgboost_ranker"].predict(inference_features)
        
        # 5. Sort and Export with MMR Diversity Re-ranking
        candidate_properties['relevance_score'] = relevance_predictions
        candidate_properties = candidate_properties.sort_values(by='relevance_score', ascending=False)
        
        # Apply MMR to the top 30 to pick a diverse final top_k
        top_ranked_candidates = candidate_properties.head(30)
        
        # Extract embeddings for the top 30 candidates to calculate Semantic Diversity
        top_embedding_indices = [listing_id_to_index_mapping[lid] for lid in top_ranked_candidates['listing_id']]
        top_embeddings = training_embeddings[top_embedding_indices]
        
        # 6. Determine if the query is "Vague" and Calculate Dynamic MMR Lambda
        required_specific_features = ['property_type', 'bedrooms', 'bathrooms', 'area_value', 'amenities', 'completion_status']
        missing_feature_count = sum(1 for feat in required_specific_features if not query.get(feat))
        is_query_vague = missing_feature_count >= 5
        
        # Dynamic Diversity: If highly specific (0 missing), lambda=0.95 (favor relevance). 
        # If highly vague (5 missing), lambda=0.5 (favor diversity).
        specificity_score = max(0.0, 1.0 - (missing_feature_count / len(required_specific_features)))
        dynamic_lambda = 0.5 + (0.45 * specificity_score)
        
        logger.info(f"Dynamic MMR Lambda set to {dynamic_lambda:.2f} (Specificity Score: {specificity_score:.2f})")
        
        diverse_final_candidates = mmr_rerank(
            top_ranked_candidates, 
            top_ranked_candidates['relevance_score'].values, 
            candidate_embeddings=top_embeddings,
            top_k_results=top_k, 
            lambda_tuning_param=dynamic_lambda
        )
        
        logger.info(f"Successfully ranked and diversified top {len(diverse_final_candidates)} candidates. IsVague={is_query_vague}")
        return QueryAdapter._format_output(diverse_final_candidates, is_vague=is_query_vague)

    @staticmethod
    def _format_output(candidates, is_vague: bool = False, message: str = "") -> dict:
        """Formats the dataframe to the JSON Data Contract structure."""
        export_columns = [
            'listing_id', 'title', 'category', 'property_type', 'price_egp',
            'area_value', 'price_sqft', 'city', 'town', 'district', 'subdistrict',
            'completion_status', 'payment_method', 'bedrooms', 'bathrooms',
            'relevance_score'
        ]
        
        if len(candidates) > 0:
            available_columns = [col for col in export_columns if col in candidates.columns]
            dictionary_records = candidates[available_columns].to_dict(orient='records')
        else:
            dictionary_records = []
            
        return {
            "candidates": dictionary_records,
            "is_vague": is_vague,
            "message": message
        }

if __name__ == "__main__":
    # Test Query
    test_query_input = {
        "category": "buy",
        "city": "Cairo",
        "town": "New Cairo City",
        "property_type": "Apartment",
        "bedrooms": 3,
        "price_egp": 6000000,
        "area_value": 150
    }
    test_results = QueryAdapter.get_ranked_candidates(test_query_input, top_k=10)
    print(json.dumps(test_results, indent=2))
