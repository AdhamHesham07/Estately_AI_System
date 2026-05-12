import importlib
import numpy as np
import faiss

# Dynamically import the configuration module to access matrices and paths
config_module = importlib.import_module("1_config_and_cache")
PROPERTY_SIMILARITY_MATRIX = config_module.PROPERTY_SIMILARITY_MATRIX

# =========================================================
# FEATURE STORE
# =========================================================
class FeatureStore:
    """
    A lightning-fast dictionary-based lookup for property features.
    Pre-calculates amenity matrices into numpy arrays for vectorized speed during queries.
    """
    def __init__(self, property_dataframe, amenity_columns_list):
        # Convert the dataframe to a dictionary for O(1) row lookups
        self.fast_lookup_store = property_dataframe.set_index('listing_id').to_dict('index')
        self.tracked_amenity_columns = amenity_columns_list
        
        # Vectorization: Pre-calculate the entire Amenity Matrix into memory
        self.ordered_listing_ids = property_dataframe['listing_id'].tolist()
        self.listing_id_to_matrix_position = {str(listing_id): index for index, listing_id in enumerate(self.ordered_listing_ids)}
        self.global_amenity_matrix = property_dataframe[amenity_columns_list].fillna(0).values.astype(np.float32)
        
        # Calculate Market Medians globally and locally for Premium feature calculation
        self.town_median_price_per_sqft = property_dataframe.groupby('town')['price_sqft'].median().to_dict()
        self.global_median_price_per_sqft = property_dataframe['price_sqft'].median()

    def get(self, target_listing_id):
        """Retrieves a single property row dictionary by its ID."""
        return self.fast_lookup_store.get(str(target_listing_id))

    def get_amenity_vec(self, target_listing_id):
        """Retrieves the pre-calculated numpy amenity vector for a single property."""
        matrix_position = self.listing_id_to_matrix_position.get(str(target_listing_id))
        if matrix_position is not None:
            return self.global_amenity_matrix[matrix_position]
        return np.zeros(len(self.tracked_amenity_columns), dtype=np.float32)

    def get_amenity_matrix_batch(self, list_of_listing_ids):
        """Retrieves a batch of amenity vectors in one go for fast Jaccard scoring."""
        retrieved_positions = [self.listing_id_to_matrix_position.get(str(listing_id)) for listing_id in list_of_listing_ids]
        
        # Filter out invalid IDs
        valid_positions = [position for position in retrieved_positions if position is not None]
        
        # Fast path: all IDs were valid
        if len(valid_positions) == len(list_of_listing_ids):
            return self.global_amenity_matrix[valid_positions]
        
        # Fallback path: handle missing/mixed IDs by injecting zeros
        fallback_matrix = np.zeros((len(list_of_listing_ids), len(self.tracked_amenity_columns)), dtype=np.float32)
        for row_index, position in enumerate(retrieved_positions):
            if position is not None: 
                fallback_matrix[row_index] = self.global_amenity_matrix[position]
        return fallback_matrix

# =========================================================
# MATHEMATICAL AND DISTANCE ENGINES
# =========================================================
def euclidean_distance_vectorized(latitude_1, longitude_1, latitude_array_2, longitude_array_2):
    """
    Calculates the simple Euclidean distance between GPS coordinates.
    """
    return np.sqrt((latitude_array_2 - latitude_1)**2 + (longitude_array_2 - longitude_1)**2)

def mmr_rerank(candidates_dataframe, candidate_scores, candidate_embeddings=None, top_k_results=10, lambda_tuning_param=0.7):
    """
    Maximal Marginal Relevance (MMR) re-ranking algorithm.
    Balances Relevance (how well it matches the query) and Diversity (how physically spread out the results are).
    Prevents the recommender from showing 10 identical apartments in the same exact building.
    """
    if len(candidates_dataframe) <= top_k_results:
        return candidates_dataframe
    
    final_selected_indices = []
    remaining_candidate_indices = list(range(len(candidates_dataframe)))
    
    # Extract coordinates to compute geographical diversity
    candidate_latitudes = candidates_dataframe['lat'].values
    candidate_longitudes = candidates_dataframe['lon'].values
    
    # 1. Start by greedily picking the highest scoring candidate regardless of diversity
    best_initial_index = np.argmax(candidate_scores)
    final_selected_indices.append(best_initial_index)
    remaining_candidate_indices.remove(best_initial_index)
    
    # 2. Iteratively pick the next best candidate balancing score vs distance to already picked items
    while len(final_selected_indices) < top_k_results and remaining_candidate_indices:
        current_mmr_scores = []
        for candidate_index in remaining_candidate_indices:
            
            relevance_score = candidate_scores[candidate_index]
            
            # Diversity penalty 1: Spatial distance (minimize clumping)
            distances_to_selected = euclidean_distance_vectorized(
                candidate_latitudes[candidate_index], candidate_longitudes[candidate_index],
                candidate_latitudes[final_selected_indices], candidate_longitudes[final_selected_indices]
            )
            spatial_diversity = np.clip(np.min(distances_to_selected) / 0.01, 0, 1)
            
            # Diversity penalty 2: Semantic similarity (minimize duplicate content/descriptions)
            if candidate_embeddings is not None:
                semantic_similarities = np.dot(
                    candidate_embeddings[final_selected_indices], 
                    candidate_embeddings[candidate_index]
                )
                # Max similarity to any selected item (1.0 = identical, 0.0 = completely different)
                max_semantic_sim = np.max(semantic_similarities)
                # Convert similarity to diversity (1.0 = highly diverse, 0.0 = identical)
                semantic_diversity = np.clip(1.0 - max_semantic_sim, 0, 1)
                
                # Combine both layers: an item must be physically OR semantically diverse to score high
                # We average them, but heavily penalize if either is completely identical (0)
                combined_diversity_score = (spatial_diversity + semantic_diversity) / 2.0
            else:
                combined_diversity_score = spatial_diversity
            
            # Calculate final MMR metric: combine tuned relevance and tuned diversity
            final_mmr_value = lambda_tuning_param * relevance_score + (1 - lambda_tuning_param) * combined_diversity_score
            current_mmr_scores.append(final_mmr_value)
            
        # Pick the candidate that scored highest on the MMR formula
        best_mmr_index = remaining_candidate_indices[np.argmax(current_mmr_scores)]
        final_selected_indices.append(best_mmr_index)
        remaining_candidate_indices.remove(best_mmr_index)
        
    return candidates_dataframe.iloc[final_selected_indices]

def normalize_query(query_embedding_vector):
    """
    Ensures that query vectors are L2 normalized.
    This is mathematically required for FAISS to correctly compute Cosine Similarity via Inner Product.
    """
    query_embedding_vector = np.ascontiguousarray(query_embedding_vector.astype('float32').reshape(1, -1))
    faiss.normalize_L2(query_embedding_vector)
    return query_embedding_vector

def get_type_sim(property_type_1, property_type_2):
    """
    Looks up the similarity between two property types using the heuristic matrix.
    E.g., A Villa and a Townhouse are 80% similar, but a Villa and a Studio are heavily penalized.
    """
    pt1 = str(property_type_1).title()
    pt2 = str(property_type_2).title()
    if pt1 == pt2:
        return 1.0
    return PROPERTY_SIMILARITY_MATRIX.get(pt1, {}).get(pt2, 0.05)

def calculate_jaccard(query_amenity_vector, candidate_amenity_matrix):
    """
    Calculates the Jaccard similarity (Intersection over Union) 
    between a query's requested amenities and a matrix of candidate amenities.
    Highly vectorized for performance.
    """
    query_amenity_vector = query_amenity_vector.astype(np.float32).reshape(1, -1)
    candidate_amenity_matrix = candidate_amenity_matrix.astype(np.float32)
    
    amenity_intersection = np.sum(query_amenity_vector * candidate_amenity_matrix, axis=1)
    amenity_union = np.sum(np.maximum(query_amenity_vector, candidate_amenity_matrix), axis=1)
    
    # Avoid division by zero by adding a microscopic constant
    return np.nan_to_num(amenity_intersection / (amenity_union + 1e-9))

# =========================================================
# FEATURE ENGINE
# =========================================================
class FeatureEngine:
    """
    Centralized mathematical engine to build the feature stack for XGBoost Ranking.
    Critically ensures that the exact same transformations occur in Training, Evaluation, and Live Inference.
    """
    @staticmethod
    def build_feature_stack(query_row_dictionary, candidate_dataframe, candidate_embedding_matrix, query_embedding_vector, feature_store_instance):
        """
        Builds a stacked numpy array of engineered features for a set of candidates against a single query.
        """
        # 1. Price Proximity & Budget Penalty
        raw_price_difference = candidate_dataframe['price_egp'].values - query_row_dictionary['price_egp']
        price_proximity_log = np.clip(np.log1p(np.abs(raw_price_difference) / (query_row_dictionary['price_egp'] + 1e-9)), 0, 5)
        
        # Soft Budget Penalty: Apply a mathematical penalty ONLY if the property exceeds the budget
        asymmetric_budget_penalty = np.where(raw_price_difference > 0, np.log1p(raw_price_difference / (query_row_dictionary['price_egp'] + 1e-9)), 0)
        
        # 2. Semantic Similarity via Inner Product
        semantic_similarity_scores = np.dot(candidate_embedding_matrix, query_embedding_vector)
        if semantic_similarity_scores.ndim > 1: 
            semantic_similarity_scores = semantic_similarity_scores.flatten()
        
        # 3. Property Type Heuristic Similarity
        type_similarity_scores = np.array([get_type_sim(query_row_dictionary['property_type'], candidate_type) for candidate_type in candidate_dataframe['property_type']])
        
        # 4. Amenity Jaccard Overlap
        query_amenities = feature_store_instance.get_amenity_vec(query_row_dictionary.get('listing_id', -1))
        # Fallback if the query doesn't exist in the store (i.e., synthetic user input)
        if 'listing_id' not in query_row_dictionary or query_row_dictionary['listing_id'] == -1:
             query_amenities = np.array([query_row_dictionary.get(col, 0) for col in feature_store_instance.tracked_amenity_columns], dtype=np.float32)
             
        candidate_amenity_matrix = feature_store_instance.get_amenity_matrix_batch(candidate_dataframe['listing_id'])
        amenity_jaccard_scores = calculate_jaccard(query_amenities, candidate_amenity_matrix)
        
        # 5. Structural Differences (Log Normalized)
        size_difference_log = np.log1p(np.abs(candidate_dataframe['area_value'].values - query_row_dictionary['area_value']) / (query_row_dictionary['area_value'] + 1e-9))
        price_per_sqft_difference_log = np.log1p(np.abs(candidate_dataframe['price_sqft'].values - query_row_dictionary['price_sqft']) / (query_row_dictionary['price_sqft'] + 1e-9))
        description_length_difference = np.abs(candidate_dataframe['desc_log'].values - query_row_dictionary['desc_log'])
        bedroom_difference = np.abs(candidate_dataframe['bedrooms'].values - query_row_dictionary['bedrooms'])
        bathroom_difference = np.abs(candidate_dataframe['bathrooms'].values - query_row_dictionary['bathrooms'])
        
        # 6. Category Match (Binary: buy vs rent)
        category_match_binary = (candidate_dataframe['category'].values == query_row_dictionary['category']).astype(float)
        
        # 7. Quality & Metadata Extractors
        def _get_column_values_safe(dataframe, column_name, default_value=0):
            if column_name in dataframe.columns:
                return dataframe[column_name].fillna(default_value).values
            return np.full(len(dataframe), default_value)

        days_on_market_scores = _get_column_values_safe(candidate_dataframe, 'days_on_market')
        listing_quality_scores = _get_column_values_safe(candidate_dataframe, 'listing_quality_score')
        completion_status_binary = (_get_column_values_safe(candidate_dataframe, 'completion_status', '') == 'completed').astype(float)

        # 8. Market Premium (New Feature): Evaluates if a property is overpriced relative to its specific town
        candidate_towns = candidate_dataframe['town'].values
        town_median_prices = np.array([feature_store_instance.town_median_price_per_sqft.get(town, feature_store_instance.global_median_price_per_sqft) for town in candidate_towns])
        market_premium_ratio = candidate_dataframe['price_sqft'].values / (town_median_prices + 1e-9)

        # Stack all engineered arrays into a single massive 2D matrix for XGBoost
        engineered_feature_matrix = [
            price_proximity_log, semantic_similarity_scores, type_similarity_scores, 
            amenity_jaccard_scores, description_length_difference, bedroom_difference, bathroom_difference, 
            price_per_sqft_difference_log, size_difference_log, category_match_binary, days_on_market_scores, 
            listing_quality_scores, completion_status_binary, asymmetric_budget_penalty, market_premium_ratio
        ]

        return np.stack(engineered_feature_matrix, axis=1), amenity_jaccard_scores
