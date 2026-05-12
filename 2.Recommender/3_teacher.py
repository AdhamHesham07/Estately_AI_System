import importlib
import numpy as np
import pandas as pd

# Import mathematical utilities
math_and_features_module = importlib.import_module("2_math_and_features")
get_type_sim = math_and_features_module.get_type_sim

# =========================================================
# PERSONA-BASED BEHAVIORAL SIMULATOR
# =========================================================
# The "Teacher" module simulates how different types of human buyers and renters 
# would evaluate a list of properties. By generating synthetic labels based on these 
# diverse personas, we can train the XGBoost ranker to satisfy a wide variety of users.
PERSONAS = {
    # --- BUYERS ---
    0: {
        "name": "The Bargain Hunter (Buy)",
        "category": "buy",
        "desc": "Prioritizes price and unit economics above all else. Willing to compromise on location and amenities.",
        "weights": {"price": 0.8, "town": 0.1, "amenities": 0.1, "size": 0.0}
    },
    1: {
        "name": "The Family Weaver (Buy)",
        "category": "buy",
        "desc": "Prioritizes neighborhood location (Geo) and space (Area/Beds) for a growing family.",
        "weights": {"price": 0.2, "town": 0.4, "amenities": 0.1, "size": 0.3}
    },
    2: {
        "name": "The Luxury Elite (Buy)",
        "category": "buy",
        "desc": "Prioritizes property features, luxury amenities, and prestige. Price is not an object.",
        "weights": {"price": 0.1, "town": 0.1, "amenities": 0.5, "size": 0.3}
    },
    # --- RENTERS ---
    3: {
        "name": "The Bargain Hunter (Rent)",
        "category": "rent",
        "desc": "Prioritizes low monthly rent cost. Often seeks unfurnished units.",
        "weights": {"price": 0.7, "town": 0.2, "amenities": 0.0, "size": 0.1}
    },
    4: {
        "name": "The Family Weaver (Rent)",
        "category": "rent",
        "desc": "Prioritizes proximity to key locations and sufficient bedrooms.",
        "weights": {"price": 0.3, "town": 0.4, "amenities": 0.1, "size": 0.2}
    },
    5: {
        "name": "The Luxury Elite (Rent)",
        "category": "rent",
        "desc": "Prioritizes fully furnished status, luxury amenities, and immediate comfort.",
        "weights": {"price": 0.1, "town": 0.2, "amenities": 0.6, "size": 0.1}
    },
    # --- CONTRARIAN / STRESS TEST ---
    6: {
        "name": "The Hidden Gem Hunter (Contrarian)",
        "category": "buy",
        "desc": "Has a high budget but strictly looks for high-quality properties at significantly lower price points (Value Seeker).",
        "weights": {"price": 0.1, "town": 0.4, "amenities": 0.2, "size": 0.3}
    }
}

def get_spm_labels(query_row_dictionary, candidate_dataframe, amenity_jaccard_array, forced_persona_id=None):
    """
    Generates behavioral relevance labels (1-4 scale) by simulating a specific human persona.
    If forced_persona_id is None, a random persona matching the query category is selected.
    This creates the "Ground Truth" data (y) that the XGBoost model trains against.
    """
    if len(candidate_dataframe) == 0:
        return np.array([])

    # 1. Persona Selection: Match the persona to the Buy/Rent intent of the query
    if forced_persona_id is None:
        query_category = query_row_dictionary.get('category', 'buy').lower()
        valid_persona_ids = [persona_id for persona_id, persona_config in PERSONAS.items() if persona_config.get("category") == query_category]
        if not valid_persona_ids: # Failsafe if category is missing or mangled
            valid_persona_ids = list(PERSONAS.keys())
        forced_persona_id = np.random.choice(valid_persona_ids)
    
    persona_configuration = PERSONAS[forced_persona_id]
    persona_weights = persona_configuration["weights"]

    # --- Pre-calculate Individual Component Scores (Normalized 0 to 1) ---
    
    # A. Price Score (Inverted relative mathematical error)
    target_query_price = query_row_dictionary['price_egp']
    relative_price_error = np.abs(candidate_dataframe['price_egp'].values - target_query_price) / (target_query_price + 1e-9)
    normalized_price_score = np.clip(1.0 - (relative_price_error / 0.5), 0, 1) # Full score if exact match, 0 if >50% difference
    
    # B. Town Score (Binary exact match replaces geographical distance)
    is_exact_town_match = (candidate_dataframe['town'].values == query_row_dictionary['town']).astype(float)
    
    # C. Size Score (Area difference and exact Bedroom match)
    relative_area_error = np.abs(candidate_dataframe['area_value'].values - query_row_dictionary['area_value']) / (query_row_dictionary['area_value'] + 1e-9)
    normalized_area_score = np.clip(1.0 - (relative_area_error / 0.5), 0, 1)
    exact_bedroom_match_binary = (candidate_dataframe['bedrooms'].values == query_row_dictionary['bedrooms']).astype(float)
    combined_size_score = (normalized_area_score + exact_bedroom_match_binary) / 2.0
    
    # D. Amenities Score (Direct pass-through of the Jaccard similarity index)
    normalized_amenity_score = np.clip(amenity_jaccard_array, 0, 1)

    # --- Feature Engineering Bonuses tailored specifically to the chosen Persona ---
    accumulated_bonus_score = np.zeros(len(candidate_dataframe))
    
    # Apply Buyer Specific Bonuses
    if forced_persona_id == 0: # Buy - Bargain Hunter
        days_on_market_array = candidate_dataframe.get('days_on_market', pd.Series(np.zeros(len(candidate_dataframe)))).values
        accumulated_bonus_score += np.clip(days_on_market_array / 365.0, 0, 0.2) # Max 0.2 bonus if property has sat on the market for a year (higher negotiating power)
        
    elif forced_persona_id == 1: # Buy - Family Weaver
        completion_status_array = candidate_dataframe.get('completion_status', pd.Series([''] * len(candidate_dataframe))).values
        accumulated_bonus_score += (completion_status_array == 'completed').astype(float) * 0.15 # Bonus for properties ready to inhabit immediately
        
    elif forced_persona_id == 2: # Buy - Luxury Elite
        listing_quality_scores_array = candidate_dataframe.get('listing_quality_score', pd.Series(np.zeros(len(candidate_dataframe)))).values
        accumulated_bonus_score += np.clip(listing_quality_scores_array, 0, 1) * 0.2 # Heavily reward highly detailed listings
        
    # Apply Renter Specific Bonuses
    elif forced_persona_id == 3: # Rent - Bargain Hunter
        furnished_status_array = candidate_dataframe.get('furnished', pd.Series([''] * len(candidate_dataframe))).values
        accumulated_bonus_score += (furnished_status_array == 'no').astype(float) * 0.15 # Unfurnished units avoid premium fees
        days_on_market_array = candidate_dataframe.get('days_on_market', pd.Series(np.zeros(len(candidate_dataframe)))).values
        accumulated_bonus_score += np.clip(days_on_market_array / 365.0, 0, 0.1)
        
    elif forced_persona_id == 4: # Rent - Family Weaver
        listing_quality_scores_array = candidate_dataframe.get('listing_quality_score', pd.Series(np.zeros(len(candidate_dataframe)))).values
        accumulated_bonus_score += np.clip(listing_quality_scores_array, 0, 1) * 0.1
        
    elif forced_persona_id == 5: # Rent - Luxury Elite
        furnished_status_array = candidate_dataframe.get('furnished', pd.Series([''] * len(candidate_dataframe))).values
        accumulated_bonus_score += (furnished_status_array == 'yes').astype(float) * 0.15 # Heavily reward fully furnished turnkey units
        listing_quality_scores_array = candidate_dataframe.get('listing_quality_score', pd.Series(np.zeros(len(candidate_dataframe)))).values
        accumulated_bonus_score += np.clip(listing_quality_scores_array, 0, 1) * 0.2
        
    # Apply Contrarian Logic
    elif forced_persona_id == 6: # Contrarian - Hidden Gem Hunter
        # Give a massive bonus to listings that have extremely high quality descriptions but cost < 70% of the query's stated maximum budget
        listing_quality_scores_array = candidate_dataframe.get('listing_quality_score', pd.Series(np.zeros(len(candidate_dataframe)))).values
        price_ratio_array = candidate_dataframe['price_egp'].values / (target_query_price + 1e-9)
        
        quality_bonus = np.clip(listing_quality_scores_array, 0, 1) * 0.3
        value_bonus = np.where(price_ratio_array < 0.7, 0.2, 0)
        accumulated_bonus_score += (quality_bonus + value_bonus) / 2.0

    # Apply Global Freshness Bias (Time Decay)
    # Convert 'days_on_market' to a freshness score. 0 days = 1.0, 365 days = ~0.0
    days_on_market = candidate_dataframe.get('days_on_market', pd.Series([100]*len(candidate_dataframe))).fillna(100).values
    freshness_score = np.clip(1.0 - (days_on_market / 365.0), 0, 1)

    # --- Calculate the Final Weighted Performance Score ---
    final_weighted_score = (
        (normalized_price_score * persona_weights['price']) +
        (is_exact_town_match * persona_weights['town']) +
        (combined_size_score * persona_weights['size']) +
        (normalized_amenity_score * persona_weights['amenities']) +
        (0.10 * freshness_score) + # 10% global bonus for fresh listings
        accumulated_bonus_score
    )
    
    # Normalize score back slightly to account for the freshness bonus
    final_weighted_score = final_weighted_score / 1.10

    # Calculate absolute mismatches to serve as "Kill Switches" for high ranking labels
    is_exact_city_match = (candidate_dataframe['city'].values == query_row_dictionary['city']).astype(int)
    is_exact_category_match = (candidate_dataframe['category'].values == query_row_dictionary['category']).astype(int)
    property_type_similarity_array = np.array([get_type_sim(query_row_dictionary['property_type'], candidate_type) for candidate_type in candidate_dataframe['property_type']])

    # Map the continuous Multi-Persona score into discrete categorical labels (1-4)
    # We inject tiny noise to simulate the slight inconsistencies in human evaluation
    simulated_human_noise = np.random.normal(0, 0.02, len(final_weighted_score))
    noisy_final_score = np.clip(final_weighted_score + simulated_human_noise, 0, 1)
    
    final_relevance_labels = np.zeros(len(candidate_dataframe))
    
    # HARD FILTER: If it's a mismatch between Rent and Buy, the relevance is strictly capped at 0.
    valid_category_mask = (is_exact_category_match == 1)
    
    # Level 4: Elite Match (Requires high personal score, exact city match, and high property type similarity)
    final_relevance_labels[valid_category_mask & (noisy_final_score >= 0.7) & (is_exact_city_match == 1) & (property_type_similarity_array >= 0.8)] = 4
    
    # Level 3: Strong Match
    final_relevance_labels[valid_category_mask & (final_relevance_labels == 0) & (noisy_final_score >= 0.5) & (is_exact_city_match == 1)] = 3
    
    # Level 2: Relevant Match (Acceptable fallback, potentially across city lines)
    final_relevance_labels[valid_category_mask & (final_relevance_labels == 0) & (noisy_final_score >= 0.3)] = 2
    
    # Level 1: Weak/Discovery Match (Barely relevant, used for broad catalog exploration)
    final_relevance_labels[valid_category_mask & (final_relevance_labels == 0) & (noisy_final_score >= 0.1)] = 1

    return final_relevance_labels.astype(int)
