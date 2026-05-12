import os
import json
import pandas as pd
import importlib

config_module = importlib.import_module("1_config")
ANALYZER_CONFIG = config_module.ANALYZER_CONFIG

ingestion_adapter = importlib.import_module("2_ingestion_adapter")
get_global_clean_data = ingestion_adapter.get_ingested_catalog

class PreferenceEngine:
    """
    LAYER 2: PREFERENCE-AWARE ANALYTICS
    Unlike the General Market Engine (which looks at the whole city), this engine 
    specifically consumes the Recommender's Top Candidate pool to generate 
    hyper-personalized insights for the specific user's exact preferences.
    """
    
    @staticmethod
    def _normalize_string(string_value) -> str:
        """Helper to safely lower and strip strings for pandas boolean masking."""
        return str(string_value).strip().lower()

    @staticmethod
    def load_candidate_contract() -> dict:
        """
        Reads the personalized candidate list exported by the Recommender module.
        This allows the Analyzer to know exactly what the user is currently looking at.
        """
        candidate_contract_path = os.path.join(ANALYZER_CONFIG["paths"]["contracts_dir"], "Rec_To_Ana_Candidates.json")
        try:
            with open(candidate_contract_path, 'r') as file_handler:
                return json.load(file_handler)
        except FileNotFoundError:
            return None

    @staticmethod
    def generate_segment_report() -> dict:
        """
        2.1 "Your Market" Segment Report
        Treats the Recommender's candidate pool as the user's "personal micro-market", 
        calculating medians and ranges specifically for the properties they were just recommended.
        """
        candidate_payload = PreferenceEngine.load_candidate_contract()
        if not candidate_payload:
            return {"status": "error", "message": "No candidate contract found."}
            
        user_query_context = candidate_payload['query']
        recommended_candidates_dataframe = pd.DataFrame(candidate_payload['candidates'])
        
        if len(recommended_candidates_dataframe) == 0:
            return {"status": "error", "message": "No candidates provided by Recommender."}
            
        segment_report = {
            "query_context": user_query_context,
            "segment_size": len(recommended_candidates_dataframe),
            "stats": {
                "median_price": float(recommended_candidates_dataframe['price_egp'].median()),
                "price_range": [float(recommended_candidates_dataframe['price_egp'].min()), float(recommended_candidates_dataframe['price_egp'].max())],
                "avg_price_per_sqm": float(recommended_candidates_dataframe['price_sqft'].median() * 10.764) if 'price_sqft' in recommended_candidates_dataframe else None,
                "avg_area": float(recommended_candidates_dataframe['area_value'].mean()),
                "completion_mix": recommended_candidates_dataframe['completion_status'].value_counts(normalize=True).apply(lambda percentage: round(percentage*100, 1)).to_dict(),
                "payment_mix": recommended_candidates_dataframe['payment_method'].value_counts(normalize=True).apply(lambda percentage: round(percentage*100, 1)).to_dict()
            }
        }
        return {"status": "success", "report": segment_report}

    @staticmethod
    def generate_tradeoff_advisor() -> dict:
        """
        2.3 Trade-off Advisor
        Looks at the global catalog to tell the user what would happen if they 
        relaxed certain constraints (e.g., "If you increase your budget by 20%, 
        you unlock 50 more properties").
        """
        candidate_payload = PreferenceEngine.load_candidate_contract()
        if not candidate_payload:
            return {"status": "error", "message": "No candidate contract found."}
            
        user_query_dictionary = candidate_payload['query']
        global_database_dataframe = get_global_clean_data()
        
        current_exact_matches_count = len(candidate_payload['candidates'])
        
        # Safely extract and normalize the user's current constraints
        target_category = PreferenceEngine._normalize_string(user_query_dictionary.get('category', ''))
        target_town = PreferenceEngine._normalize_string(user_query_dictionary.get('town', ''))
        target_property_type = PreferenceEngine._normalize_string(user_query_dictionary.get('property_type', ''))
        target_bedrooms = user_query_dictionary.get('bedrooms')
        target_budget = user_query_dictionary.get('price_egp', user_query_dictionary.get('price_max', 0))
        
        # Scenario 1: What if the user increases their budget by 20%?
        budget_increase_threshold = target_budget * 1.2
        base_market_mask = (
            (global_database_dataframe['category'].astype(str).str.lower() == target_category) & 
            (global_database_dataframe['town'].astype(str).str.lower() == target_town) & 
            (global_database_dataframe['property_type'].astype(str).str.lower() == target_property_type) &
            (global_database_dataframe['bedrooms'] == target_bedrooms)
        )
        
        original_budget_mask = base_market_mask & (global_database_dataframe['price_egp'] <= target_budget)
        increased_budget_mask = base_market_mask & (global_database_dataframe['price_egp'] <= budget_increase_threshold)
        
        catalog_original_matches_count = len(global_database_dataframe[original_budget_mask])
        catalog_new_matches_count = len(global_database_dataframe[increased_budget_mask])
        
        suggested_adjustments = []
        if catalog_new_matches_count > catalog_original_matches_count:
            suggested_adjustments.append({
                "change": f"Increase budget by 20% (to {budget_increase_threshold:,.0f} EGP)",
                "new_matches": catalog_new_matches_count,
                "delta": f"+{catalog_new_matches_count - catalog_original_matches_count} properties"
            })
            
        # Scenario 2: What if the user drops the strict bedroom requirement?
        flexible_beds_mask = (
            (global_database_dataframe['category'].astype(str).str.lower() == target_category) & 
            (global_database_dataframe['town'].astype(str).str.lower() == target_town) & 
            (global_database_dataframe['property_type'].astype(str).str.lower() == target_property_type) &
            (global_database_dataframe['price_egp'] <= target_budget)
        )
        
        flexible_beds_count = len(global_database_dataframe[flexible_beds_mask])
        if flexible_beds_count > catalog_original_matches_count:
            suggested_adjustments.append({
                "change": "Be flexible on bedroom count",
                "new_matches": flexible_beds_count,
                "delta": f"+{flexible_beds_count - catalog_original_matches_count} properties"
            })
            
        return {
            "status": "success",
            "current_matches": current_exact_matches_count,
            "catalog_exact_matches": catalog_original_matches_count,
            "adjustments": suggested_adjustments
        }
