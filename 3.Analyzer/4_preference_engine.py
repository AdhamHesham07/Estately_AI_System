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
    LAYER 2: PREFERENCE-AWARE ANALYTICS (The "Master Strategist")
    Evaluates user constraints against market reality and identifies highly sophisticated 
    qualitative compromises or opportunity highlights before the recommender runs.
    """
    
    @staticmethod
    def _normalize_string(string_value) -> str:
        """Helper to safely lower and strip strings for pandas boolean masking."""
        if not string_value:
            return ""
        return str(string_value).strip().lower()

    @staticmethod
    def evaluate_tradeoffs(filters: dict) -> dict:
        """
        The multi-scenario decision tree.
        Analyzes the user's constraints and suggests strategic pivots based on the current market depth.
        """
        global_db = get_global_clean_data()
        
        target_category = PreferenceEngine._normalize_string(filters.get('category', 'buy'))
        target_town = PreferenceEngine._normalize_string(filters.get('town', ''))
        target_property_type = PreferenceEngine._normalize_string(filters.get('property_type', ''))
        
        try:
            target_bedrooms = int(filters.get('bedrooms', 0)) if filters.get('bedrooms') else 0
        except ValueError:
            target_bedrooms = 0
            
        try:
            target_budget = float(filters.get('price_max', filters.get('price_egp', 0)))
        except ValueError:
            target_budget = 0.0
            
        # 1. Build Base Masks
        base_mask = (global_db['category'].astype(str).str.lower() == target_category)
        if target_town:
            base_mask = base_mask & (global_db['town'].astype(str).str.lower() == target_town)
        if target_property_type:
            base_mask = base_mask & (global_db['property_type'].astype(str).str.lower() == target_property_type)
            
        # Exact Matches
        exact_mask = base_mask.copy()
        if target_bedrooms > 0:
            exact_mask = exact_mask & (global_db['bedrooms'] == target_bedrooms)
        if target_budget > 0:
            exact_mask = exact_mask & (global_db['price_egp'] <= target_budget)
            
        exact_matches_count = len(global_db[exact_mask])
        suggested_adjustments = []
        
        # Dynamically generate market summary for the LLM context
        try:
            market_engine = importlib.import_module("3_market_engine")
            market_snapshot = market_engine.MarketPulse.get_snapshot(filters)
            market_summary = json.dumps(market_snapshot, indent=2)
        except Exception as e:
            market_summary = "Market data unavailable."
            
        # Call the LLM instead of using hardcoded rules
        try:
            analyzer_llm = importlib.import_module("8_analyzer_llm")
            result = analyzer_llm.AnalyzerSynthesizer.generate_ai_tradeoffs(filters, exact_matches_count, market_summary)
            return result
        except Exception as e:
            print(f"Failed to use LLM for tradeoffs: {e}")
            return {
                "status": "success",
                "catalog_exact_matches": exact_matches_count,
                "adjustments": []
            }
        
    @staticmethod
    def generate_tradeoff_advisor(filters: dict = None) -> dict:
        """Alias for DataBridge compatibility."""
        return PreferenceEngine.evaluate_tradeoffs(filters or {})
