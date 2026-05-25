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
        
        # --- SCENARIO A: PRICED OUT OR TOO RESTRICTIVE (0-2 Matches) ---
        if exact_matches_count <= 2:
            # Tactic 1: Budget Expansion
            if target_budget > 0:
                budget_boost = target_budget * 1.3  # 30% increase
                boost_mask = base_mask & (global_db['price_egp'] <= budget_boost)
                if target_bedrooms > 0:
                    boost_mask = boost_mask & (global_db['bedrooms'] == target_bedrooms)
                boost_count = len(global_db[boost_mask])
                if boost_count > exact_matches_count:
                    suggested_adjustments.append({
                        "strategy": "Budget Expansion",
                        "change": f"Increase budget by 30% (to {budget_boost:,.0f} EGP)",
                        "delta": f"+{boost_count - exact_matches_count} high-quality options unlocked."
                    })
            
            # Tactic 2: Location Shift (Emerging Markets)
            if target_town == "new cairo city" and target_budget > 0:
                alt_mask = (global_db['category'].astype(str).str.lower() == target_category) & \
                           (global_db['town'].astype(str).str.lower() == "mostakbal city") & \
                           (global_db['price_egp'] <= target_budget)
                if target_bedrooms > 0:
                    alt_mask = alt_mask & (global_db['bedrooms'] == target_bedrooms)
                alt_count = len(global_db[alt_mask])
                if alt_count > exact_matches_count:
                    suggested_adjustments.append({
                        "strategy": "Location Pivot",
                        "change": "Shift focus to Mostakbal City (adjacent to New Cairo)",
                        "delta": f"Provides {alt_count} excellent options within your current budget."
                    })
                    
            # Tactic 3: Property Type Shift (Downsizing)
            if target_property_type == "villa" and target_budget > 0:
                type_mask = (global_db['category'].astype(str).str.lower() == target_category) & \
                            (global_db['property_type'].astype(str).str.lower().isin(['townhouse', 'twinhouse'])) & \
                            (global_db['price_egp'] <= target_budget)
                if target_town:
                    type_mask = type_mask & (global_db['town'].astype(str).str.lower() == target_town)
                type_count = len(global_db[type_mask])
                if type_count > exact_matches_count:
                    suggested_adjustments.append({
                        "strategy": "Property Type Pivot",
                        "change": "Consider a Townhouse or Twinhouse instead of a Standalone Villa",
                        "delta": f"Unlocks {type_count} premium properties without breaking the budget."
                    })
                    
            # Tactic 4: Completion Status Shift (Off-Plan)
            if target_budget > 0:
                offplan_mask = base_mask & (global_db['price_egp'] <= target_budget) & \
                               (global_db['completion_status'].astype(str).str.lower() == 'under construction')
                offplan_count = len(global_db[offplan_mask])
                if offplan_count > exact_matches_count:
                    suggested_adjustments.append({
                        "strategy": "Investment Strategy Shift",
                        "change": "Focus on Off-Plan properties with extended installment plans (7-10 years)",
                        "delta": f"Reveals {offplan_count} developer-direct opportunities."
                    })

        # --- SCENARIO B: TIGHT MARKET (3-10 Matches) ---
        elif exact_matches_count <= 10:
            if target_bedrooms > 2:
                flex_beds_mask = base_mask & (global_db['bedrooms'] >= (target_bedrooms - 1))
                if target_budget > 0:
                    flex_beds_mask = flex_beds_mask & (global_db['price_egp'] <= target_budget)
                flex_count = len(global_db[flex_beds_mask])
                if flex_count > exact_matches_count:
                    suggested_adjustments.append({
                        "strategy": "Bedroom Flexibility",
                        "change": f"Be open to {target_bedrooms - 1} bedrooms",
                        "delta": f"Expands your options to {flex_count} properties."
                    })

        # --- SCENARIO C: SPOILED FOR CHOICE (15+ Matches) ---
        elif exact_matches_count > 15:
            suggested_adjustments.append({
                "strategy": "Premium Filtering",
                "change": "You have a massive abundance of options in this bracket.",
                "delta": "I highly recommend filtering by Tier-1 developers (like Emaar, SODIC, Palm Hills) or focusing purely on Ready-to-Move options to narrow down the absolute best investments."
            })
            
        return {
            "status": "success",
            "catalog_exact_matches": exact_matches_count,
            "adjustments": suggested_adjustments
        }
        
    @staticmethod
    def generate_tradeoff_advisor(filters: dict = None) -> dict:
        """Alias for DataBridge compatibility."""
        return PreferenceEngine.evaluate_tradeoffs(filters or {})
