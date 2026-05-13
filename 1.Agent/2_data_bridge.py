import os
import sys
import json
import importlib
import logging

# Ensure cross-module imports work by dynamically adjusting the system path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "4.Data"))
sys.path.append(os.path.join(BASE_DIR, "4.Data", "2_DataBase"))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))
sys.path.append(os.path.join(BASE_DIR, "3.Analyzer"))

# Load necessary external dependencies
from db_adapter import DBAdapter
import preprocessing_engine

# Dynamically import modules to prevent circular dependencies
query_adapter = importlib.import_module("5_query_adapter")
insight_builder = importlib.import_module("5_insight_builder")
market_engine = importlib.import_module("3_market_engine")

logger = logging.getLogger(__name__)

class DataBridge:
    """
    The 'Hands' of the Agent.
    Bridges the LangGraph conversational state to the production Recommender and Analyzer logic.
    """
    
    # Caches the locations available in the live database to avoid constant DB polling
    _location_grounding_data = None

    @classmethod
    def _get_grounding_data(cls):
        """
        Loads valid towns/cities from the LIVE SQLite database and caches them.
        This provides a definitive vocabulary for user location matching.
        """
        if cls._location_grounding_data is not None:
            return cls._location_grounding_data
        
        try:
            logger.info("Grounding Agent with Live Database Locations...")
            property_dataframe = DBAdapter.fetch_properties()
            
            # Extract unique valid names for hierarchical location levels
            cls._location_grounding_data = {
                "towns": property_dataframe['town'].dropna().unique().tolist(),
                "districts": property_dataframe['district'].dropna().unique().tolist() if 'district' in property_dataframe.columns else [],
                "subdistricts": property_dataframe['subdistrict'].dropna().unique().tolist() if 'subdistrict' in property_dataframe.columns else []
            }
            
            # Flatten all valid names into one list for rapid fuzzy matching
            cls._all_valid_location_names = list(set(
                cls._location_grounding_data["towns"] + 
                cls._location_grounding_data["districts"] + 
                cls._location_grounding_data["subdistricts"]
            ))
            return cls._location_grounding_data
        except Exception as e:
            logger.error(f"Failed to load grounding data: {e}")
            return {"towns": []}

    @staticmethod
    def execute_fair_price_check(prop: dict, category: str) -> str:
        """
        Calls the Analyzer module to verify if a recommended property is fairly priced.
        Returns a narrative string explaining the value proposition.
        """
        try:
            # Query the core market engine for valuation stats
            fair_price_estimation_result = market_engine.FairPriceEstimator.estimate(
                category=category,
                town=prop.get('town', ''),
                district=prop.get('district', ''),
                property_type=prop.get('property_type', ''),
                bedrooms=int(prop.get('bedrooms', 0)),
                asking_price=float(prop.get('price_egp', 0)),
                row_area=float(prop.get('area_value', 0)),
                is_furnished=True if str(prop.get('furnished', '')).lower() == 'furnished' else False
            )
            
            # Format the raw stats into a clean, human-readable context block
            return insight_builder.LLMContextBuilder.prepare_fair_price_context(fair_price_estimation_result)
        except Exception:
            return ""  # Silently skip — never expose analysis errors to the LLM prompt

    @staticmethod
    def resolve_location(location_name: str) -> dict:
        """
        Expert Logic: Uses Fuzzy Matching to cleanly map user input to database-native values.
        E.g., mapping 'Shiek Zayed' to 'Sheikh Zayed City' for optimal querying.
        """
        if not location_name:
            return {"value": "", "level": None, "score": 0}
            
        try:
            from rapidfuzz import process, utils
        except ImportError:
            logger.warning("RapidFuzz not found. Falling back to simple matching.")
            return {"value": location_name, "level": None, "score": 0}

        grounding_data = DataBridge._get_grounding_data()
        
        # Prepare arrays for hierarchical location matching
        location_level_mappings = {
            "town": grounding_data.get("towns", []),
            "district": grounding_data.get("districts", []),
            "subdistrict": grounding_data.get("subdistricts", [])
        }
        
        flattened_location_options = []
        for names in location_level_mappings.values():
            flattened_location_options.extend([name for name in names if name])
        
        if not flattened_location_options:
            return {"value": location_name, "level": None, "score": 0}

        # Perform the actual fuzzy match with a reasonable confidence cutoff
        fuzzy_match_result = process.extractOne(
            location_name, 
            flattened_location_options,
            processor=utils.default_process,
            score_cutoff=60 # Minimum confidence threshold
        )
        
        # If a match is found, deduce its hierarchical level (e.g., is it a town or district?)
        if fuzzy_match_result:
            matched_location_value, match_score, _ = fuzzy_match_result
            matched_location_level = None
            normalized_input_string = str(matched_location_value).strip().lower()
            
            for level, names in location_level_mappings.items():
                if any(str(name).strip().lower() == normalized_input_string for name in names):
                    matched_location_level = level
                    break
                    
            logger.info(f"Grounding: '{location_name}' matched to '{matched_location_value}' as {matched_location_level} (Score: {match_score})")
            return {"value": matched_location_value, "level": matched_location_level, "score": match_score}
        
        return {"value": location_name, "level": None, "score": 0}

    @staticmethod
    def normalize_location(location_name: str) -> str:
        """Backward-compatible location normalizer."""
        return DataBridge.resolve_location(location_name).get("value", location_name)

    @staticmethod
    def execute_recommendation(filters: dict) -> dict:
        """
        Calls the backend Recommender system and returns the candidate properties.
        Automatically intercepts and normalizes locations before querying.
        """
        # Iterate over location fields to ensure they align exactly with the database schema
        for location_filter_key in ("town", "district", "subdistrict"):
            if location_filter_key in filters and filters.get(location_filter_key):
                resolved_location_data = DataBridge.resolve_location(filters[location_filter_key])
                normalized_location_value = resolved_location_data.get("value")
                detected_location_level = resolved_location_data.get("level")
                
                # Apply the fixed value
                if normalized_location_value:
                    filters[location_filter_key] = normalized_location_value
                
                # Relocate the filter to the correct hierarchical tier if misassigned
                if detected_location_level and detected_location_level != location_filter_key:
                    filters[detected_location_level] = normalized_location_value
            
        try:
            # Query the core engine for top 10 ranked candidates
            return query_adapter.QueryAdapter.get_ranked_candidates(filters, top_k=10)
        except Exception as e:
            return {"error": str(e), "candidates": []}

    @staticmethod
    def execute_analysis(analysis_type: str, data: dict) -> str:
        """
        Calls the Analyzer module to generate structured context strings for the LLM.
        """
        context_builder = insight_builder.LLMContextBuilder
        
        try:
            # Route to the appropriate builder function based on analysis requested
            if analysis_type == "fair_price":
                return context_builder.prepare_fair_price_context(data)
            elif analysis_type == "market_pulse":
                return context_builder.prepare_market_pulse_context(data)
            elif analysis_type == "segment_report":
                return context_builder.prepare_segment_report_context(data)
        except Exception:
            return ""  # Silently skip

        return ""  # No matching analysis type — return empty, not an error string

if __name__ == "__main__":
    # Test execution block to verify fuzzy matching logic locally
    print(f"Normalizing 'Zayed': {DataBridge.normalize_location('Zayed')}")
    print(f"Normalizing 'New Cairo': {DataBridge.normalize_location('New Cairo')}")
