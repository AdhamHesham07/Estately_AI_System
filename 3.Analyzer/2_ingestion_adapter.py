import os
import sys
import pandas as pd
import importlib
import logging

# Ensure cross-module imports work to reach the global Data directory
BASE_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIRECTORY_PATH, "4.Data"))

# Import the centralized preprocessing logic
from preprocessing_engine import preprocess

# Import the Analyzer configurations
config_module = importlib.import_module("1_config")
ANALYZER_CONFIG = config_module.ANALYZER_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ingestion_logger = logging.getLogger(__name__)

# In-memory cache to prevent re-processing the dataset on every analytical query
_GLOBAL_ANALYZER_DATAFRAME_CACHE = None

def get_ingested_catalog() -> pd.DataFrame:
    """
    Acts as the Live Ingester for the Analyzer Module.
    Calls the shared Preprocessing Engine so the Analyzer processes the exact same 
    cleaned, outlier-free dataset as the Recommender, avoiding logical mismatches.
    """
    global _GLOBAL_ANALYZER_DATAFRAME_CACHE
    
    # Return the cached dataset if it has already been loaded into RAM
    if _GLOBAL_ANALYZER_DATAFRAME_CACHE is None:
        ingestion_logger.info("Ingesting Cleaned Data via Preprocessing Engine...")
        try:
            # Reusing the robust logic from the Recommender pipeline to guarantee consistency
            highly_cleaned_dataframe, tracked_amenities = preprocess()
            
            # Enforce crucial string types for reliable grouping and analytical joins
            highly_cleaned_dataframe['listing_id'] = highly_cleaned_dataframe['listing_id'].astype(str)
            
            # Store in the global cache and reset indexes for clean loop iterations
            _GLOBAL_ANALYZER_DATAFRAME_CACHE = highly_cleaned_dataframe.reset_index(drop=True)
            ingestion_logger.info(f"Successfully ingested {len(highly_cleaned_dataframe)} Cleaned records into the Analyzer.")
            
        except Exception as ingestion_error:
            ingestion_logger.error(f"Failed to ingest data: {ingestion_error}")
            raise
            
    return _GLOBAL_ANALYZER_DATAFRAME_CACHE
