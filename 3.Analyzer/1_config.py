import os

# Set up the baseline directory structure for Analyzer artifacts
BASE_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Global configurations for the analytical engine
ANALYZER_CONFIG = {
    "paths": {
        "contracts_dir": os.path.join(BASE_DIRECTORY_PATH, "4.Data", "1_Contracts"),
        "cache": os.path.join(BASE_DIRECTORY_PATH, "3.Analyzer", "cache")
    },
    "thresholds": {
        # Mathematical boundaries for statistical validity
        "min_sample_size": 10,           # Refuse to run analysis if fewer than 10 comparable properties exist
        "mape_target": 0.15,             # Mean Absolute Percentage Error acceptable threshold (15%)
        "premium_iqr_multiplier": 1.5    # Multiplier for the Interquartile Range to detect extreme outliers/luxuries
    },
    "agent": {
        # The default LLM responsible for translating analytical stats into natural language insights
        "model": "gemini/gemini-1.5-flash-latest"
    }
}
