import os
import sys
import pandas as pd
import importlib.util

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))

spec = importlib.util.spec_from_file_location("query_adapter", os.path.join(BASE_DIR, "2.Recommender", "5_query_adapter.py"))
query_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_adapter)
QueryAdapter = query_adapter.QueryAdapter

def test_regions():
    # 1. Test Hurghada (Red Sea)
    print("--- Testing Hurghada (Red Sea) ---")
    query_1 = {
        "category": "buy",
        "town": "Hurghada",
        "property_type": "Chalet"
    }
    results_1 = QueryAdapter.get_ranked_candidates(query_1, top_k=5)
    for i, cand in enumerate(results_1.get("candidates", []), 1):
        print(f"{i}. ID: {cand['listing_id']} | Price: {cand['price_egp']:,} | Town: {cand.get('town', 'N/A')} | City: {cand.get('city', 'N/A')}")

    # 2. Test North Coast
    print("\n--- Testing North Coast ---")
    query_2 = {
        "category": "rent",
        "town": "North Coast",
        "property_type": "Villa"
    }
    results_2 = QueryAdapter.get_ranked_candidates(query_2, top_k=5)
    for i, cand in enumerate(results_2.get("candidates", []), 1):
        print(f"{i}. ID: {cand['listing_id']} | Price: {cand['price_egp']:,} | Town: {cand.get('town', 'N/A')} | City: {cand.get('city', 'N/A')}")

if __name__ == "__main__":
    test_regions()
