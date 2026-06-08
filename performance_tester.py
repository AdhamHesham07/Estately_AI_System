import os
import sys
import time
import json
import uuid
import traceback
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Ensure relative imports from project root work
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))
sys.path.append(os.path.join(BASE_DIR, "3.Analyzer"))

load_dotenv(os.path.join(BASE_DIR, ".env"))

import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

QueryAdapter_module = load_module_from_path("query_adapter", os.path.join(BASE_DIR, "2.Recommender", "5_query_adapter.py"))
QueryAdapter = QueryAdapter_module.QueryAdapter

MarketEngine_module = load_module_from_path("market_engine", os.path.join(BASE_DIR, "3.Analyzer", "3_market_engine.py"))
FairPriceEstimator = MarketEngine_module.FairPriceEstimator
MarketPulse = MarketEngine_module.MarketPulse
AreaComparator = MarketEngine_module.AreaComparator

GraphBuilder_module = load_module_from_path("graph_builder", os.path.join(BASE_DIR, "1.Agent", "4_graph_builder.py"))
build_agent_graph = GraphBuilder_module.build_agent_graph

# =========================================================
# REPORTING STRUCTURE
# =========================================================
performance_report = {
    "Recommender": {"tests": [], "summary": {}},
    "Analyzer": {"tests": [], "summary": {}},
    "Agent": {"tests": [], "summary": {}}
}

def record_test(module, test_name, latency, quality_score, is_valid, details):
    performance_report[module]["tests"].append({
        "name": test_name,
        "latency_sec": round(latency, 4),
        "quality_score": quality_score, # Can be Confidence Score, relevance_score average, etc.
        "is_valid": is_valid,
        "details": details
    })
    print(f"[{module}] {test_name}: {latency:.4f}s | Valid: {is_valid} | Quality: {quality_score}")

# =========================================================
# 1. RECOMMENDER TESTS
# =========================================================
def test_recommender():
    print("\n--- Testing RECOMMENDER ---")
    
    # 1.1 Cold Start Time (Loading Models)
    start_time = time.perf_counter()
    try:
        models = QueryAdapter._load_models()
        latency = time.perf_counter() - start_time
        is_valid = "xgboost_ranker" in models and "faiss_search_index" in models
        record_test("Recommender", "Cold_Start_Load", latency, "N/A", is_valid, "Loaded models successfully")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Recommender", "Cold_Start_Load", latency, "N/A", False, f"Error: {str(e)}")
        return

    # 1.2 Query: Valid Specific Query
    query_1 = {
        "category": "buy",
        "city": "Cairo",
        "town": "New Cairo City",
        "property_type": "Apartment",
        "bedrooms": 3,
        "price_max": 10000000
    }
    start_time = time.perf_counter()
    try:
        results = QueryAdapter.get_ranked_candidates(query_1, top_k=10)
        latency = time.perf_counter() - start_time
        
        candidates = results.get("candidates", [])
        is_valid = len(candidates) > 0 and isinstance(candidates, list) and not results.get("is_vague", True)
        
        # Calculate Quality: Average Relevance Score of Top Candidates
        avg_relevance = sum(c.get("relevance_score", 0) for c in candidates) / len(candidates) if candidates else 0
        
        record_test("Recommender", "Specific_Query_Buy", latency, round(avg_relevance, 4), is_valid, f"Returned {len(candidates)} candidates")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Recommender", "Specific_Query_Buy", latency, 0, False, f"Error: {str(e)}")

    # 1.3 Query: Vague Query Rejection/Handling
    query_vague = {
        "category": "buy"
        # Missing location and budget
    }
    start_time = time.perf_counter()
    try:
        results = QueryAdapter.get_ranked_candidates(query_vague, top_k=10)
        latency = time.perf_counter() - start_time
        
        is_valid = results.get("is_vague", False) or len(results.get("candidates", [])) == 0
        record_test("Recommender", "Vague_Query_Handling", latency, "N/A", is_valid, "Correctly handled vague/rejected query")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Recommender", "Vague_Query_Handling", latency, "N/A", False, f"Error: {str(e)}")


# =========================================================
# 2. ANALYZER TESTS
# =========================================================
def test_analyzer():
    print("\n--- Testing ANALYZER ---")
    
    # 2.1 Fair Price Estimator
    start_time = time.perf_counter()
    try:
        result = FairPriceEstimator.estimate(
            category_intent="buy",
            town_name="New Cairo City",
            district_name="",
            property_type="Apartment",
            number_of_bedrooms=3,
            asking_price=5200000,
            property_area_sqm=150
        )
        latency = time.perf_counter() - start_time
        is_valid = result.get("status") == "success" and "market_stats" in result
        quality = result.get("market_stats", {}).get("confidence_score", 0) if is_valid else 0
        record_test("Analyzer", "Fair_Price_Estimator", latency, quality, is_valid, result.get("verdict", "N/A"))
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Analyzer", "Fair_Price_Estimator", latency, 0, False, f"Error: {str(e)}")

    # 2.2 Market Pulse
    start_time = time.perf_counter()
    try:
        result = MarketPulse.get_snapshot()
        latency = time.perf_counter() - start_time
        is_valid = "market_volume" in result and "price_economics" in result
        quality = "High" if is_valid and result["market_volume"]["total_active_listings"] > 0 else "Low"
        record_test("Analyzer", "Market_Pulse", latency, quality, is_valid, f"Total Listings: {result.get('market_volume', {}).get('total_active_listings', 0)}")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Analyzer", "Market_Pulse", latency, "N/A", False, f"Error: {str(e)}")

    # 2.3 Area Comparator
    start_time = time.perf_counter()
    try:
        result = AreaComparator.compare("buy", "New Cairo City", "Sheikh Zayed City", "Apartment")
        latency = time.perf_counter() - start_time
        is_valid = result.get("status") == "success" and "comparison" in result
        record_test("Analyzer", "Area_Comparator", latency, "N/A", is_valid, "Successfully compared two areas")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Analyzer", "Area_Comparator", latency, "N/A", False, f"Error: {str(e)}")


# =========================================================
# 3. AGENT TESTS
# =========================================================
def test_agent():
    print("\n--- Testing AGENT ---")
    
    # 3.1 Agent Graph Building
    start_time = time.perf_counter()
    try:
        agent_app = build_agent_graph()
        latency = time.perf_counter() - start_time
        is_valid = agent_app is not None
        record_test("Agent", "Build_Graph", latency, "N/A", is_valid, "Graph built successfully")
    except Exception as e:
        latency = time.perf_counter() - start_time
        record_test("Agent", "Build_Graph", latency, "N/A", False, f"Error: {str(e)}")
        return

    # 3.2 End-to-End User Message Processing
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    user_inputs = [
        "Hi, I am looking for a house.",
        "I want a 3 bedroom apartment in New Cairo, max budget 6 million."
    ]
    
    state = {
        "messages": [],
        "current_filters": {},
        "tool_outputs": {},
        "missing_info": [],
        "active_intent": "idle",
        "booking_details": {},
        "confidence_score": 1.0,
        "is_out_of_domain": False,
        "audit_retries": 0,
        "user_language": "en",
        "recent_properties": [],
        "discussion_context": {},
        "dialogue_state": {},
        "reference_map": {},
        "focus_listing_id": None,
        "last_recommendation_snapshot": [],
        "response_mode": "tool_required",
        "dialogue_act": "general",
        "target_property_refs": [],
        "carry_forward_slots": {},
        "slot_updates": {},
        "response_plan": {},
    }

    for idx, user_input in enumerate(user_inputs):
        state["messages"].append(HumanMessage(content=user_input))
        start_time = time.perf_counter()
        try:
            result = agent_app.invoke(state, config)
            latency = time.perf_counter() - start_time
            
            # Check validity
            is_valid = "messages" in result and len(result["messages"]) > 0
            
            # Update state for next turn
            state = result
            
            # Check quality (intent classification, confidence)
            intent = result.get("active_intent", "unknown")
            confidence = result.get("confidence_score", 0)
            
            record_test("Agent", f"Message_{idx+1}_Response", latency, intent, is_valid, f"Confidence: {confidence}")
        except Exception as e:
            latency = time.perf_counter() - start_time
            record_test("Agent", f"Message_{idx+1}_Response", latency, "Error", False, traceback.format_exc())

# =========================================================
# RUNNER & REPORTER
# =========================================================
def main():
    test_recommender()
    test_analyzer()
    test_agent()
    
    # Save Report
    report_path = os.path.join(BASE_DIR, "performance_report_raw.json")
    with open(report_path, "w") as f:
        json.dump(performance_report, f, indent=4)
        
    print(f"\n[DONE] Performance testing complete. Raw data saved to {report_path}.")

if __name__ == "__main__":
    main()
