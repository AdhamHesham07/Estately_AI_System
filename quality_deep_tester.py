import os
import sys
import uuid
import traceback
import importlib.util
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))
sys.path.append(os.path.join(BASE_DIR, "3.Analyzer"))

load_dotenv(os.path.join(BASE_DIR, ".env"))

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load Modules
QueryAdapter_module = load_module_from_path("query_adapter", os.path.join(BASE_DIR, "2.Recommender", "5_query_adapter.py"))
QueryAdapter = QueryAdapter_module.QueryAdapter

MarketEngine_module = load_module_from_path("market_engine", os.path.join(BASE_DIR, "3.Analyzer", "3_market_engine.py"))
FairPriceEstimator = MarketEngine_module.FairPriceEstimator
AreaComparator = MarketEngine_module.AreaComparator

GraphBuilder_module = load_module_from_path("graph_builder", os.path.join(BASE_DIR, "1.Agent", "4_graph_builder.py"))
build_agent_graph = GraphBuilder_module.build_agent_graph

# Store results
results = {
    "Recommender": [],
    "Analyzer": [],
    "Agent": []
}

def log_test(module, name, scenario, passed, details):
    results[module].append({"name": name, "scenario": scenario, "passed": passed, "details": details})
    status = "PASS" if passed else "FAIL/IMPERFECTION"
    print(f"[{module}] {name} -> {status}\n  Details: {details}\n")

# =========================================================
# 1. RECOMMENDER DEEP TESTS
# =========================================================
def test_recommender_imperfections():
    print("====== RECOMMENDER DEEP TESTS ======")
    try:
        QueryAdapter._load_models()
    except Exception as e:
        print(f"Failed to load recommender models: {e}")
        return

    # Edge Case 1: Extreme Misspelling (Should not crash, should either find semantic matches or return empty/vague safely)
    q1 = {"category": "buy", "city": "Alxndria", "town": "Gleem", "budget": 5000000, "property_type": "Apartment"}
    try:
        res = QueryAdapter.get_ranked_candidates(q1, top_k=5)
        # Imperfection: Does it hallucinate a wrong city or just return 0?
        if len(res.get("candidates", [])) == 0:
            log_test("Recommender", "Extreme_Misspelling", "Querying non-existent/misspelled city", True, "Gracefully returned 0 candidates without crashing.")
        else:
            log_test("Recommender", "Extreme_Misspelling", "Querying non-existent/misspelled city", False, f"Hallucinated {len(res['candidates'])} matches from wrong cities.")
    except Exception as e:
        log_test("Recommender", "Extreme_Misspelling", "Crash test", False, f"CRASHED: {str(e)}")

    # Edge Case 2: Constraint Contradiction (Luxury Villa for 10,000 EGP)
    q2 = {"category": "buy", "city": "Cairo", "town": "New Cairo City", "property_type": "Villa", "bedrooms": 6, "price_max": 10000}
    try:
        res = QueryAdapter.get_ranked_candidates(q2, top_k=5)
        cands = res.get("candidates", [])
        if len(cands) == 0:
            log_test("Recommender", "Constraint_Contradiction", "6-bed Villa for 10k EGP", True, "Correctly found 0 candidates.")
        else:
            # Check if it returned things far outside the budget because it prioritized semantic "Villa" features
            prices = [c.get("price_egp", 0) for c in cands]
            log_test("Recommender", "Constraint_Contradiction", "6-bed Villa for 10k EGP", False, f"Ignored strict budget and returned properties costing {prices}")
    except Exception as e:
        log_test("Recommender", "Constraint_Contradiction", "Crash test", False, f"CRASHED: {str(e)}")


# =========================================================
# 2. ANALYZER DEEP TESTS
# =========================================================
def test_analyzer_imperfections():
    print("====== ANALYZER DEEP TESTS ======")
    
    # Edge Case 1: Mathematical Extremes (Zero Area)
    try:
        res = FairPriceEstimator.estimate("buy", "New Cairo City", "", "Apartment", 3, 5000000, 0)
        if res.get("status") == "error":
            log_test("Analyzer", "Zero_Area_Math", "Passed 0 area to see if division by zero occurs", True, f"Caught gracefully: {res.get('message')}")
        else:
            log_test("Analyzer", "Zero_Area_Math", "Passed 0 area", False, "Returned success despite 0 area. Possible math corruption.")
    except Exception as e:
        log_test("Analyzer", "Zero_Area_Math", "Passed 0 area", False, f"CRASHED with exception: {str(e)}")

    # Edge Case 2: Data Sparsity (Town with no data)
    try:
        res = AreaComparator.compare("buy", "Atlantis City", "El Dorado")
        if res.get("status") == "error":
            log_test("Analyzer", "Extreme_Sparsity", "Compare two non-existent cities", True, f"Handled gracefully: {res.get('message')}")
        else:
            log_test("Analyzer", "Extreme_Sparsity", "Compare two non-existent cities", False, "Attempted to compare phantom cities, logic failure.")
    except Exception as e:
        log_test("Analyzer", "Extreme_Sparsity", "Compare two non-existent cities", False, f"CRASHED: {str(e)}")


# =========================================================
# 3. AGENT DEEP TESTS (Adversarial)
# =========================================================
def test_agent_imperfections():
    print("====== AGENT DEEP TESTS ======")
    try:
        agent_app = build_agent_graph()
    except Exception as e:
        print(f"Failed to build agent graph: {e}")
        return

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    def run_agent_turn(state, text):
        state["messages"].append(HumanMessage(content=text))
        try:
            return agent_app.invoke(state, config), None
        except Exception as e:
            return None, str(e)

    base_state = {
        "messages": [], "current_filters": {}, "tool_outputs": {}, "missing_info": [],
        "active_intent": "idle", "booking_details": {}, "confidence_score": 1.0,
        "is_out_of_domain": False, "audit_retries": 0, "user_language": "en",
        "recent_properties": [], "discussion_context": {}, "dialogue_state": {},
        "reference_map": {}, "focus_listing_id": None, "last_recommendation_snapshot": [],
        "response_mode": "tool_required", "dialogue_act": "general", "target_property_refs": [],
        "carry_forward_slots": {}, "slot_updates": {}, "response_plan": {}
    }

    # Turn 1: Out-of-Domain Guardrail test
    new_state, err = run_agent_turn(base_state.copy(), "Give me a recipe for chocolate cake.")
    if err:
        log_test("Agent", "Out_of_Domain", "Asked for cake recipe", False, f"CRASHED: {err}")
    else:
        response_msg = new_state["messages"][-1].content
        intent = new_state.get("active_intent")
        if new_state.get("is_out_of_domain") or "real estate" in response_msg.lower() or "can't help" in response_msg.lower():
            log_test("Agent", "Out_of_Domain", "Asked for cake recipe", True, f"Guardrail active. Intent: {intent}. Response: {response_msg[:50]}...")
        else:
            log_test("Agent", "Out_of_Domain", "Asked for cake recipe", False, f"Failed guardrail. Hallucinated answer. Response: {response_msg[:50]}...")

    # Turn 2: Abrupt Context Switch
    switch_state, err = run_agent_turn(base_state.copy(), "I want to buy a villa in Cairo. Actually no, I want to rent a small studio in Alexandria.")
    if err:
        log_test("Agent", "Context_Switch", "Buy villa Cairo -> Rent studio Alex", False, f"CRASHED: {err}")
    else:
        filters = switch_state.get("current_filters", {})
        if filters.get("category") == "rent" and "alexandria" in str(filters.get("city", "")).lower() and "studio" in str(filters.get("property_type", "")).lower():
            log_test("Agent", "Context_Switch", "Buy villa Cairo -> Rent studio Alex", True, f"Correctly extracted new context: {filters}")
        else:
            log_test("Agent", "Context_Switch", "Buy villa Cairo -> Rent studio Alex", False, f"Failed to override old context. Extracted: {filters}")

    # Turn 3: Gibberish Test
    gibberish_state, err = run_agent_turn(base_state.copy(), "asdfqwerzxcv 1234")
    if err:
        log_test("Agent", "Gibberish_Input", "asdfqwerzxcv 1234", False, f"CRASHED: {err}")
    else:
        intent = gibberish_state.get("active_intent")
        if intent in ["idle", "unknown"]:
            log_test("Agent", "Gibberish_Input", "asdfqwerzxcv 1234", True, f"Gracefully handled. Intent assigned: {intent}")
        else:
            log_test("Agent", "Gibberish_Input", "asdfqwerzxcv 1234", False, f"Hallucinated intent from gibberish. Intent: {intent}")


if __name__ == "__main__":
    test_recommender_imperfections()
    test_analyzer_imperfections()
    test_agent_imperfections()
    
    with open(os.path.join(BASE_DIR, "imperfections_raw.txt"), "w") as f:
        for mod, ts in results.items():
            f.write(f"\n--- {mod} ---\n")
            for t in ts:
                f.write(f"{t['name']} | Pass: {t['passed']} | {t['details']}\n")
    print("\nDeep tests completed. Results saved to imperfections_raw.txt")
