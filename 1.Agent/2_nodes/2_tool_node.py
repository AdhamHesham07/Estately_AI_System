import os
import sys
import importlib
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure absolute imports work from nodes up to the main agent root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

AgentState = importlib.import_module("3_state_definition").AgentState
DataBridge = importlib.import_module("2_data_bridge").DataBridge
BookingEngine = importlib.import_module("3_sceduler.booking_logic").BookingEngine

def _resolve_discussion_listing_ids(current_state: AgentState) -> list:
    discussion_context = current_state.get("discussion_context", {}) or {}
    property_ids = [str(pid) for pid in (discussion_context.get("property_ids", []) or []) if pid is not None]
    if property_ids:
        return property_ids

    target_refs = current_state.get("target_property_refs", []) or []
    reference_map = current_state.get("reference_map", {}) or {}
    focus_listing_id = current_state.get("focus_listing_id")
    resolved = []
    for ref in target_refs:
        kind = ref.get("kind")
        if kind == "listing_id":
            resolved.append(str(ref.get("value")))
        elif kind == "ordinal":
            listing = reference_map.get(str(ref.get("value")))
            if listing:
                resolved.append(str(listing))
        elif kind == "pronoun" and focus_listing_id:
            resolved.append(str(focus_listing_id))
    if resolved:
        return resolved
    if focus_listing_id:
        return [str(focus_listing_id)]
    return []

def _build_property_fact(prop: dict) -> dict:
    if not prop:
        return {}
    return {
        "listing_id": prop.get("listing_id"),
        "price_egp": prop.get("price_egp"),
        "bedrooms": prop.get("bedrooms"),
        "bathrooms": prop.get("bathrooms"),
        "area_value": prop.get("area_value") or prop.get("area"),
        "town": prop.get("town"),
        "district": prop.get("district"),
        "property_type": prop.get("property_type"),
        "payment_method": prop.get("payment_method"),
        "completion_status": prop.get("completion_status"),
    }

def _handle_search(current_state: AgentState, tool_execution_results: dict):
    search_filters = current_state.get("current_filters", {})
    recommendation_results = DataBridge.execute_recommendation(search_filters)
    candidate_properties = recommendation_results.get("candidates", [])
    
    # Extract user_query from the last human message
    user_query = ""
    messages = current_state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            user_query = getattr(msg, "content", "")
            break
            
    # Use Analyzer to intelligently pick the best 3 properties and provide reasoning
    try:
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "3.Analyzer")))
        analyzer_llm = importlib.import_module("8_analyzer_llm")
        selected_data = analyzer_llm.AnalyzerSynthesizer.analyze_and_select_top_properties(user_query, candidate_properties)
        
        # Merge reasoning back into candidates
        selected_ids = {str(item.get("listing_id")): item.get("reasoning", "") for item in selected_data}
        final_candidates = []
        for prop in candidate_properties:
            pid_str = str(prop.get("listing_id"))
            if pid_str in selected_ids:
                prop["analyzer_reasoning"] = selected_ids[pid_str]
                final_candidates.append(prop)
                
        # Fallback if no match
        if not final_candidates:
            final_candidates = candidate_properties[:3]
            
    except Exception as e:
        print(f"[ERROR] Analyzer intelligent selection failed: {e}")
        final_candidates = candidate_properties[:3]
        
    recommendation_results["candidates"] = final_candidates
    
    # ✅ PHASE 3A: Parallelize per-property fair-price checks
    # Each property's fair-price analysis is independent — run all 3 concurrently
    category = search_filters.get('category', 'buy')
    property_analysis_briefs = [None] * len(final_candidates)

    def _fetch_brief(idx_prop):
        idx, property_data = idx_prop
        brief = DataBridge.execute_fair_price_check(property_data, category)
        return idx, {
            "listing_id": property_data.get('listing_id'),
            "analysis": brief,
            "analyzer_reasoning": property_data.get("analyzer_reasoning", "Selected based on criteria.")
        }

    with ThreadPoolExecutor(max_workers=min(3, len(final_candidates))) as pool:
        futures = {pool.submit(_fetch_brief, (i, prop)): i for i, prop in enumerate(final_candidates)}
        for future in as_completed(futures):
            try:
                idx, brief_result = future.result()
                property_analysis_briefs[idx] = brief_result
            except Exception as exc:
                print(f"[WARNING] Fair-price check failed for a property: {exc}")

    # Remove any None slots (failed calls)
    property_analysis_briefs = [b for b in property_analysis_briefs if b is not None]
        
    tool_execution_results["recommendations"] = recommendation_results
    tool_execution_results["valuation_briefs"] = property_analysis_briefs

def _handle_analyze(current_state: AgentState, tool_execution_results: dict):
    search_filters = current_state.get("current_filters", {})
    
    # Extract user_query from the last human message
    user_query = ""
    messages = current_state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            user_query = getattr(msg, "content", "")
            break
    
    # The new Strategy Layer: Invoke the complete Analyzer Intelligence Hub
    synthesis_report = DataBridge.execute_analyzer_synthesis(user_query, search_filters)
    tool_execution_results["analyzer_synthesis"] = synthesis_report

def _handle_book(current_state: AgentState, tool_execution_results: dict):
    booking_registration_result = BookingEngine.register_booking(current_state.get("booking_details", {}))
    tool_execution_results["booking_response"] = booking_registration_result

def _handle_discussion(current_state: AgentState, tool_execution_results: dict) -> list:
    discussion_context = current_state.get("discussion_context", {})
    property_ids = _resolve_discussion_listing_ids(current_state)
    discussion_context["property_ids"] = property_ids
    discussion_context["question_type"] = current_state.get("dialogue_act", discussion_context.get("question_type", "general"))
    
    discussion_briefs = []
    recent_properties = current_state.get("recent_properties", []) or []
    recommendation_candidates = tool_execution_results.get("recommendations", {}).get("candidates", []) or []
    indexed_properties = {}
    for prop in (recent_properties + recommendation_candidates):
        listing_id = prop.get("listing_id")
        if listing_id is not None:
            indexed_properties[str(listing_id)] = prop

    search_filters = current_state.get("current_filters", {})
    if property_ids:
        for property_id in property_ids:
            matched_prop = indexed_properties.get(str(property_id), {"listing_id": property_id})
            prop_analysis = DataBridge.execute_fair_price_check(matched_prop, search_filters.get('category', 'buy'))
            discussion_briefs.append({
                "listing_id": property_id,
                "analysis": prop_analysis,
                "property_facts": _build_property_fact(matched_prop)
            })
    
    tool_execution_results["discussion_briefs"] = discussion_briefs
    tool_execution_results["discussion_context"] = discussion_context
    if discussion_briefs:
        tool_execution_results["focused_property"] = discussion_briefs[0].get("property_facts", {})
        tool_execution_results["comparison_set"] = [item.get("property_facts", {}) for item in discussion_briefs[:3]]
    return property_ids

def tool_node(current_state: AgentState) -> Dict[str, Any]:
    """
    NODE 2: Tool Execution (The Hands)
    This node intercepts the intent, invokes the correct backend modules, and packages 
    the analytical outputs back into the graph state for the LLM to interpret.
    """
    detected_intent = current_state.get("active_intent")
    print(f"--- [HANDS] Executing Tool for Intent: {detected_intent} ---")
    
    tool_execution_results = current_state.get("tool_outputs", {}).copy()
    property_ids = []
    
    # Route to specialized handlers
    if detected_intent == "search":
        _handle_search(current_state, tool_execution_results)
    elif detected_intent == "analyze":
        _handle_analyze(current_state, tool_execution_results)
    elif detected_intent == "book":
        _handle_book(current_state, tool_execution_results)
    elif detected_intent == "discussion":
        property_ids = _handle_discussion(current_state, tool_execution_results)
    
    # Store recent properties from recommendations for discussion context
    if detected_intent == "search":
        recent_props = tool_execution_results.get("recommendations", {}).get("candidates", [])
        if recent_props:
            tool_execution_results["recent_properties"] = recent_props[:5]  # Keep top 5
        
    return {
        "tool_outputs": tool_execution_results,
        "recent_properties": tool_execution_results.get("recent_properties", current_state.get("recent_properties", [])),
        "focus_listing_id": str(property_ids[0]) if detected_intent == "discussion" and property_ids else current_state.get("focus_listing_id"),
        "last_recommendation_snapshot": [_build_property_fact(prop) for prop in (tool_execution_results.get("recent_properties", []) or [])[:5]]
    }
