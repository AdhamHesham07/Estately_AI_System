import os
import sys
import importlib
from typing import Dict, Any

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

def tool_node(current_state: AgentState) -> Dict[str, Any]:
    """
    NODE 2: Tool Execution (The Hands)
    This node intercepts the intent, invokes the correct backend modules, and packages 
    the analytical outputs back into the graph state for the LLM to interpret.
    """
    # 1. Retrieve the current intent and user constraints
    detected_intent = current_state.get("active_intent")
    search_filters = current_state.get("current_filters", {})
    
    print(f"--- [HANDS] Executing Tool for Intent: {detected_intent} ---")
    
    # 2. Maintain a copy of previous tool execution data
    tool_execution_results = current_state.get("tool_outputs", {}).copy()
    property_ids = []
    
    # 3. Route Execution Path based on Intent
    if detected_intent == "search":
        # [SEARCH] Call the Recommender Bridge to retrieve ranked candidates
        recommendation_results = DataBridge.execute_recommendation(search_filters)
        candidate_properties = recommendation_results.get("candidates", [])
        
        # Expert Logic: Proactively run the 'Fair Price' valuation model on the top 3 candidates.
        # This provides the 'Tongue' (Translator Node) with deep, factual insights it can quote.
        property_analysis_briefs = []
        for property_data in candidate_properties[:3]:
            fair_price_brief = DataBridge.execute_fair_price_check(property_data, search_filters.get('category', 'buy'))
            property_analysis_briefs.append({
                "listing_id": property_data.get('listing_id'),
                "analysis": fair_price_brief
            })
            
        # Store both raw listings and their valuation analysis into the state context
        tool_execution_results["recommendations"] = recommendation_results
        tool_execution_results["valuation_briefs"] = property_analysis_briefs
        
    elif detected_intent == "analyze":
        # [ANALYZE] Call the Analyzer Bridge to pull macroeconomic data (Market Pulse)
        market_pulse_context = DataBridge.execute_analysis("market_pulse", search_filters)
        tool_execution_results["market_context"] = market_pulse_context
        
    elif detected_intent == "book":
        # [BOOK] Register the appointment securely in the SQL Database
        booking_registration_result = BookingEngine.register_booking(current_state.get("booking_details", {}))
        tool_execution_results["booking_response"] = booking_registration_result
    
    elif detected_intent == "discussion":
        # [DISCUSSION] Provide detailed insights, comparisons, and expert opinions on specific properties
        discussion_context = current_state.get("discussion_context", {})
        property_ids = _resolve_discussion_listing_ids(current_state)
        discussion_context["property_ids"] = property_ids
        discussion_context["question_type"] = current_state.get("dialogue_act", discussion_context.get("question_type", "general"))
        
        # Fetch detailed analysis for the properties being discussed
        discussion_briefs = []
        recent_properties = current_state.get("recent_properties", []) or []
        recommendation_candidates = tool_execution_results.get("recommendations", {}).get("candidates", []) or []
        indexed_properties = {}
        for prop in (recent_properties + recommendation_candidates):
            listing_id = prop.get("listing_id")
            if listing_id is not None:
                indexed_properties[str(listing_id)] = prop

        if property_ids:
            for property_id in property_ids:
                matched_prop = indexed_properties.get(str(property_id), {"listing_id": property_id})
                prop_analysis = DataBridge.execute_fair_price_check(matched_prop, search_filters.get('category', 'buy'))
                discussion_briefs.append({
                    "listing_id": property_id,
                    "analysis": prop_analysis,
                    "property_facts": _build_property_fact(matched_prop)
                })
        
        # Store discussion data for the Tongue to format into a conversational response
        tool_execution_results["discussion_briefs"] = discussion_briefs
        tool_execution_results["discussion_context"] = discussion_context
        if discussion_briefs:
            tool_execution_results["focused_property"] = discussion_briefs[0].get("property_facts", {})
            tool_execution_results["comparison_set"] = [item.get("property_facts", {}) for item in discussion_briefs[:3]]
    
    # Store recent properties from recommendations for discussion context
    if detected_intent == "search":
        recent_props = tool_execution_results.get("recommendations", {}).get("candidates", [])
        if recent_props:
            tool_execution_results["recent_properties"] = recent_props[:5]  # Keep top 5
        
    # Return the newly mutated tool states to the graph
    return {
        "tool_outputs": tool_execution_results,
        "recent_properties": tool_execution_results.get("recent_properties", current_state.get("recent_properties", [])),
        "focus_listing_id": str(property_ids[0]) if detected_intent == "discussion" and property_ids else current_state.get("focus_listing_id"),
        "last_recommendation_snapshot": [_build_property_fact(prop) for prop in (tool_execution_results.get("recent_properties", []) or [])[:5]]
    }
