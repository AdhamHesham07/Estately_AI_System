import os
import sys
import importlib
from typing import Dict, Any

# Ensure absolute imports work from nodes up to the main agent root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

AgentState = importlib.import_module("3_state_definition").AgentState
DataBridge = importlib.import_module("2_data_bridge").DataBridge
BookingEngine = importlib.import_module("3_sceduler.booking_logic").BookingEngine

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
        property_ids = discussion_context.get("property_ids", [])
        
        # Fetch detailed analysis for the properties being discussed
        discussion_briefs = []
        if property_ids:
            for property_id in property_ids:
                # Build a search context around the property to get analysis
                prop_analysis = DataBridge.execute_fair_price_check({
                    "listing_id": property_id
                }, search_filters.get('category', 'buy'))
                discussion_briefs.append({
                    "listing_id": property_id,
                    "analysis": prop_analysis
                })
        
        # Store discussion data for the Tongue to format into a conversational response
        tool_execution_results["discussion_briefs"] = discussion_briefs
        tool_execution_results["discussion_context"] = discussion_context
    
    # Store recent properties from recommendations for discussion context
    if detected_intent == "search":
        recent_props = tool_execution_results.get("recommendations", {}).get("candidates", [])
        if recent_props:
            tool_execution_results["recent_properties"] = recent_props[:5]  # Keep top 5
        
    # Return the newly mutated tool states to the graph
    return {
        "tool_outputs": tool_execution_results,
        "recent_properties": tool_execution_results.get("recent_properties", current_state.get("recent_properties", []))
    }
