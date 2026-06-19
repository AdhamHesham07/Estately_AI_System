import os
import sys
import importlib
import re
import json
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure absolute imports work from nodes up to the main agent root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

AgentState = importlib.import_module("3_state_definition").AgentState
DataBridge = importlib.import_module("2_data_bridge").DataBridge
BookingEngine = importlib.import_module("3_sceduler.booking_logic").BookingEngine

DEMO_PROPERTY_IMAGES = [
    "assets/property-1.png",
    "assets/property-2.png",
    "assets/property-3.png",
]

LOCATION_ALIASES = {
    "fifth settlement": {"town": "New Cairo City", "district": "The 5th Settlement"},
    "5th settlement": {"town": "New Cairo City", "district": "The 5th Settlement"},
    "tagamoa": {"town": "New Cairo City", "district": "The 5th Settlement"},
    "new cairo": {"town": "New Cairo City"},
    "sheikh zayed": {"town": "Sheikh Zayed City"},
    "zayed": {"town": "Sheikh Zayed City"},
    "maadi": {"town": "Maadi"},
    "heliopolis": {"town": "Heliopolis"},
}

def _attach_demo_image(prop: dict) -> dict:
    if not prop:
        return prop
    if not prop.get("image_url"):
        listing_key = str(prop.get("listing_id") or "")
        image_index = sum(ord(ch) for ch in listing_key) % len(DEMO_PROPERTY_IMAGES)
        prop["image_url"] = DEMO_PROPERTY_IMAGES[image_index]
    return prop

def _attach_demo_images(properties: list) -> list:
    return [_attach_demo_image(prop) for prop in (properties or [])]

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
        "image_url": prop.get("image_url"),
    }

def _select_search_page(current_state: AgentState, tool_execution_results: dict) -> tuple:
    existing_pool = tool_execution_results.get("recommendation_pool") or []
    if current_state.get("dialogue_act") == "show_more" and existing_pool:
        offset = int(tool_execution_results.get("recommendation_offset", 3) or 3)
        selected = existing_pool[offset:offset + 3]
        if not selected:
            offset = 0
            selected = existing_pool[:3]
        is_vague = (tool_execution_results.get("recommendations") or {}).get("is_vague", False)
        return selected, existing_pool, offset + len(selected), True, is_vague

    recommendation_results = DataBridge.execute_recommendation(current_state.get("current_filters", {}))
    pool = recommendation_results.get("candidates", [])
    selected = pool[:3]
    return selected, pool, len(selected), False, recommendation_results.get("is_vague", False)

def _handle_search(current_state: AgentState, tool_execution_results: dict):
    search_filters = current_state.get("current_filters", {})
    final_candidates, recommendation_pool, next_offset, is_more_request, is_vague = _select_search_page(
        current_state, tool_execution_results
    )

    for prop in final_candidates:
        prop["analyzer_reasoning"] = (
            "Additional option from your saved recommendation pool."
            if is_more_request
            else "Selected by the recommender based on your budget, location, and property preferences."
        )

    _attach_demo_images(final_candidates)
    _attach_demo_images(recommendation_pool)

    recommendation_results = {
        "candidates": final_candidates,
        "pool_size": len(recommendation_pool),
        "next_offset": next_offset,
        "has_more": next_offset < len(recommendation_pool),
        "is_vague": is_vague,
    }
    
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
    tool_execution_results["recommendation_pool"] = recommendation_pool
    tool_execution_results["recommendation_offset"] = next_offset
    tool_execution_results["valuation_briefs"] = property_analysis_briefs

def _latest_user_query(current_state: AgentState) -> str:
    messages = current_state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            return getattr(msg, "content", "") or ""
    return ""

def _area_filters_from_phrase(area_phrase: str) -> dict:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(area_phrase or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for alias, filters in LOCATION_ALIASES.items():
        if alias in normalized:
            return filters.copy()
    return {"town": area_phrase.strip()} if area_phrase.strip() else {}

def _extract_comparison_area_filters(user_query: str) -> list:
    normalized = str(user_query or "").lower()
    if not any(token in normalized for token in (" vs ", " versus ", " compare ")):
        return []

    cleaned = re.sub(r"\bcompare\b", " ", normalized)
    cleaned = re.sub(r"\b(apartment|apartments|villa|villas|property|properties|prices|price|market|in|between|for)\b", " ", cleaned)
    parts = re.split(r"\s+(?:vs|versus)\s+", cleaned, maxsplit=1)
    if len(parts) != 2:
        return []

    left_filters = _area_filters_from_phrase(parts[0])
    right_filters = _area_filters_from_phrase(parts[1])
    return [filters for filters in (left_filters, right_filters) if filters]

def _build_comparison_samples(current_state: AgentState, search_filters: dict, user_query: str) -> list:
    area_filters = _extract_comparison_area_filters(user_query)
    if len(area_filters) < 2:
        return []

    comparison_samples = []
    base_filters = search_filters.copy()
    base_filters.setdefault("category", "buy")
    if not any(base_filters.get(key) for key in ("price_egp", "price_min", "price_max")):
        if base_filters.get("category") == "rent":
            base_filters["price_min"] = 0
            base_filters["price_max"] = 5_000_000
        else:
            base_filters["price_min"] = 0
            base_filters["price_max"] = 200_000_000
    for area_filter in area_filters[:2]:
        sample_filters = base_filters.copy()
        for location_key in ("city", "town", "district", "subdistrict"):
            sample_filters.pop(location_key, None)
        sample_filters.update(area_filter)

        results = DataBridge.execute_recommendation(sample_filters)
        candidates = _attach_demo_images((results.get("candidates") or [])[:3])
        area_label = area_filter.get("district") or area_filter.get("town") or area_filter.get("city")
        for prop in candidates:
            prop["analyzer_reasoning"] = (
                f"Representative {sample_filters.get('property_type', 'property')} sample from {area_label} "
                "for this area comparison."
            )
        comparison_samples.append({
            "area": area_label,
            "filters": sample_filters,
            "candidates": candidates,
        })
    return comparison_samples

def _handle_analyze(current_state: AgentState, tool_execution_results: dict):
    search_filters = current_state.get("current_filters", {})
    
    # Extract user_query from the last human message
    user_query = _latest_user_query(current_state)
    
    # The new Strategy Layer: Invoke the complete Analyzer Intelligence Hub
    synthesis_report = DataBridge.execute_analyzer_synthesis(user_query, search_filters)
    comparison_samples = _build_comparison_samples(current_state, search_filters, user_query)
    if comparison_samples:
        synthesis_report = (
            f"{synthesis_report}\n\n"
            "LIVE LISTING SAMPLES FOR THIS COMPARISON:\n"
            f"{json.dumps(comparison_samples, indent=2)}"
        )
    tool_execution_results["analyzer_synthesis"] = synthesis_report
    tool_execution_results["comparison_samples"] = comparison_samples

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
