import os
import sys
import json
import importlib
import re
from typing import Dict, Any
from dotenv import load_dotenv
import litellm

# Load API Keys and inject into environment
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("!!! [WARNING] GEMINI_API_KEY not found. Gemini models will fail.")
else:
    os.environ["GEMINI_API_KEY"] = gemini_api_key

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Import internal configurations and states
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
AgentState = importlib.import_module("3_state_definition").AgentState
AGENT_CONFIG = importlib.import_module("1_agent_config").AGENT_CONFIG
from translation_utils import build_english_history, contains_arabic, translate_to_english, litellm_completion_with_groq_key_fallback
PRIMARY_MODEL = AGENT_CONFIG["model"]
FALLBACK_MODELS_LIST = AGENT_CONFIG.get("fallback_models", [])

# Template for the Intent Extraction Prompt: Used to instruct the LLM on how to parse user queries
INTENT_EXTRACTION_PROMPT = """
You are the AI Brain of an Elite Real Estate Concierge. 
Extract user requirements from the history into structured JSON.

EXTRACT:
- intent: 'search', 'analyze', 'book', 'discussion', or 'idle'.
- filters: { "category": "buy"|"rent", "town": string, "bedrooms": int, "property_type": string, "price_max": float, "price_min": float }
- booking_info: { "property_id": string, "user_name": string, "phone": string, "date": string }
- discussion_context: { "property_ids": [string], "comparison_properties": [string], "question_type": "comparison"|"details"|"opinion"|"general" }
- confidence: 0.0 to 1.0
- out_of_domain: true if user is talking about something unrelated to real estate.

RULE: "Examine", "Visit", "See in person", "Book", or "Appointment" = intent 'book'.
RULE: "Which is better", "Compare", "Difference", "vs", "Tell me more", "What about", "Pros and cons" = intent 'discussion'.
RULE: "Your opinion", "What do you think", "Should I", "Recommendation", "Better choice" = intent 'discussion'.
RULE: CONTEXT RESOLUTION - If the user says "the first one", "the second one", "that villa", or "property #123", look at the IDs (e.g., Listing ID 11225) mentioned in the previous AI message and put that ID into discussion_context.property_ids or booking_info.property_id.
RULE: "7 million" = 7,000,000. "700k" = 700,000. Be extremely careful with zeros.
RULE: If the user says "I want to see properties in Zayed", intent is 'search'.
RULE: If the user asks "How is the market in New Cairo?", intent is 'analyze'.
RULE: If the user mentions rent/monthly/per month/lease, set filters.category to "rent".
RULE: If the user mentions buy/purchase/own/for sale/sell, set filters.category to "buy".
RULE: When both appear, prioritize the latest explicit user request in the history.
"""

def infer_transaction_category(conversation_history_text: str) -> str:
    """
    Fallback category detector to harden rent vs buy extraction.
    Searches the conversation history for explicit transactional keywords and deduces intent based on proximity.
    """
    normalized_text = (conversation_history_text or "").lower()
    rent_keywords = [" rent ", " rental ", " monthly ", " per month ", " lease "]
    buy_keywords = [" buy ", " purchase ", " own ", " for sale ", " sell ", " sale "]

    last_rent_index = max((normalized_text.rfind(token) for token in rent_keywords), default=-1)
    last_buy_index = max((normalized_text.rfind(token) for token in buy_keywords), default=-1)
    
    if last_rent_index == -1 and last_buy_index == -1:
        return None
    return "rent" if last_rent_index > last_buy_index else "buy"

def extract_entities_via_regex(user_text: str) -> dict:
    """
    Regex-based safety net for critical real estate entities.
    Used to catch explicit inputs that the LLM might hallucinate or miss during JSON extraction.
    """
    extracted_data = {}
    normalized_text = user_text.lower()
    
    # 1. Location Detection (Common Areas): Maps slang to canonical database names
    common_locations_map = {
        "maadi": ["maadi", "madi"],
        "zayed": ["zayed", "sheikh zayed", "october"],
        "tagamoa": ["tagamoa", "new cairo", "fifth settlement"],
        "shorouk": ["shorouk", "shorok"],
        "heliopolis": ["heliopolis", "masr el gdida"]
    }
    
    for standard_location, trigger_tokens in common_locations_map.items():
        if any(token in normalized_text for token in trigger_tokens):
            extracted_data["town"] = standard_location.capitalize()
            break
            
    # 2. Price Detection (Looking for textual shorthand like '7 million', '7M', etc.)
    price_pattern_match = re.search(r"(\d+(?:\.\d+)?)\s*(million|m|mln)", normalized_text)
    if price_pattern_match:
        try:
            numeric_value = float(price_pattern_match.group(1))
            extracted_data["price_max"] = numeric_value * 1_000_000
        except ValueError:
            pass
    elif "million" not in normalized_text:
        # Check for raw numbers > 100k assuming they are budgets
        large_numbers = re.findall(r"\b(\d{6,10})\b", normalized_text)
        if large_numbers:
            extracted_data["price_max"] = float(large_numbers[0])

    # 3. Property Type categorization
    if "apart" in normalized_text or "flat" in normalized_text:
        extracted_data["property_type"] = "Apartment"
    elif "villa" in normalized_text or "townhouse" in normalized_text:
        extracted_data["property_type"] = "Villa"
        
    return extracted_data

def _detect_dialogue_act(latest_user_message: str) -> str:
    normalized = f" {str(latest_user_message or '').lower()} "
    if any(token in normalized for token in [" compare ", " vs ", " versus ", " difference ", " better "]):
        return "compare"
    if any(token in normalized for token in [" details ", " tell me more ", " more about ", " explain ", " pros ", " cons "]):
        return "request_details"
    if any(token in normalized for token in [" cheaper ", " same ", " instead ", " change ", " but "]):
        return "refine_search"
    if any(token in normalized for token in [" yes ", " okay ", " go ahead ", " confirm "]):
        return "confirm"
    if any(token in normalized for token in [" hi ", " hello ", " hey "]):
        return "chitchat"
    return "general"

def _extract_property_refs(latest_user_message: str) -> list:
    normalized = str(latest_user_message or "").lower()
    refs = []
    ordinal_map = {
        "first": 1, "1st": 1, "one": 1,
        "second": 2, "2nd": 2, "two": 2,
        "third": 3, "3rd": 3, "three": 3,
        "fourth": 4, "4th": 4, "four": 4,
        "fifth": 5, "5th": 5, "five": 5
    }
    for token, ordinal in ordinal_map.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            refs.append({"kind": "ordinal", "value": ordinal})
    for match in re.findall(r"(?:#|id\s*)(\d{3,10})", normalized):
        refs.append({"kind": "listing_id", "value": str(match)})
    if any(pronoun in normalized for pronoun in ["this one", "that one", "it", "that property"]):
        refs.append({"kind": "pronoun", "value": "focus"})
    return refs

def intent_node(current_state: AgentState) -> Dict[str, Any]:
    """
    NODE 1: Intent & Entity Extraction (The Brain)
    This node serves as the semantic router for the agent. It parses conversational history
    into deterministic state variables (JSON) using LLMs and robust regex fallbacks.
    """
    # 1. Prepare conversation history for the LLM context window
    conversation_messages = current_state.get("messages", [])

    # Important: During auditor retry loops, trailing AI draft messages can appear at the end.
    # Always anchor extraction on the latest human message only.
    latest_user_index = None
    for idx in range(len(conversation_messages) - 1, -1, -1):
        if getattr(conversation_messages[idx], "type", "") == "human":
            latest_user_index = idx
            break

    latest_user_message = (
        conversation_messages[latest_user_index].content
        if latest_user_index is not None
        else ""
    )
    messages_for_intent = (
        conversation_messages[: latest_user_index + 1]
        if latest_user_index is not None
        else conversation_messages
    )

    history_contains_arabic = any(contains_arabic(getattr(msg, "content", "")) for msg in messages_for_intent)
    formatted_history_string = "\n".join([f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in messages_for_intent])
    history_for_llm = build_english_history(messages_for_intent) if history_contains_arabic else formatted_history_string
    latest_user_message_for_regex = translate_to_english(latest_user_message) if contains_arabic(latest_user_message) else latest_user_message
    user_language = "ar-EG" if contains_arabic(latest_user_message) else "en"
    
    max_retries_allowed = 2
    last_encountered_error = None
    
    # 2. Sticky Fallback logic: Ensure we continue using the fallback model if the primary is down
    target_llm_model = current_state.get("active_model", PRIMARY_MODEL)
    available_fallbacks = FALLBACK_MODELS_LIST if target_llm_model == PRIMARY_MODEL else []
    
    extracted_json_data = {}
    actual_model_used = target_llm_model
    
    # 3. Execute LLM Extraction Call with Retry Loop
    for retry_attempt in range(max_retries_allowed):
        print(f"--- [BRAIN] Calling Model: {target_llm_model} ---")
        try:
            llm_response = litellm_completion_with_groq_key_fallback(
                model=target_llm_model,
                messages=[
                    {"role": "system", "content": INTENT_EXTRACTION_PROMPT},
                    {"role": "user", "content": f"History:\n{history_for_llm}\n\nReturn JSON:"}
                ],
                response_format={"type": "json_object"},
                fallbacks=available_fallbacks
            )
            actual_model_used = llm_response.model
            
            # Re-attach provider prefix if LiteLLM stripped it during fallback
            if "/" not in actual_model_used:
                for fallback_candidate in [PRIMARY_MODEL] + FALLBACK_MODELS_LIST:
                    if actual_model_used in fallback_candidate:
                        actual_model_used = fallback_candidate
                        break
                        
            print(f"--- [BRAIN] Target: {target_llm_model} | Actual: {actual_model_used} ---")
            raw_llm_content = llm_response.choices[0].message.content
            # Clean markdown code blocks from the JSON
            cleaned_json_string = raw_llm_content.replace("```json", "").replace("```", "").strip()
            extracted_json_data = json.loads(cleaned_json_string)
            last_encountered_error = None
            break  # Success
            
        except Exception as api_error:
            last_encountered_error = api_error
            # Provide explicit feedback to the LLM on the next attempt if JSON parsing failed
            if retry_attempt < max_retries_allowed - 1:
                history_for_llm += "\n(Note: Your previous response was not valid JSON. Please fix it.)"

    # 4. Handle Unrecoverable API Errors (e.g. Quota exhaustion)
    if last_encountered_error is not None:
        error_string_lower = str(last_encountered_error).lower()
        if "429" in error_string_lower or "quota" in error_string_lower or "rate" in error_string_lower:
            print(f"!!! [BRAIN_ERROR] API quota exhausted: {last_encountered_error}")
            return {
                "active_intent": "idle",
                "current_filters": current_state.get("current_filters", {}),
                "booking_details": current_state.get("booking_details", {}),
                "missing_info": ["__quota_exceeded__"],
                "confidence_score": 0.0,
                "is_out_of_domain": False,
                "audit_retries": 0,
            }
        else:
            print(f"!!! [BRAIN_ERROR] Extraction failed after {max_retries_allowed} attempts: {last_encountered_error}")
            extracted_json_data = {"intent": "idle", "filters": {}, "confidence": 0.0, "out_of_domain": False}
    
    # 5. Merge state updates intelligently (Preserve existing filters while overriding new ones)
    merged_filters = current_state.get("current_filters", {}).copy()
    llm_extracted_filters = extracted_json_data.get("filters") or {}
    
    # SAFETY NET: Run hardened regex extraction on the most recent user message
    regex_extracted_data = extract_entities_via_regex(latest_user_message_for_regex)
    
    # Priority 1: Apply LLM extracted results
    for filter_key, filter_value in llm_extracted_filters.items():
        if filter_value:
            merged_filters[filter_key] = filter_value
            
    # Priority 2: Apply Safety Net results (only if LLM completely missed them)
    for filter_key, filter_value in regex_extracted_data.items():
        if not merged_filters.get(filter_key):
            merged_filters[filter_key] = filter_value

    # EXPERT LOGIC: Enforce budget typecasting safely
    if merged_filters.get("price_max"):
        try:
            parsed_maximum_price = float(merged_filters["price_max"])
        except ValueError:
            pass

    # Safety net: Infer category from explicit wording if the LLM completely missed it
    inferred_category_from_text = infer_transaction_category(f" {history_for_llm.lower()} ")
    if inferred_category_from_text and not merged_filters.get("category"):
        merged_filters["category"] = inferred_category_from_text

    # Default property_type to both Apartment and Villa if not specified, to make search more flexible
    if "property_type" not in merged_filters:
        merged_filters["property_type"] = "Apartment,Villa"
            
    # Expert Logic: Handle Bookings by merging partial details across turns
    merged_booking_details = current_state.get("booking_details", {}).copy()
    llm_extracted_booking = extracted_json_data.get("booking_info") or {}
    for booking_key, booking_value in llm_extracted_booking.items():
        if booking_value:
            merged_booking_details[booking_key] = booking_value
    
    # 6. Guardrails: Calculate missing prerequisites dynamically based on workflow intent
    missing_required_fields = []
    detected_intent = extracted_json_data.get("intent", "idle")
    discussion_context = extracted_json_data.get("discussion_context", {})
    dialogue_act = _detect_dialogue_act(latest_user_message_for_regex)
    target_property_refs = _extract_property_refs(latest_user_message_for_regex)
    previous_reference_map = current_state.get("reference_map", {}) or {}
    previous_focus_listing_id = current_state.get("focus_listing_id")
    slot_updates = {
        k: v for k, v in llm_extracted_filters.items() if v is not None and v != ""
    }
    carry_forward_slots = merged_filters.copy()

    # Build fresh ordinal map from recent properties whenever available.
    recent_properties = current_state.get("recent_properties", []) or []
    reference_map = previous_reference_map.copy()
    if recent_properties:
        reference_map = {}
        for idx, prop in enumerate(recent_properties[:5], start=1):
            listing_id = prop.get("listing_id")
            if listing_id is not None:
                reference_map[str(idx)] = str(listing_id)

    resolved_reference_ids = []
    for ref in target_property_refs:
        if ref.get("kind") == "listing_id":
            resolved_reference_ids.append(str(ref.get("value")))
        elif ref.get("kind") == "ordinal":
            resolved_from_ordinal = reference_map.get(str(ref.get("value")))
            if resolved_from_ordinal:
                resolved_reference_ids.append(str(resolved_from_ordinal))
        elif ref.get("kind") == "pronoun" and previous_focus_listing_id:
            resolved_reference_ids.append(str(previous_focus_listing_id))
    if resolved_reference_ids:
        discussion_context["property_ids"] = resolved_reference_ids
    
    if detected_intent == "search":
        # Search queries must have at least these three fields to execute cleanly
        critical_search_fields = ["town", "property_type", "category"]
        missing_required_fields = [field for field in critical_search_fields if field not in merged_filters]
    elif detected_intent == "book":
        # Consolidate date fields for the downstream Booking Engine
        if "preferred_date" in merged_booking_details and "date" not in merged_booking_details:
            merged_booking_details["date"] = merged_booking_details["preferred_date"]
        # Booking queries require identity and target
        critical_booking_fields = ["property_id", "user_name", "phone", "date"]
        missing_required_fields = [field for field in critical_booking_fields if field not in merged_booking_details]
    elif detected_intent == "discussion":
        # Discussion requires we have context about which properties they're discussing
        # If no IDs mentioned, ask the user to specify
        if not discussion_context.get("property_ids"):
            missing_required_fields = ["property_reference"]  # Ask user which property they mean

    # Deterministic policy overrides for user-friendly multi-turn behavior.
    if dialogue_act in ["request_details", "compare"] and reference_map:
        detected_intent = "discussion"
        if not discussion_context.get("property_ids") and previous_focus_listing_id:
            discussion_context["property_ids"] = [str(previous_focus_listing_id)]
            missing_required_fields = []
    if dialogue_act == "refine_search":
        detected_intent = "search"
        missing_required_fields = [field for field in ["town", "property_type", "category"] if field not in merged_filters]

    response_mode = "tool_required"
    if missing_required_fields:
        response_mode = "clarify_needed"
    elif detected_intent in ["discussion", "idle"] and dialogue_act in ["confirm", "chitchat"] and not discussion_context.get("property_ids"):
        response_mode = "context_only"

    focus_listing_id = previous_focus_listing_id
    if discussion_context.get("property_ids"):
        focus_listing_id = str(discussion_context["property_ids"][0])

    dialogue_state = {
        "last_dialogue_act": dialogue_act,
        "pending_question": missing_required_fields[0] if missing_required_fields else None,
        "turn_goal": detected_intent
    }
    
    # Final Payload returned to the Graph State
    return {
        "active_intent": detected_intent,
        "current_filters": merged_filters,
        "booking_details": merged_booking_details,
        "missing_info": missing_required_fields,
        "confidence_score": extracted_json_data.get("confidence", 0.5),
        "is_out_of_domain": extracted_json_data.get("out_of_domain", False),
        "active_model": actual_model_used if last_encountered_error is None else target_llm_model,
        "user_language": user_language,
        "discussion_context": discussion_context,
        "dialogue_state": dialogue_state,
        "reference_map": reference_map,
        "focus_listing_id": focus_listing_id,
        "response_mode": response_mode,
        "dialogue_act": dialogue_act,
        "target_property_refs": target_property_refs,
        "carry_forward_slots": carry_forward_slots,
        "slot_updates": slot_updates
    }
