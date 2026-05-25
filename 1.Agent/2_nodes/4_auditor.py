import os
import sys
import json
import importlib
import re
from typing import Dict, Any

# Set paths to allow cross-module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# Load and configure API Keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

AgentState = importlib.import_module("3_state_definition").AgentState

AGENT_CONFIG = importlib.import_module("1_agent_config").AGENT_CONFIG
PRIMARY_MODEL = AGENT_CONFIG["model"]
FALLBACK_MODELS_LIST = AGENT_CONFIG.get("fallback_models", [])
from translation_utils import translate_for_audit, litellm_completion_with_groq_key_fallback

# The system instruction set for the Auditor to enforce anti-hallucination guardrails
from prompts import AUDITOR_SYSTEM_PROMPT

def _latest_human_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            return getattr(msg, "content", "") or ""
    return ""

def _latest_ai_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "ai":
            return getattr(msg, "content", "") or ""
    return ""

def _extract_listing_ids(text: str) -> list:
    if not text:
        return []
    return re.findall(r"(?:#|listing\s*#?|id\s*)(\d{3,10})", text.lower())

def _is_followup_request(user_text: str) -> bool:
    normalized = f" {str(user_text or '').lower()} "
    follow_up_markers = [
        " first ", " second ", " third ", " that one ", " this one ",
        " tell me more ", " details ", " compare ", " difference ", " what about "
    ]
    return any(marker in normalized for marker in follow_up_markers)

def _deterministic_audit(current_state: AgentState) -> Dict[str, Any]:
    detected_intent = current_state.get("active_intent", "idle")
    tool_execution_results = current_state.get("tool_outputs", {}) or {}
    candidate_properties = tool_execution_results.get("recommendations", {}).get("candidates", []) or []
    discussion_briefs = tool_execution_results.get("discussion_briefs", []) or []
    latest_user_message = _latest_human_text(current_state.get("messages", []))
    latest_assistant_message = _latest_ai_text(current_state.get("messages", []))
    assistant_for_audit = translate_for_audit(latest_assistant_message)
    lower_assistant = assistant_for_audit.lower()
    asked_follow_up = _is_followup_request(latest_user_message)

    allowed_listing_ids = set()
    for prop in candidate_properties:
        if prop.get("listing_id") is not None:
            allowed_listing_ids.add(str(prop.get("listing_id")))
    for brief in discussion_briefs:
        if brief.get("listing_id") is not None:
            allowed_listing_ids.add(str(brief.get("listing_id")))

    mentioned_ids = [str(item) for item in _extract_listing_ids(assistant_for_audit)]
    if allowed_listing_ids and mentioned_ids:
        unknown_ids = [item for item in mentioned_ids if item not in allowed_listing_ids]
        if unknown_ids:
            return {
                "pass": False,
                "reason": f"Unknown listing IDs mentioned: {unknown_ids}",
                "correction": "Use only listing IDs that exist in tool outputs."
            }

    # If user asked a focused follow-up, assistant must anchor to concrete property facts.
    if detected_intent == "discussion" and asked_follow_up:
        has_anchor = ("#" in lower_assistant) or ("listing" in lower_assistant) or ("egp" in lower_assistant)
        if not has_anchor:
            return {
                "pass": False,
                "reason": "Follow-up answer is too generic",
                "correction": "Start with direct property-specific facts (listing id/price/location)."
            }

    # For search with candidates, avoid empty/generic non-answer.
    if detected_intent == "search" and candidate_properties:
        generic_markers = [
            "how can i help", "i'm your ai", "could you please specify",
            "i need more details", "please provide"
        ]
        if any(marker in lower_assistant for marker in generic_markers):
            return {
                "pass": False,
                "reason": "Search response ignored available results",
                "correction": "Present at least 1-3 concrete properties from the tool results."
            }

    return {"pass": True}

def auditor_node(current_state: AgentState) -> Dict[str, Any]:
    """
    NODE 4: The 'Quality Guardrail' Layer (The Auditor)
    This node intercepts the final drafted response from the Tongue and validates it against 
    the raw database facts. If it detects hallucination or omitted constraints, it rejects 
    the output and forces the graph to restart the generation phase with explicit corrections.
    """
    # 1. Retrieve the active context to judge the response against
    search_filters = current_state.get("current_filters", {})
    detected_intent = current_state.get("active_intent", "idle")
    is_query_out_of_domain = current_state.get("is_out_of_domain", False)
    tool_execution_results = current_state.get("tool_outputs", {})
    candidate_properties = tool_execution_results.get("recommendations", {}).get("candidates", [])
    discussion_briefs = tool_execution_results.get("discussion_briefs", [])

    # Skip only truly non-actionable turns.
    if detected_intent == "idle" or is_query_out_of_domain or (not candidate_properties and not discussion_briefs and detected_intent in ["search", "discussion", "analyze"]):
        skip_reason = "Greeting/Idle" if detected_intent == "idle" else "Out of Domain" if is_query_out_of_domain else "No Data matches"
        print(f"--- [AUDITOR_PASS] Skipping strict check for {skip_reason} ---")
        return {
            "confidence_score": 1.0, 
            "audit_retries": 0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }

    # 2. Deterministic audit first (conversation-aware, fast, and stable).
    deterministic_result = _deterministic_audit(current_state)
    current_retry_count = current_state.get("audit_retries", 0)
    max_allowed_retries = AGENT_CONFIG["max_audit_retries"]
    if not deterministic_result.get("pass", True):
        if current_retry_count >= max_allowed_retries:
            print(f"--- [AUDITOR] Max retries reached after deterministic fail. Passing through. ---")
            return {
                "confidence_score": 1.0,
                "audit_retries": 0,
                "active_model": current_state.get("active_model", PRIMARY_MODEL)
            }
        print(f"!!! [AUDITOR_FAIL] {deterministic_result.get('reason')}")
        return {
            "confidence_score": 0.0,
            "audit_retries": current_retry_count + 1,
            "missing_info": [f"CORRECTION: {deterministic_result.get('correction', 'Make the answer precise and grounded in tool outputs.')}"],
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }

    # 3. Optional LLM audit (hybrid mode) for deeper hallucination checks.
    auditor_mode = AGENT_CONFIG.get("auditor_mode", "hybrid")
    use_llm_audit = AGENT_CONFIG.get("auditor_use_llm", False)
    if auditor_mode == "deterministic" or not use_llm_audit:
        print("--- [AUDITOR_PASS] Deterministic checks passed ---")
        return {
            "confidence_score": 1.0,
            "audit_retries": 0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }

    # 4. Prepare a dense, factual summary of the database results for the Auditor LLM
    tool_context_string = f"Found {len(candidate_properties)} properties."
    if candidate_properties:
        # Include detailed specs so the Auditor doesn't falsely flag accurate statements as hallucinations
        property_details_summary = []
        for property_data in candidate_properties[:3]:
            property_details_summary.append(f"ID:{property_data.get('listing_id')} | {property_data.get('price_egp'):,.0f} EGP | {property_data.get('bedrooms')}BR | {property_data.get('area')}sqm | {property_data.get('payment_method')}")
        tool_context_string += f" Details: {'; '.join(property_details_summary)}"
    else:
        tool_context_string += " (Zero results found in database for these filters)"
    
    # Extract the actual text the Assistant just tried to say
    latest_assistant_message = _latest_ai_text(current_state.get("messages", []))
    latest_assistant_message_for_audit = translate_for_audit(latest_assistant_message)
    
    print(f"--- [AUDITOR] Verifying Response Integrity (attempt {current_retry_count+1}) ---")
    
    # 5. Escape Hatch: If we are stuck in an infinite correction loop, force an exit
    if current_retry_count >= max_allowed_retries:
        print(f"--- [AUDITOR] Max retries reached. Passing through. ---")
        return {
            "confidence_score": 1.0, 
            "audit_retries": 0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }
    
    # Use the proven model to maintain stability
    target_llm_model = current_state.get("active_model", PRIMARY_MODEL)
    available_fallbacks = FALLBACK_MODELS_LIST if target_llm_model == PRIMARY_MODEL else []
    
    # 6. Execute the verification prompt
    try:
        formatted_auditor_prompt = AUDITOR_SYSTEM_PROMPT.format(
            intent=detected_intent,
            filters=json.dumps(search_filters), 
            tool_context=tool_context_string,
            response=latest_assistant_message_for_audit
        )
        print(f"--- [AUDITOR] Calling Model: {target_llm_model} ---")
        llm_response = litellm_completion_with_groq_key_fallback(
            model=target_llm_model,
            messages=[
                {"role": "system", "content": formatted_auditor_prompt},
                {"role": "user", "content": "Analyze compliance."}
            ],
            response_format={"type": "json_object"},
            fallbacks=available_fallbacks
        )
        
        actual_model_used = llm_response.model
        
        # Restore base model string if altered by the LiteLLM fallback wrapper
        if "/" not in actual_model_used:
            for fallback_candidate in [PRIMARY_MODEL] + FALLBACK_MODELS_LIST:
                if actual_model_used in fallback_candidate:
                    actual_model_used = fallback_candidate
                    break
                    
        print(f"--- [AUDITOR] Target: {target_llm_model} | Actual: {actual_model_used} ---")
        raw_llm_response_content = llm_response.choices[0].message.content or "{}"
        parsed_audit_result = json.loads(raw_llm_response_content)
        audit_verdict = parsed_audit_result.get("verdict", "PASS")

        # 7. Handle the Verdict: If FAIL, decrement confidence to route back to Brain
        if audit_verdict == "FAIL":
            print(f"!!! [AUDITOR_FAIL] {parsed_audit_result.get('reason')}")
            return {
                "confidence_score": 0.0, # Forces the conditional router back to start
                "audit_retries": current_retry_count + 1,
                "missing_info": [f"CORRECTION: {parsed_audit_result.get('correction_instruction')}"],
                "active_model": actual_model_used
            }
        
        # If PASS, allow the graph to reach END
        print(f"--- [AUDITOR_PASS] Integrity Verified ---")
        return {
            "confidence_score": 1.0, 
            "audit_retries": 0,
            "active_model": actual_model_used
        }
        
    except Exception as auditor_error:
        # Failsafe: if optional LLM audit crashes, do not block the response.
        print(f"!!! [AUDITOR_ERROR] Optional LLM verification skipped: {auditor_error}")
        return {
            "confidence_score": 1.0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }
