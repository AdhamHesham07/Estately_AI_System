import os
import sys
import json
import importlib
from typing import Dict, Any, Literal
from langchain_core.messages import AIMessage
import litellm

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
AUDITOR_SYSTEM_PROMPT = """
You are the Quality Auditor for a Real Estate AI. 
Compare the Assistant's Response against the User's Intent and Required Filters.

CURRENT INTENT: {intent}
USER FILTERS: {filters}
TOOL RESULTS: {tool_context}

ASSISTANT RESPONSE:
{response}

AUDIT RULES:
1. GREETINGS/IDLE: If Intent is 'idle', only check for professional tone. Skip price/location rules.
2. GOAL ALIGNMENT: Recommending properties from the tool results is the CORRECT and DESIRED behavior. Do NOT fail the assistant for "recommending" instead of "searching".
3. CRITICAL - SCALE GATE: If a user asks for 'Millions' (e.g. 7M) and the assistant presents a property worth 'Thousands' (e.g. 670k) as a "Primary Match", it is a FAIL. 
   - Recommending a 670k property as a "Budget Alternative" to a 7M request is acceptable, but NOT as a primary match.
4. ANTI-GENERIC: If Tool Results contain properties, the Assistant MUST mention them. If the Assistant gives a generic greeting or asks for info we already have, mark as FAIL.
5. HALLUCINATION: If the Assistant mentions a price or property NOT found in the Tool Results, mark as FAIL.

JSON OUTPUT:
{{
  "verdict": "PASS" | "FAIL",
  "reason": "Short explanation",
  "correction_instruction": "E.g., 'Stop asking for location, we already have it' or 'Price scale error: User asked for Millions, you showed Thousands as primary'"
}}
"""

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
    
    # AGGRESSIVE GUARD: Do not audit trivial greetings or queries outside the AI's capability
    has_meaningful_filters = any(search_filters.get(key) for key in ["town", "price_max", "district", "town_id"])
    is_query_out_of_domain = current_state.get("is_out_of_domain", False)
    
    tool_execution_results = current_state.get("tool_outputs", {})
    candidate_properties = tool_execution_results.get("recommendations", {}).get("candidates", [])
    
    if detected_intent not in ["search", "analyze"] or not has_meaningful_filters or is_query_out_of_domain or not candidate_properties:
        skip_reason = "Greeting/Idle" if detected_intent == "idle" else "Discussion/Booking" if detected_intent in ["discussion", "book"] else "Out of Domain" if is_query_out_of_domain else "No Data matches"
        print(f"--- [AUDITOR_PASS] Skipping strict check for {skip_reason} ---")
        return {
            "confidence_score": 1.0, 
            "audit_retries": 0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }
    
    # 2. Prepare a dense, factual summary of the database results for the Auditor LLM
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
    latest_assistant_message = current_state["messages"][-1].content if current_state["messages"] else ""
    latest_assistant_message_for_audit = translate_for_audit(latest_assistant_message)
    current_retry_count = current_state.get("audit_retries", 0)
    
    print(f"--- [AUDITOR] Verifying Response Integrity (attempt {current_retry_count+1}) ---")
    
    # 3. Escape Hatch: If we are stuck in an infinite correction loop, force an exit
    max_allowed_retries = AGENT_CONFIG["max_audit_retries"]
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
    
    # 4. Execute the verification prompt
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
        
        # 5. Handle the Verdict: If FAIL, decrement confidence to route back to Brain
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
        # Failsafe: if the auditor crashes, assume the response is okay rather than breaking
        print(f"!!! [AUDITOR_ERROR] Verification skipped: {auditor_error}")
        return {
            "confidence_score": 1.0,
            "active_model": current_state.get("active_model", PRIMARY_MODEL)
        }
