"""
Centralized Agent configuration.
Keeps all tunable parameters in one place so no node needs to hardcode anything.
"""

AGENT_CONFIG = {
    # LiteLLM model string - Primary model for the agent (Cloud)
    "model": "groq/llama-3.3-70b-versatile",
    
    # List of fallback models to use if the primary model fails
    "fallback_models": ["gemini/gemini-2.0-flash"],

    # Auditor retry guard: max times the loop can send back for correction
    "max_audit_retries": 2,

    # Intent extraction minimum confidence to proceed without asking for more info
    "min_intent_confidence": 0.5,
}
