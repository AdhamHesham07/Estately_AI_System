import os
import sys
import json
import importlib

# Ensure cross-module imports work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Dynamically import LiteLLM utilities from the Agent module
try:
    sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
    translation_utils = importlib.import_module("translation_utils")
    litellm_completion_with_groq_key_fallback = translation_utils.litellm_completion_with_groq_key_fallback
except ImportError:
    print("!!! [WARNING] Could not load LiteLLM fallback from Agent module.")
    litellm_completion_with_groq_key_fallback = None

ANALYZER_MODEL = "groq/llama3-70b-8192"
FALLBACK_MODEL = "gemini/gemini-1.5-pro"

ANALYZER_LLM_PROMPT = """
You are the Deep Analytical Engine for an Elite Real Estate AI.
Your sole purpose is to ingest raw mathematical statistics (from Pandas) and qualitative market news (from RAG), 
and synthesize them into a highly factual, logically sound Market Analysis Report.

USER QUERY:
"{user_query}"

STREAM 1: QUALITATIVE MARKET NEWS (RAG Context)
{knowledge_context}

STREAM 2: QUANTITATIVE MACRO STATS (Market Pulse)
{market_pulse_stats}

STREAM 3: PERSONAL PREFERENCES & TRADEOFFS (Preference Engine)
{preference_data}

INSTRUCTIONS FOR SYNTHESIS:
1. Merge the Streams: Correlate the hard numbers from Stream 2 with the narrative "why" from Stream 1.
2. Address Tradeoffs: If Stream 3 contains tradeoffs (e.g., "Budget too low for 3 beds in Zayed"), explain the mathematical reality clearly.
3. Be Factual and Direct: Do NOT use conversational filler (no "Hello", "I'd be happy to help", etc.). You are an analytical engine.
4. Output Format: Use Markdown. Use bullet points for key metrics and bold text for numbers.
5. Limit your synthesis to 200-250 words. Focus strictly on answering the User Query using the provided streams.
"""

class AnalyzerSynthesizer:
    """
    The independent LLM Node for the Analyzer Module.
    Takes disparate data streams (Pandas, RAG, Preferences) and generates a unified, 
    data-driven analytical report before handing it back to the Agent.
    """
    
    @staticmethod
    def synthesize_report(user_query: str, market_pulse_stats: dict, knowledge_context: str, preference_data: dict) -> str:
        """
        Invokes the LLM to generate the analytical synthesis.
        """
        if litellm_completion_with_groq_key_fallback is None:
            return "[Error: Analyzer LLM dependencies not loaded.]"
            
        formatted_prompt = ANALYZER_LLM_PROMPT.format(
            user_query=user_query,
            knowledge_context=knowledge_context if knowledge_context else "No external context available.",
            market_pulse_stats=json.dumps(market_pulse_stats, indent=2) if market_pulse_stats else "No macro stats available.",
            preference_data=json.dumps(preference_data, indent=2) if preference_data else "No personal tradeoff data required."
        )
        
        try:
            print(f"--- [ANALYZER LLM] Synthesizing report for query: '{user_query}' ---")
            
            # We prefer Llama-3-70B for strict logical synthesis, with Gemini 1.5 Pro as the massive-context fallback
            response = litellm_completion_with_groq_key_fallback(
                model=ANALYZER_MODEL,
                messages=[{"role": "system", "content": "You are a factual statistical synthesis engine."}, 
                          {"role": "user", "content": formatted_prompt}],
                fallbacks=[FALLBACK_MODEL]
            )
            
            synthesis_text = response.choices[0].message.content
            print("--- [ANALYZER LLM] Synthesis Complete ---")
            return synthesis_text
            
        except Exception as e:
            print(f"!!! [ERROR] Analyzer LLM Synthesis Failed: {e}")
            return f"[Synthesis Error: The Analytical Engine failed to generate a report due to an API error: {str(e)}]"

if __name__ == "__main__":
    # Test execution
    test_synthesis = AnalyzerSynthesizer.synthesize_report(
        user_query="Is it a good idea to invest in a Villa in New Cairo right now?",
        market_pulse_stats={"median_price": 15000000, "market_velocity": "High"},
        knowledge_context="New Cairo is seeing massive infrastructure upgrades, making it prime for long-term villa investments.",
        preference_data={}
    )
    print("\n[SYNTHESIS RESULT]\n", test_synthesis)
