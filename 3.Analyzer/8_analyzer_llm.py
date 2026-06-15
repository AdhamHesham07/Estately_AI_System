import os
import sys
import json
import hashlib
import importlib
from cachetools import TTLCache

# --- LLM Result Caches (TTL = 1 hour, max 256 entries each) ---
# Identical queries bypass the LLM entirely, cutting API costs by ~50%.
_selection_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)
_tradeoff_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

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

ANALYZER_MODEL = "groq/llama-3.3-70b-versatile"
FALLBACK_MODEL = "gemini/gemini-2.0-flash"

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

    @staticmethod
    def analyze_and_select_top_properties(user_query: str, candidates: list) -> list:
        """
        Uses the LLM to evaluate candidate properties against the user's query,
        picks the best 3, and generates a short reasoning for each.
        Returns a list of dicts: [{'listing_id': id, 'reasoning': text}, ...]
        Results are cached by (query + candidate IDs) for 1 hour.
        """
        if not candidates:
            return []

        # ✅ FIX: Build a stable cache key from query + candidate IDs
        candidate_ids_str = ",".join(str(c.get("listing_id", "")) for c in candidates)
        cache_key = hashlib.md5(f"{user_query.strip().lower()}|{candidate_ids_str}".encode()).hexdigest()
        if cache_key in _selection_cache:
            print("--- [ANALYZER LLM] Cache HIT for property selection ---")
            return _selection_cache[cache_key]

        if litellm_completion_with_groq_key_fallback is None:
            return [{"listing_id": c.get("listing_id"), "reasoning": "Selected based on matching criteria."} for c in candidates[:3]]

        # Prepare candidates summary for prompt
        candidates_summary = []
        for c in candidates:
            candidates_summary.append(
                f"- ID {c.get('listing_id')}: {c.get('price_egp')} EGP, {c.get('bedrooms')} beds, {c.get('property_type')}, {c.get('town')} / {c.get('district')}, Area: {c.get('area_value')} sqm"
            )
        candidates_text = "\n".join(candidates_summary)

        prompt = f"""
You are the Deep Analytical Engine for an Elite Real Estate AI.
Your task is to select the TOP 3 best matching properties from the candidate list below that best satisfy the User Query.

USER QUERY:
"{user_query}"

CANDIDATES:
{candidates_text}

INSTRUCTIONS:
1. Evaluate the candidates against the user's query (budget, size, location).
2. Pick exactly the top 3 best matching properties (or fewer if less than 3 exist).
3. Provide a concise, 1-sentence reasoning for WHY each was selected.
4. Output ONLY valid JSON in the following exact format:
[
  {{"listing_id": "123", "reasoning": "This property offers the best value within the budget in the desired district."}}
]
"""
        try:
            print(f"--- [ANALYZER LLM] Selecting top properties for query: '{user_query}' ---")
            
            response = litellm_completion_with_groq_key_fallback(
                model=ANALYZER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise analytical engine that outputs only JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                fallbacks=[FALLBACK_MODEL]
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up markdown JSON formatting if present
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            selected_properties = json.loads(content.strip())
            print("--- [ANALYZER LLM] Property Selection Complete ---")
            # ✅ Store result in cache
            _selection_cache[cache_key] = selected_properties
            return selected_properties
            
        except Exception as e:
            print(f"!!! [ERROR] Analyzer LLM Property Selection Failed: {e}")
            # Fallback: Top 3 with generic reasoning
            return [{"listing_id": c.get("listing_id"), "reasoning": "Matches basic criteria."} for c in candidates[:3]]

    @staticmethod
    def generate_ai_tradeoffs(filters: dict, exact_matches_count: int, market_summary: str) -> dict:
        """
        Dynamically generates strategic tradeoffs or pivot suggestions.
        Results are cached by (filters + match count) for 1 hour.
        """
        if litellm_completion_with_groq_key_fallback is None:
            return {"status": "success", "catalog_exact_matches": exact_matches_count, "adjustments": []}

        # ✅ FIX: Build a stable cache key from filters + match count
        filters_str = json.dumps(filters, sort_keys=True)
        cache_key = hashlib.md5(f"{filters_str}|{exact_matches_count}".encode()).hexdigest()
        if cache_key in _tradeoff_cache:
            print("--- [ANALYZER LLM] Cache HIT for tradeoffs ---")
            return _tradeoff_cache[cache_key]

        prompt = f"""
You are the "Master Strategist" of an Elite Real Estate AI.
A user has submitted the following property search filters:
{json.dumps(filters, indent=2)}

The database currently has {exact_matches_count} exact matches for these criteria.

MARKET SUMMARY:
{market_summary}

INSTRUCTIONS:
1. Based on the number of matches and the market summary, suggest 1 to 3 strategic pivots or tradeoffs.
2. If matches are low (<= 2), suggest expanding the budget, shifting the location slightly, or looking at off-plan properties.
3. If matches are high (> 15), suggest filtering by top developers or ready-to-move status.
4. Keep the suggestions highly professional, data-driven, and specific to the Egyptian market.
5. Output ONLY valid JSON in the following exact format:
{{
  "status": "success",
  "catalog_exact_matches": {exact_matches_count},
  "adjustments": [
    {{
      "strategy": "Budget Expansion",
      "change": "Consider increasing your budget to X EGP",
      "delta": "This will unlock Y more premium options."
    }}
  ]
}}
"""
        try:
            print(f"--- [ANALYZER LLM] Generating strategic tradeoffs for {exact_matches_count} matches ---")
            
            response = litellm_completion_with_groq_key_fallback(
                model=ANALYZER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strategic real estate advisor that outputs only JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                fallbacks=[FALLBACK_MODEL]
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            tradeoffs = json.loads(content.strip())
            print("--- [ANALYZER LLM] Tradeoff Generation Complete ---")
            # ✅ Store result in cache
            _tradeoff_cache[cache_key] = tradeoffs
            return tradeoffs
            
        except Exception as e:
            print(f"!!! [ERROR] Analyzer LLM Tradeoff Generation Failed: {e}")
            return {"status": "success", "catalog_exact_matches": exact_matches_count, "adjustments": []}

if __name__ == "__main__":
    # Test execution
    test_synthesis = AnalyzerSynthesizer.synthesize_report(
        user_query="Is it a good idea to invest in a Villa in New Cairo right now?",
        market_pulse_stats={"median_price": 15000000, "market_velocity": "High"},
        knowledge_context="New Cairo is seeing massive infrastructure upgrades, making it prime for long-term villa investments.",
        preference_data={}
    )
    print("\n[SYNTHESIS RESULT]\n", test_synthesis)
