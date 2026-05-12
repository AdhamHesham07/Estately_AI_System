import os
from dotenv import load_dotenv
import litellm

# Load .env from root
load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
else:
    print(f"Success: GEMINI_API_KEY found (length: {len(api_key)})")
    
    try:
        print("--- Testing LLM Call (Gemini 3 Flash) ---")
        response = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Say 'Gemini 3 is working'"}],
            api_key=api_key
        )
        print(f"Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: LLM Call Failed: {e}")
