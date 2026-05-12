import os
import litellm
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

litellm._turn_on_debug()

print(f"Testing model: gemini/gemini-3-flash-preview")
try:
    response = litellm.completion(
        model="gemini/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hi"}]
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
