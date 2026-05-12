import requests
import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
if response.status_code == 200:
    models = response.json().get("models", [])
    for m in models:
        print(f"- {m['name']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
