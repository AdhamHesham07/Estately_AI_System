import os
import sys
import re
import json
import importlib
from typing import Any, List, Optional

import litellm
from dotenv import load_dotenv

# Allow imports from the parent agent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load shared API keys for translation calls
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEYS = [key for key in [groq_api_key, groq_api_key_2] if key]
if GROQ_API_KEYS:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEYS[0]


def _is_groq_quota_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(token in error_text for token in [
        "rate limit",
        "rate_limit",
        "quota",
        "too many requests",
        "429",
        "quota exceeded",
        "rate_limit_exceeded",
        "resource_exhausted",
    ])


def litellm_completion_with_groq_key_fallback(model: str, **kwargs):
    if not model or not str(model).startswith("groq/") or len(GROQ_API_KEYS) <= 1:
        return litellm.completion(model=model, **kwargs)

    original_key = os.environ.get("GROQ_API_KEY")
    try:
        for index, key in enumerate(GROQ_API_KEYS):
            if key:
                os.environ["GROQ_API_KEY"] = key
            try:
                return litellm.completion(model=model, **kwargs)
            except Exception as ex:
                if _is_groq_quota_error(ex) and index < len(GROQ_API_KEYS) - 1:
                    continue
                raise
    finally:
        if original_key is not None:
            os.environ["GROQ_API_KEY"] = original_key
        elif "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]

AGENT_CONFIG = importlib.import_module("1_agent_config").AGENT_CONFIG
PRIMARY_MODEL = AGENT_CONFIG["model"]
FALLBACK_MODELS_LIST = AGENT_CONFIG.get("fallback_models", [])

ARABIC_DIALECT_MARKERS = [
    "عايز", "عاوز", "فين", "إزاي", "إيه", "ايه", "محتاج", "بقى", "طب", "تمام", "ماشي", "حلو", "ليه", "مش", "ممكن",
    "ليه", "بارح", "النهاردة", "كده", "ده", "دي"
]

ARABIC_SCRIPT_REGEX = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def contains_arabic(text: str) -> bool:
    if not text:
        return False
    if ARABIC_SCRIPT_REGEX.search(text):
        return True
    normalized = text.lower()
    return any(marker in normalized for marker in ARABIC_DIALECT_MARKERS)


def clean_translation_response(translated_text: str) -> str:
    if not translated_text:
        return ""
    cleaned = translated_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    return cleaned


def translate_text(text: str, instructions: str) -> str:
    if not text:
        return ""

    prompt_messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Translate this text and return only the translated output.\n\n{text}"}
    ]

    target_model = PRIMARY_MODEL
    available_fallbacks = FALLBACK_MODELS_LIST

    try:
        llm_response = litellm_completion_with_groq_key_fallback(
            model=target_model,
            messages=prompt_messages,
            fallbacks=available_fallbacks,
            max_tokens=1024
        )
        translated_text = llm_response.choices[0].message.content or ""
        return clean_translation_response(translated_text)
    except Exception:
        # Fail safely: return the original text if translation cannot be completed
        return text


def translate_to_english(text: str) -> str:
    if not contains_arabic(text):
        return text

    instructions = (
        "You are a precision translator from Arabic, including Egyptian dialect, into clear English.\n"
        "Translate the text faithfully and preserve meaning, intent, and any numeric values exactly.\n"
        "Return only the translated English text without additional commentary."
    )
    return translate_text(text, instructions)


def translate_to_arabic(text: str) -> str:
    if not text:
        return ""

    instructions = (
        "You are a precision translator. Translate the English text into natural conversational Egyptian Arabic using Arabic script.\n"
        "Use local spoken phrasing when appropriate and preserve the original meaning.\n"
        "Return only the translated Arabic text without extra explanation."
    )
    return translate_text(text, instructions)


def build_english_history(messages: List[Any]) -> str:
    history_lines: List[str] = []
    for msg in messages:
        msg_content = getattr(msg, "content", str(msg))
        role = "User" if getattr(msg, "type", None) == "human" else "AI"
        if contains_arabic(msg_content):
            msg_content = translate_to_english(msg_content)
        history_lines.append(f"{role}: {msg_content}")
    return "\n".join(history_lines)


def translate_for_audit(text: str) -> str:
    if contains_arabic(text):
        return translate_to_english(text)
    return text
