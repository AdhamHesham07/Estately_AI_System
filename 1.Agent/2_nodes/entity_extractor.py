import re

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
    
    # 1. Location: Fifth Settlement / Tagamoa — in our data this is a district under New Cairo, not a town alias.
    fifth_settlement_triggers = [
        "fifth settlement", "5th settlement", "fifth settl", "5th settl", "tagamoa", "tagamoaa",
        "التجمع الخامس",
    ]
    if any(tok in normalized_text for tok in fifth_settlement_triggers):
        extracted_data["town"] = "New Cairo City"
        extracted_data["district"] = "The 5th Settlement"

    # Other common areas (town-level)
    if "district" not in extracted_data:
        common_locations_map = {
            "maadi": ["maadi", "madi"],
            "zayed": ["zayed", "sheikh zayed", "6th of october", "6th october"],
            "shorouk": ["shorouk", "shorok"],
            "heliopolis": ["heliopolis", "masr el gdida"],
        }
        for standard_location, trigger_tokens in common_locations_map.items():
            if any(token in normalized_text for token in trigger_tokens):
                extracted_data["town"] = standard_location.capitalize()
                break

    if "town" not in extracted_data and "new cairo" in normalized_text:
        extracted_data["town"] = "New Cairo City"

    # 2. Price: range "12-13 million" then single cap
    range_million = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(million|m|mln)",
        normalized_text,
    )
    if range_million:
        try:
            lo = float(range_million.group(1)) * 1_000_000
            hi = float(range_million.group(2)) * 1_000_000
            extracted_data["price_min"] = min(lo, hi)
            extracted_data["price_max"] = max(lo, hi)
        except ValueError:
            pass

    if "price_max" not in extracted_data and "price_min" not in extracted_data:
        price_pattern_match = re.search(r"(\d+(?:\.\d+)?)\s*(million|m|mln)", normalized_text)
        if price_pattern_match:
            try:
                numeric_value = float(price_pattern_match.group(1))
                extracted_data["price_max"] = numeric_value * 1_000_000
            except ValueError:
                pass
        elif "million" not in normalized_text:
            large_numbers = re.findall(r"\b(\d{6,10})\b", normalized_text)
            if large_numbers:
                extracted_data["price_max"] = float(large_numbers[0])

    # 3. Property Type categorization
    if "apart" in normalized_text or "flat" in normalized_text:
        extracted_data["property_type"] = "Apartment"
    elif "villa" in normalized_text or "townhouse" in normalized_text:
        extracted_data["property_type"] = "Villa"
        
    return extracted_data

def detect_dialogue_act(latest_user_message: str) -> str:
    normalized = f" {str(latest_user_message or '').lower()} "
    show_more_triggers = [
        " more recommendations ", " more options ", " more properties ",
        " show me more ", " show more ", " any other options ",
        " other options ", " next options ", " next recommendations ",
        " another option ", " alternatives "
    ]
    if any(token in normalized for token in show_more_triggers):
        return "show_more"
    if any(token in normalized for token in [" compare ", " vs ", " versus ", " difference ", " better "]):
        return "compare"
    if any(token in normalized for token in [" details ", " tell me more ", " more about ", " explain ", " pros ", " cons "]):
        return "request_details"
    if any(token in normalized for token in [" cheaper ", " same ", " instead ", " change ", " but ", " different ", " alternative "]):
        return "refine_search"
    if any(token in normalized for token in [" yes ", " okay ", " go ahead ", " confirm "]):
        return "confirm"
    if any(token in normalized for token in [" hi ", " hello ", " hey ", " help ", " guide ", " what can you do "]):
        return "chitchat"
    return "general"

def extract_property_refs(latest_user_message: str) -> list:
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
