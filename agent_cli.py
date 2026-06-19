import argparse
import json
import logging
import os
import sys
import uuid
from importlib import import_module
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_API_URL = "http://127.0.0.1:8000"


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for logger_name in ("litellm", "langchain_core", "httpx", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def post_json(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    return send_request(request)


def get_json(base_url: str, path: str) -> Dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    return send_request(request)


def send_request(request: Request) -> Dict[str, Any]:
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = raw
        return {"error": f"HTTP {error.code}", "detail": detail}
    except URLError as error:
        return {
            "error": "API connection failed",
            "detail": str(error.reason),
            "hint": "Start the API first with: python 5.APIs/main.py",
        }


def clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def run_api_chat(args: argparse.Namespace) -> None:
    session_id = args.session_id or f"cli_{uuid.uuid4()}"
    if args.interactive:
        print(f"\n[API CHAT] Session: {session_id}")
        print("Type 'exit' to quit.\n")
        while True:
            message = input("User: ").strip()
            if message.lower() in {"exit", "quit"}:
                break
            if not message:
                continue
            response = post_json(
                args.api_url,
                "/api/v1/agent/chat",
                {"session_id": session_id, "message": message, "lang": args.lang},
            )
            print_agent_response(response)
        return

    response = post_json(
        args.api_url,
        "/api/v1/agent/chat",
        {"session_id": session_id, "message": args.message, "lang": args.lang},
    )
    print_agent_response(response)


def print_agent_response(response: Dict[str, Any]) -> None:
    if response.get("error"):
        print_json(response)
        return

    print(f"Intent: {response.get('intent', 'unknown')}")
    missing_info = response.get("missing_info") or []
    if missing_info:
        print(f"Missing: {', '.join(missing_info)}")
    print(f"Agent: {response.get('reply', '')}\n")

    recommended = response.get("recommended_properties") or []
    if recommended:
        print("Recommended properties:")
        for item in recommended[:5]:
            title = item.get("title") or item.get("property_type") or "Property"
            listing_id = item.get("listing_id", "unknown")
            price = item.get("price_egp", "unknown")
            town = item.get("town") or item.get("district") or ""
            print(f"- #{listing_id}: {title} | {price} EGP | {town}")
        print()


def run_api_search(args: argparse.Namespace) -> None:
    payload = clean_payload(
        {
            "category": args.category,
            "city": args.city,
            "town": args.town,
            "property_type": args.property_type,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
            "price_egp": args.price_egp,
            "price_max": args.price_max,
            "area_value": args.area_value,
            "amenities": args.amenities,
        }
    )
    print_json(post_json(args.api_url, "/api/v1/recommender/search", payload))


def run_api_fair_price(args: argparse.Namespace) -> None:
    payload = clean_payload(
        {
            "category": args.category,
            "town": args.town,
            "district": args.district,
            "property_type": args.property_type,
            "bedrooms": args.bedrooms,
            "asking_price": args.asking_price,
            "row_area": args.row_area,
            "is_furnished": args.is_furnished,
        }
    )
    print_json(post_json(args.api_url, "/api/v1/analyzer/fair-price", payload))


def run_api_market_pulse(args: argparse.Namespace) -> None:
    print_json(get_json(args.api_url, "/api/v1/analyzer/market-pulse"))


def build_direct_agent():
    if os.path.join(BASE_DIR, "1.Agent") not in sys.path:
        sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
    return import_module("4_graph_builder").build_agent_graph()


def initial_agent_state(message: str, lang: str) -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=message)],
        "current_filters": {},
        "tool_outputs": {},
        "missing_info": [],
        "active_intent": "idle",
        "booking_details": {},
        "confidence_score": 1.0,
        "is_out_of_domain": False,
        "audit_retries": 0,
        "user_language": lang,
        "recent_properties": [],
        "discussion_context": {},
        "dialogue_state": {},
        "reference_map": {},
        "focus_listing_id": None,
        "last_recommendation_snapshot": [],
        "response_mode": "tool_required",
        "dialogue_act": "general",
        "target_property_refs": [],
        "carry_forward_slots": {},
        "slot_updates": {},
        "response_plan": {},
    }


def run_direct_agent(args: argparse.Namespace) -> None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    agent_app = build_direct_agent()
    thread_id = args.session_id or f"direct_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    if args.interactive:
        print(f"\n[DIRECT AGENT] Session: {thread_id}")
        print("Type 'exit' to quit.\n")
        while True:
            message = input("User: ").strip()
            if message.lower() in {"exit", "quit"}:
                break
            if not message:
                continue
            invoke_direct_agent(agent_app, config, message, args.lang, args.show_state)
        return

    invoke_direct_agent(agent_app, config, args.message, args.lang, args.show_state)


def invoke_direct_agent(agent_app: Any, config: Dict[str, Any], message: str, lang: str, show_state: bool) -> None:
    try:
        result = agent_app.invoke(initial_agent_state(message, lang), config)
    except Exception as error:
        print(f"[ERROR] {error}")
        return

    if show_state:
        state_snapshot = {
            "active_intent": result.get("active_intent"),
            "missing_info": result.get("missing_info"),
            "user_language": result.get("user_language"),
        }
        print_json(state_snapshot)

    messages = result.get("messages", [])
    if messages:
        print(f"Agent: {messages[-1].content}\n")
    else:
        print("Agent: [No response generated]\n")


def add_common_api_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API base URL. Default: {DEFAULT_API_URL}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tester for the Estately API and local agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_chat = subparsers.add_parser("api-chat", help="Test /api/v1/agent/chat.")
    add_common_api_arg(api_chat)
    api_chat.add_argument("message", nargs="?", help="Message to send. Omit with --interactive.")
    api_chat.add_argument("--interactive", "-i", action="store_true", help="Start a multi-turn API chat session.")
    api_chat.add_argument("--session-id", help="Reuse a session id for memory.")
    api_chat.add_argument("--lang", default="en", help="User language, such as en or ar.")
    api_chat.set_defaults(func=run_api_chat)

    api_search = subparsers.add_parser("api-search", help="Test /api/v1/recommender/search.")
    add_common_api_arg(api_search)
    api_search.add_argument("--category", default="buy", choices=["buy", "rent"])
    api_search.add_argument("--city")
    api_search.add_argument("--town")
    api_search.add_argument("--property-type", default="Apartment")
    api_search.add_argument("--bedrooms", type=int)
    api_search.add_argument("--bathrooms", type=int)
    api_search.add_argument("--price-egp", type=float)
    api_search.add_argument("--price-max", type=float)
    api_search.add_argument("--area-value", type=float)
    api_search.add_argument("--amenities", nargs="*", default=[])
    api_search.set_defaults(func=run_api_search)

    fair_price = subparsers.add_parser("api-fair-price", help="Test /api/v1/analyzer/fair-price.")
    add_common_api_arg(fair_price)
    fair_price.add_argument("--category", default="buy", choices=["buy", "rent"])
    fair_price.add_argument("--town", required=True)
    fair_price.add_argument("--district")
    fair_price.add_argument("--property-type", default="Apartment")
    fair_price.add_argument("--bedrooms", type=int, default=3)
    fair_price.add_argument("--asking-price", type=float, required=True)
    fair_price.add_argument("--row-area", type=float, required=True)
    fair_price.add_argument("--is-furnished", action="store_true")
    fair_price.set_defaults(func=run_api_fair_price)

    market = subparsers.add_parser("api-market", help="Test /api/v1/analyzer/market-pulse.")
    add_common_api_arg(market)
    market.set_defaults(func=run_api_market_pulse)

    direct = subparsers.add_parser("agent", help="Test the LangGraph agent directly, without the API server.")
    direct.add_argument("message", nargs="?", help="Message to send. Omit with --interactive.")
    direct.add_argument("--interactive", "-i", action="store_true", help="Start a multi-turn direct agent chat.")
    direct.add_argument("--session-id", help="Reuse a session id for memory.")
    direct.add_argument("--lang", default="en", help="User language, such as en or ar.")
    direct.add_argument("--show-state", action="store_true", help="Print selected state fields after each turn.")
    direct.set_defaults(func=run_direct_agent)

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command in {"api-chat", "agent"} and not args.interactive and not args.message:
        parser.error(f"{args.command} requires a message unless --interactive is used.")


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    args.func(args)


if __name__ == "__main__":
    main()
