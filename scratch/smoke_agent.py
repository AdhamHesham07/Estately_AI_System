"""Quick end-to-end smoke: graph build + one invoke (idle) + one invoke (search if DB ok)."""
import os
import sys
import uuid
import importlib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.append(os.path.join(ROOT, "1.Agent"))

build_agent_graph = importlib.import_module("4_graph_builder").build_agent_graph
agent_app = build_agent_graph()


def base_state():
    return {
        "current_filters": {},
        "tool_outputs": {},
        "missing_info": [],
        "active_intent": "idle",
        "booking_details": {},
        "confidence_score": 1.0,
        "is_out_of_domain": False,
        "audit_retries": 0,
        "user_language": "en",
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


def run_turn(config, state, text):
    state["messages"] = [HumanMessage(content=text)]
    return agent_app.invoke(state, config)


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = base_state()

    r1 = run_turn(config, state, "Hello")
    m1 = (r1.get("messages") or [])[-1].content if r1.get("messages") else ""
    print("[1] idle reply:", (m1[:400] + "…") if len(m1) > 400 else m1)
    print("    intent:", r1.get("active_intent"))

    r2 = run_turn(config, state, "I want a 2 bedroom apartment in Maadi under 5 million EGP.")
    m2 = (r2.get("messages") or [])[-1].content if r2.get("messages") else ""
    print("[2] search reply:", (m2[:500] + "…") if len(m2) > 500 else m2)
    print("    intent:", r2.get("active_intent"), "recent_n:", len(r2.get("recent_properties") or []))

    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
