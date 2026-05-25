import os
import sys
import uuid
import importlib
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1.Agent"))

build_agent_graph = importlib.import_module("4_graph_builder").build_agent_graph
agent_app = build_agent_graph()

TEST_CASES = [
    (
        "EN_followup_details",
        [
            "I want to buy an apartment in New Cairo around 8 million.",
            "Tell me more details about the first suggestion.",
        ],
    ),
    (
        "EN_compare",
        [
            "Show me villas in Sheikh Zayed around 15 million.",
            "Compare the first and third options.",
        ],
    ),
    (
        "EN_refine",
        [
            "Find me a 3-bedroom apartment in Maadi for 6 million.",
            "Same area but cheaper.",
        ],
    ),
    (
        "AR_followup",
        [
            "عايز شقة في التجمع الخامس بحوالي 8 مليون",
            "ممكن تفاصيل اكتر عن اول اقتراح؟",
        ],
    ),
]


def run_case(case_name, turns):
    print(f"\n=== {case_name} ===")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = {
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
    for idx, text in enumerate(turns, start=1):
        state["messages"] = [HumanMessage(content=text)]
        result = agent_app.invoke(state, config)
        reply = (result.get("messages") or [])[-1].content if result.get("messages") else "<no response>"
        print(f"Turn {idx} User: {text}")
        print(f"Turn {idx} Agent: {reply}")
        print(
            "State:",
            {
                "intent": result.get("active_intent"),
                "dialogue_act": result.get("dialogue_act"),
                "focus_listing_id": result.get("focus_listing_id"),
                "response_mode": result.get("response_mode"),
            },
        )


if __name__ == "__main__":
    for case_name, turns in TEST_CASES:
        run_case(case_name, turns)
