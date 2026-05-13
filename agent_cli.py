import logging
import os
import sys
import uuid
import importlib
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Ensure relative imports from Agent module work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
build_agent_graph = importlib.import_module("4_graph_builder").build_agent_graph

load_dotenv(".env")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_cli():
    agent_app = build_agent_graph()
    
    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\n[AGENT] Real Estate AI Agent CLI")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Wrap input in HumanMessage with fully initialized state
        inputs = {
            "messages": [HumanMessage(content=user_input)],
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
        
        try:
            # Execute Graph
            result = agent_app.invoke(inputs, config)
            
            state_snapshot = {
                "active_intent": result.get("active_intent"),
                "user_language": result.get("user_language"),
            }
            print(f"[STATE] {state_snapshot}")

            # Extract final message
            messages = result.get("messages", [])
            if messages:
                print(f"Agent: {messages[-1].content}\n")
            else:
                print("Agent: [No response generated]\n")
                
        except Exception as e:
            print(f"[ERROR] {e}\n")

if __name__ == "__main__":
    run_cli()
