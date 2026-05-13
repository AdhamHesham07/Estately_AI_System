import os
import sys
import importlib
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add current directory to path to allow absolute imports
sys.path.append(os.path.dirname(__file__))

# Import the State Definition and the Node modules
AgentState = importlib.import_module("3_state_definition").AgentState
intent_node = importlib.import_module("2_nodes.1_intent_node").intent_node
tool_node = importlib.import_module("2_nodes.2_tool_node").tool_node
translator_node = importlib.import_module("2_nodes.3_translator").translator_node
auditor_node = importlib.import_module("2_nodes.4_auditor").auditor_node

def route_from_brain_to_next_step(current_agent_state: AgentState) -> Literal["tools", "tongue"]:
    """
    Determines where the workflow should go after the Brain (Intent Node) processes the input.
    If information is missing or the topic is out of domain, skip tools and go to the Tongue.
    Otherwise, route to the Tools to execute the required backend operations.
    """
    if current_agent_state.get("is_out_of_domain") or current_agent_state.get("missing_info"):
        return "tongue"
    if current_agent_state.get("response_mode") == "context_only":
        return "tongue"
    if current_agent_state.get("active_intent") in ["search", "analyze", "book", "discussion"]:
        return "tools"
    return "tongue"

def route_from_auditor_to_next_step(current_agent_state: AgentState) -> Literal["pass", "fail"]:
    """
    Evaluates the quality check from the Auditor node.
    If the confidence score is too low, loop back to the Brain to correct the error.
    Otherwise, finish the workflow.
    """
    if current_agent_state.get("confidence_score", 1.0) < 0.5:
        return "fail"
    return "pass"

def build_agent_graph():
    """
    Assembles the LangGraph workflow, defining the nodes and the conditional edges connecting them.
    This architecture includes an Auditor layer for strict quality control.
    """
    # 1. Initialize the State Graph with the predefined AgentState
    agent_workflow_graph = StateGraph(AgentState)
    
    # 2. Add the core functional nodes to the graph
    agent_workflow_graph.add_node("brain", intent_node)         # Node for intent extraction
    agent_workflow_graph.add_node("tools", tool_node)           # Node for backend tool execution
    agent_workflow_graph.add_node("tongue", translator_node)    # Node for narrative generation
    agent_workflow_graph.add_node("auditor", auditor_node)      # Node for quality assurance
    
    # Set the starting point of the graph
    agent_workflow_graph.set_entry_point("brain")
    
    # 3. Define the conditional routing logic between nodes
    # From Brain: route to either Tools (if actionable) or Tongue (if idle/missing info)
    agent_workflow_graph.add_conditional_edges(
        "brain", 
        route_from_brain_to_next_step, 
        {"tools": "tools", "tongue": "tongue"}
    )
    
    # Tools always flow into the Tongue to present results
    agent_workflow_graph.add_edge("tools", "tongue")
    # Tongue always flows into the Auditor to verify output safety
    agent_workflow_graph.add_edge("tongue", "auditor")
    
    # From Auditor: route to END if passed, or loop back to Brain if failed
    agent_workflow_graph.add_conditional_edges(
        "auditor",
        route_from_auditor_to_next_step,
        {
            "pass": END,
            "fail": "brain" # Go back to brain to re-evaluate based on correction
        }
    )
    
    # 4. Add Memory Persistence to maintain conversational context
    checkpoint_memory_saver = MemorySaver()
    
    # Compile the graph into an executable application
    return agent_workflow_graph.compile(checkpointer=checkpoint_memory_saver)

if __name__ == "__main__":
    # Test the compilation of the workflow
    compiled_agent_application = build_agent_graph()
    print("Agent Graph successfully compiled.")
