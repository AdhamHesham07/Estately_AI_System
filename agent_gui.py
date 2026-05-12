import os
import sys
import gradio as gr
import uuid
import importlib
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Ensure relative imports from Agent module work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
build_agent_graph = importlib.import_module("4_graph_builder").build_agent_graph

load_dotenv(".env")

# 1. Initialize the Agent
print("--- [GUI] Booting AI Agent Graph... ---")
agent_app = build_agent_graph()

def chat_response(message, history):
    """
    Handles communication between Gradio UI and LangGraph Agent.
    """
    # Gradio 6.0 session persistence: we'll use a session-based ID if possible, 
    # but for simplicity in the basic GUI, we'll use a stable one or generate per chat.
    thread_id = "gui_default_session" 
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        inputs = {"messages": [HumanMessage(content=message)]}
        result = agent_app.invoke(inputs, config)
        
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
            
        return "I'm sorry, I encountered an internal processing error."

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# 2. Build the Gradio Interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="🏢 Real Estate AI Agent v2.0",
    description="Connected to Live SQL Database & Recommender Engine",
    examples=["Find me an apartment in New Cairo", "Show me villa prices", "I want to buy a 3-bedroom apartment in Fifth Settlement for 8 million"],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7861, 
        share=False
    )
