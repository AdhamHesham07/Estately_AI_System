import logging
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
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

agent_app = build_agent_graph()

def chat_response(message, history, request: gr.Request):
    """
    Handles communication between Gradio UI and LangGraph Agent.
    """
    # Use a stable per-browser-session thread id to preserve current chat context
    # without leaking memory across different users/sessions.
    session_hash = getattr(request, "session_hash", None) if request else None
    thread_id = f"gui_{session_hash}" if session_hash else str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        inputs = {"messages": [HumanMessage(content=message)], "user_language": "en"}
        result = agent_app.invoke(inputs, config)
        
        if not result:
            print("[ERROR] Agent returned an empty state or None.")
            return "I'm sorry, I encountered an internal processing error (Agent returned no data)."
            
        state_snapshot = {
            "active_intent": result.get("active_intent"),
            "user_language": result.get("user_language"),
        }
        print(f"[STATE] {state_snapshot}")

        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
            
        return "I'm sorry, I encountered an internal processing error."

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"⚠️ Error: {str(e)}"

# 2. Build the Gradio Interface
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    body_background_fill="*background_fill_primary",
    block_background_fill="*background_fill_secondary",
)

with gr.Blocks(title="Real Estate AI Agent") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="color: var(--body-text-color); margin-bottom: 5px; font-weight: 800; font-size: 2em;">🏢 Real Estate AI Agent <span style='font-size: 0.5em; color: var(--body-text-color-subdued);'>v2.0</span></h1>
            <p style="font-size: 1.1em; color: var(--body-text-color-subdued);">Connected to Live SQL Database & Recommender Engine</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown(
                """
                ### 🚀 Key Capabilities
                - **Smart Search:** Find properties using natural language.
                - **Market Analytics:** Get insights on average prices and trends.
                - **Recommendations:** Discover AI-driven property matches based on your needs.
                - **Multilingual:** Ask in English or Arabic.
                
                ---
                
                ### 💡 Example Prompts
                - *"Find an apartment in New Cairo under 5 million."*
                - *"Compare villa prices in Sheikh Zayed vs Fifth Settlement."*
                - *"I need a 3-bedroom place near the AUC for 8 million EGP."*
                """
            )
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_response,
                chatbot=gr.Chatbot(
                    height=450,
                ),
                textbox=gr.Textbox(
                    placeholder="Describe your ideal property...",
                    container=False,
                    scale=7
                ),
                examples=[
                    "Find me an apartment in New Cairo", 
                    "Show me villa prices", 
                    "I want to buy a 3-bedroom apartment in Fifth Settlement for 8 million"
                ],
                cache_examples=False,
            )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7861, 
        share=False,
        theme=custom_theme,
        css="footer {visibility: hidden}"
    )
