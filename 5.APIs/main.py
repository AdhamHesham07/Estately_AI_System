import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

# Add project root to sys.path so we can import the other modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import our custom models
from schemas import (
    RecommenderQuery, RecommenderResponse,
    FairPriceQuery, FairPriceResponse,
    MarketPulseResponse,
    ChatMessage, ChatResponse
)

# Import the core AI modules
# Note: We wrap the imports in try-except in case they require specific environments, 
# but they should work since we're in the right directory.
try:
    from importlib import import_module
    
    # Recommender
    sys.path.append(os.path.join(BASE_DIR, "2.Recommender"))
    QueryAdapter = import_module("5_query_adapter").QueryAdapter
    
    # Analyzer
    sys.path.append(os.path.join(BASE_DIR, "3.Analyzer"))
    market_engine = import_module("3_market_engine")
    FairPriceEstimator = market_engine.FairPriceEstimator
    MarketPulse = market_engine.MarketPulse
    
    # Agent Chatbot
    sys.path.append(os.path.join(BASE_DIR, "1.Agent"))
    build_agent_graph = import_module("4_graph_builder").build_agent_graph
    
    # Initialize the Agent graph once at startup
    print("Initializing Agent Graph...")
    agent_app = build_agent_graph()
    
except Exception as e:
    import traceback
    print(f"Error importing core modules: {e}")
    traceback.print_exc()
    # Still start the API but endpoints will fail, useful for debugging
    QueryAdapter = None
    FairPriceEstimator = None
    MarketPulse = None
    agent_app = None

# Initialize FastAPI
app = FastAPI(
    title="Real Estate AI System API",
    description="API for Recommender, Analyzer, and Agent Chatbot. Built for .NET integration.",
    version="1.0.0"
)

# Enable CORS for the .NET frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for easy integration
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods
    allow_headers=["*"], # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate AI API. Visit /docs for the Swagger UI."}

# ==========================================
# 1. Recommender Endpoints
# ==========================================
@app.post("/api/v1/recommender/search", response_model=RecommenderResponse, tags=["Recommender"])
def search_properties(query: RecommenderQuery):
    if not QueryAdapter:
        raise HTTPException(status_code=500, detail="Recommender module not initialized.")
    
    try:
        # Convert Pydantic model to dictionary, removing None values
        query_dict = {k: v for k, v in query.model_dump().items() if v is not None}
        
        results = QueryAdapter.get_ranked_candidates(query_dict, top_k=20)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# ==========================================
# 2. Analyzer Endpoints
# ==========================================
@app.post("/api/v1/analyzer/fair-price", response_model=FairPriceResponse, tags=["Analyzer"])
def check_fair_price(query: FairPriceQuery):
    if not FairPriceEstimator:
        raise HTTPException(status_code=500, detail="Analyzer module not initialized.")
    
    try:
        res = FairPriceEstimator.estimate(
            category=query.category,
            town=query.town,
            district=query.district,
            property_type=query.property_type,
            bedrooms=query.bedrooms,
            asking_price=query.asking_price,
            row_area=query.row_area,
            is_furnished=query.is_furnished
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {str(e)}")

@app.get("/api/v1/analyzer/market-pulse", response_model=MarketPulseResponse, tags=["Analyzer"])
def get_market_pulse():
    if not MarketPulse:
        raise HTTPException(status_code=500, detail="Analyzer module not initialized.")
    
    try:
        res = MarketPulse.get_global_pulse()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {str(e)}")

# ==========================================
# 3. Agent Chatbot Endpoints
# ==========================================
@app.post("/api/v1/agent/chat", response_model=ChatResponse, tags=["Agent Chatbot"])
def chat_with_agent(chat_input: ChatMessage):
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent module not initialized.")
    
    try:
        # Configure memory for the specific session
        config = {"configurable": {"thread_id": chat_input.session_id}}
        inputs = {"messages": [HumanMessage(content=chat_input.message)], "user_language": chat_input.lang}
        
        # Invoke the LangGraph agent
        result = agent_app.invoke(inputs, config)
        
        # Extract the latest response message
        messages = result.get("messages", [])
        intent = result.get("active_intent", "unknown")
        missing_info = result.get("missing_info", [])
        
        if not messages:
            return {"reply": "I'm sorry, I couldn't generate a response.", "intent": "unknown", "missing_info": []}
            
        # Get the last message's content
        last_message = messages[-1]
        reply_text = getattr(last_message, "content", str(last_message))
        
        return {
            "reply": reply_text, 
            "intent": intent, 
            "missing_info": missing_info
        }
        
    except Exception as e:
         return {"reply": "", "error": str(e), "intent": "error", "missing_info": []}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
