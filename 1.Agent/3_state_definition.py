from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of the AI Real Estate Agent.
    This dictionary flows through every node in the graph, accumulating context and decisions.
    """
    # LangGraph standard: A continuous sequence of conversational messages
    messages: Annotated[List[Any], add_messages]
    
    # Structured context: Holds semantic entities extracted from the conversation (e.g. city, budget)
    current_filters: dict 
    
    # Tool output storage: Holds backend API results (e.g. property listings, market analysis)
    tool_outputs: dict 
    
    # Expert logic state: Fields the user hasn't provided yet to fulfill their intent
    missing_info: List[str] 
    # Current active workflow objective: 'search', 'analyze', 'book', or 'idle'
    active_intent: str 
    
    # Booking Workflow: Stores all necessary constraints to confirm an appointment
    booking_details: dict 
    
    # Confidence & Safety Guardrails
    confidence_score: float     # LLM confidence metric, reviewed by the Auditor
    is_out_of_domain: bool      # Flag to prevent hallucination on non-real-estate queries
    audit_retries: int          # Tracks how many times the Auditor rejected the response
    active_model: str           # Remembers the successfully used LLM for sticky fallbacks
