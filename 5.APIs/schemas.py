from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ==========================================
# Recommender Schemas
# ==========================================
class RecommenderQuery(BaseModel):
    category: str = Field(default="buy", description="'buy' or 'rent'")
    city: Optional[str] = Field(default=None, description="e.g. 'Cairo'")
    town: Optional[str] = Field(default=None, description="e.g. 'New Cairo City'")
    property_type: str = Field(default="Apartment", description="e.g. 'Apartment', 'Villa'")
    bedrooms: Optional[int] = Field(default=None, description="Number of bedrooms")
    bathrooms: Optional[int] = Field(default=None, description="Number of bathrooms")
    price_egp: Optional[float] = Field(default=None, description="Target price in EGP")
    price_max: Optional[float] = Field(default=None, description="Maximum budget in EGP")
    area_value: Optional[float] = Field(default=None, description="Target area in sqm")
    amenities: Optional[List[str]] = Field(default=[], description="List of desired amenities")
    
class RecommenderResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    is_vague: bool

# ==========================================
# Analyzer Schemas
# ==========================================
class FairPriceQuery(BaseModel):
    category: str = Field(default="buy", description="'buy' or 'rent'")
    town: str = Field(..., description="e.g. 'New Cairo City'")
    district: Optional[str] = Field(default=None)
    property_type: str = Field(default="Apartment")
    bedrooms: int = Field(default=3)
    asking_price: float = Field(..., description="The price to evaluate")
    row_area: float = Field(..., description="Area in sqm")
    is_furnished: bool = Field(default=False)

class FairPriceResponse(BaseModel):
    status: str
    message: Optional[str] = None
    verdict: Optional[str] = None
    market_stats: Optional[Dict[str, Any]] = None
    query: Optional[Dict[str, Any]] = None

class MarketPulseResponse(BaseModel):
    status: str
    market_volume: Optional[Dict[str, Any]] = None
    price_economics: Optional[Dict[str, Any]] = None
    market_health: Optional[Dict[str, Any]] = None

# ==========================================
# Agent Chatbot Schemas
# ==========================================
class ChatMessage(BaseModel):
    session_id: str = Field(..., description="Unique ID for the user's conversation to maintain memory")
    message: str = Field(..., description="The user's message text")
    lang: str = Field(default="en", description="Language of the user (e.g., 'en', 'ar')")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The AI agent's response")
    intent: Optional[str] = Field(default="unknown", description="The active intent detected by the agent")
    missing_info: Optional[List[str]] = Field(default=[], description="List of missing information fields")
    recommended_properties: Optional[List[Dict[str, Any]]] = Field(default=None, description="Structured properties data to display as cards")
    error: Optional[str] = None
