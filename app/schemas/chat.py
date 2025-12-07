from typing import Optional
from pydantic import BaseModel

class ChatIndexRequest(BaseModel):
    question: str
    answer: str
    inbox_id: str
    property_id: str
    user_id: str
    timestamp: int  # Epoch timestamp

class ChatSearchRequest(BaseModel):
    query: str
    size: int = 5
    inbox_id: Optional[str] = None
    property_id: Optional[str] = None
    user_id: Optional[str] = None
    date_from: Optional[str] = None  # YYYY-MM-DD...
    date_to: Optional[str] = None

class ChatSearchResult(BaseModel):
    text: str
    score: float
    question: str
    answer: str
    inbox_id: str
    property_id: str
    user_id: str
    timestamp: int
