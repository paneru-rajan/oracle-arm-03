from typing import Optional
from pydantic import BaseModel

class ChatIndexRequest(BaseModel):
    question: str
    answer: str
    property_id: str
    message_id: Optional[str] = None
    host_id: Optional[str] = None
    guest_id: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ChatSearchRequest(BaseModel):
    query: str
    size: int = 5
    property_id: Optional[str] = None
    host_id: Optional[str] = None
    guest_id: Optional[str] = None
    category: Optional[str] = None
    date_from: Optional[str] = None  # YYYY-MM-DD...
    date_to: Optional[str] = None

class ChatSearchResult(BaseModel):
    score: float
    question: str
    answer: str
    property_id: str
    message_id: Optional[str] = None
    host_id: Optional[str] = None
    guest_id: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
