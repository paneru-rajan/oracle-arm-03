from typing import List, Optional
from pydantic import BaseModel

class SemanticIndexRequest(BaseModel):
    chunk: str
    alias: str
    timestamp: int  # Epoch timestamp
    chunk_id: int
    tag: List[str]
    content_type: str

class SemanticSearchRequest(BaseModel):
    query: str
    size: int = 5
    date_from: Optional[str] = None  # YYYY-MM-DD HH:MM:SS
    date_to: Optional[str] = None    # YYYY-MM-DD HH:MM:SS
    tags: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    include_embeddings: bool = False

class SemanticSearchResult(BaseModel):
    chunk: str
    score: float
    alias: str
    chunk_id: int
    timestamp: int
    tag: List[str]
    content_type: str
    embedding: Optional[List[float]] = None