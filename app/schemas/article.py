from typing import List, Optional
from pydantic import BaseModel

class SemanticIndexRequest(BaseModel):
    chunk: str
    id: str
    timestamp: int  # Epoch timestamp
    chunk_id: int
    tag: List[str]
    type: str

class SemanticSearchRequest(BaseModel):
    query: str
    size: int = 5
    date_from: Optional[str] = None  # YYYY-MM-DD HH:MM:SS
    date_to: Optional[str] = None    # YYYY-MM-DD HH:MM:SS
    tags: Optional[List[str]] = None
    types: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    include_embeddings: bool = False

class SemanticSearchResult(BaseModel):
    chunk: str
    score: float
    id: str
    chunk_id: int
    timestamp: int
    tag: List[str]
    type: str
    embedding: Optional[List[float]] = None