from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class EmbedRequest(BaseModel):
    texts: List[str]
    task: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str