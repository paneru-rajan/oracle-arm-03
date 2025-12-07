from typing import List
from fastapi import APIRouter
from schemas.article import SemanticIndexRequest, SemanticSearchRequest, SemanticSearchResult
from services.embedder import embedder
from services.vector_store import semantic_store
from config import settings
from dateutil import parser

router = APIRouter(prefix="/semantic", tags=["Semantic Search"])

@router.post("/index")
async def index_semantic(req: SemanticIndexRequest):
    model_name = settings.default_model_type
    
    # Embed the chunk
    embeddings = await embedder.embed([req.chunk], model_name)
    
    await semantic_store.index(
        chunk=req.chunk,
        vector=embeddings[0],
        id=req.id,
        chunk_id=req.chunk_id,
        timestamp=req.timestamp,
        tag=req.tag,
        type=req.type
    )
    return {"status": "indexed", "id": f"{req.id}_{req.chunk_id}"}

@router.post("/search", response_model=List[SemanticSearchResult])
async def search_semantic(req: SemanticSearchRequest):
    model_name = settings.default_model_type
    config = settings.models.get(model_name)
    
    # Embed Query
    prompt = config.query_instruction_template.format(task="Retrieve relevant articles")
    embeddings = await embedder.embed([req.query], model_name, prompt=prompt)
    
    # 1. Range Filters (Timestamp)
    range_filters = {}
    if req.date_from or req.date_to:
        ts_range = {}
        if req.date_from:
            dt = parser.parse(req.date_from)
            ts_range["gte"] = int(dt.timestamp() * 1000)
        if req.date_to:
            dt = parser.parse(req.date_to)
            ts_range["lte"] = int(dt.timestamp() * 1000)
        range_filters["timestamp"] = ts_range

    # 2. Terms Filters
    terms_filters = {}
    if req.tags:
        terms_filters["tag"] = req.tags
    if req.types:
        terms_filters["type"] = req.types
    if req.ids:
        terms_filters["id"] = req.ids
        
    results = await semantic_store.search(
        vector=embeddings[0],
        top_k=req.size,
        range_filters=range_filters,
        terms_filters=terms_filters,
        include_embeddings=req.include_embeddings
    )
    
    mapped_results = []
    for res in results:
        mapped_results.append(SemanticSearchResult(
            chunk=res["text"],
            score=res["score"],
            id=res["id"],
            chunk_id=res["chunk_id"],
            timestamp=res["timestamp"],
            tag=res["tag"],
            type=res["type"],
            embedding=res["embedding"]
        ))
        
    return mapped_results