from typing import List
from fastapi import APIRouter
from app.schemas.chat import ChatIndexRequest, ChatSearchRequest, ChatSearchResult
from app.services.embedder import embedder
from app.services.vector_store import chat_store
from app.config import settings
from dateutil import parser

router = APIRouter(prefix="/chat", tags=["Chat Memory"])

@router.post("/index")
async def index_chat(req: ChatIndexRequest):
    # Always use default model
    model_name = settings.default_model_type
    
    # Concatenate Question and Answer
    combined_text = f"Q: {req.question}\nA: {req.answer}"
    
    # Embed
    embeddings = await embedder.embed([combined_text], model_name)
    
    await chat_store.index(
        text=combined_text,
        vector=embeddings[0],
        question=req.question,
        answer=req.answer,
        inbox_id=req.inbox_id,
        property_id=req.property_id,
        user_id=req.user_id,
        timestamp=req.timestamp
    )
    return {"status": "indexed", "inbox_id": req.inbox_id}

@router.post("/search", response_model=List[ChatSearchResult])
async def search_chat(req: ChatSearchRequest):
    model_name = settings.default_model_type
    config = settings.models.get(model_name)
    
    # Embed Query
    prompt = config.query_instruction_template.format(task="Retrieve similar past Q&A")
    embeddings = await embedder.embed([req.query], model_name, prompt=prompt)
    
    # 1. Exact Match Filters
    filters = {}
    if req.inbox_id:
        filters["inbox_id"] = req.inbox_id
    if req.property_id:
        filters["property_id"] = req.property_id
    if req.user_id:
        filters["user_id"] = req.user_id
        
    # 2. Range Filters (Timestamp)
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
    
    results = await chat_store.search(
        vector=embeddings[0],
        top_k=req.size,
        filters=filters,
        range_filters=range_filters
    )
    
    # Map results
    mapped_results = []
    for res in results:
        mapped_results.append(ChatSearchResult(
            text=res["text"],
            score=res["score"],
            question=res["question"],
            answer=res["answer"],
            inbox_id=res["inbox_id"],
            property_id=res["property_id"],
            user_id=res["user_id"],
            timestamp=res["timestamp"]
        ))
        
    return mapped_results