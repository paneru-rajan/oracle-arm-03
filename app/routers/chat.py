from typing import List
from fastapi import APIRouter
from schemas.chat import ChatIndexRequest, ChatSearchRequest, ChatSearchResult
from services.embedder import embedder
from services.vector_store import chat_store
from config import settings
from dateutil import parser

router = APIRouter(prefix="/chat", tags=["Chat Memory"])

@router.post("/index")
async def index_chat(req: ChatIndexRequest):
    model_name = settings.default_model_type
    combined_text = f"Q: {req.question}\nA: {req.answer}"
    
    embeddings = await embedder.embed([combined_text], model_name)
    
    await chat_store.index(
        vector=embeddings[0],
        question=req.question,
        answer=req.answer,
        property_id=req.property_id,
        message_id=req.message_id,
        host_id=req.host_id,
        guest_id=req.guest_id,
        category=req.category,
        created_at=req.created_at,
        updated_at=req.updated_at
    )
    return {"status": "indexed", "message_id": req.message_id}

@router.post("/search", response_model=List[ChatSearchResult], response_model_exclude={"created_at", "guest_id", "host_id", "message_id", "property_id"})
async def search_chat(req: ChatSearchRequest):
    model_name = settings.default_model_type
    config = settings.models.get(model_name)
    
    prompt = config.query_instruction_template.format(task="Retrieve similar past Q&A")
    embeddings = await embedder.embed([req.query], model_name, prompt=prompt)
    
    filters = {
        k: v for k, v in {
            "property_id": req.property_id,
            "host_id": req.host_id,
            "guest_id": req.guest_id,
            "category": req.category
        }.items() if v
    }
        
    range_filters = {}
    if req.date_from or req.date_to:
        date_range = {}
        if req.date_from:
            date_range["gte"] = parser.parse(req.date_from).isoformat()
        if req.date_to:
            date_range["lte"] = parser.parse(req.date_to).isoformat()
        range_filters["created_at"] = date_range
    
    results = await chat_store.search(
        vector=embeddings[0],
        top_k=req.size,
        filters=filters,
        range_filters=range_filters
    )
    
    return [
        ChatSearchResult(
            score=res.get("score", 0.0),
            question=res.get("question", ""),
            answer=res.get("answer", ""),
            property_id=res.get("property_id", ""),
            message_id=res.get("message_id"),
            host_id=res.get("host_id"),
            guest_id=res.get("guest_id"),
            category=res.get("category"),
            created_at=res.get("created_at"),
            updated_at=res.get("updated_at")
        ) for res in results
    ]