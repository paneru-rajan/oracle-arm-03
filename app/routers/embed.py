from fastapi import APIRouter, HTTPException
from app.schemas.common import EmbedRequest, EmbedResponse
from app.services.embedder import embedder
from app.config import settings

router = APIRouter(tags=["Embedding"])

@router.post("/embed", response_model=EmbedResponse)
async def embed_documents(req: EmbedRequest):
    model_name = settings.default_model_type
    embeddings = await embedder.embed(req.texts, model_name)
    return EmbedResponse(embeddings=embeddings, model=model_name)

@router.post("/query", response_model=EmbedResponse)
async def embed_query(req: EmbedRequest):
    model_name = settings.default_model_type
    config = settings.models.get(model_name)
    
    task = req.task or config.default_task
    prompt = config.query_instruction_template.format(task=task)
    
    embeddings = await embedder.embed(req.texts, model_name, prompt=prompt)
    return EmbedResponse(embeddings=embeddings, model=model_name)