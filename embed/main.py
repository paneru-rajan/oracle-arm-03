import logging
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from embed.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    for name, config in settings.models.items():
        if config.enabled:
            logger.info(f"Loading model: {name} ({config.repo_id})...")
            try:
                model = await run_in_threadpool(
                    SentenceTransformer, 
                    config.repo_id, 
                    trust_remote_code=config.trust_remote_code
                )
                # Set max sequence length if specified
                if config.max_seq_length:
                    model.max_seq_length = config.max_seq_length
                
                models[name] = model
                logger.info(f"Model {name} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                if name == settings.default_model_type:
                    raise RuntimeError(f"Critical: Failed to load default model '{name}'") from e
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

api_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_header)):
    if api_key_header == settings.api_key.get_secret_value():
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

class EmbedRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None
    task: Optional[str] = None 

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str

def _encode_sync(model: SentenceTransformer, texts: List[str], prompt: Optional[str] = None) -> List[List[float]]:
    # normalize_embeddings=True ensures cosine similarity works correctly
    embeddings = model.encode(texts, prompt=prompt, normalize_embeddings=True)
    return embeddings.tolist()

async def get_embeddings(texts: List[str], model_name: str, prompt: Optional[str] = None) -> List[List[float]]:
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found or not loaded.")
    
    model = models[model_name]
    return await run_in_threadpool(_encode_sync, model, texts, prompt)

@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(get_api_key)])
async def embed_documents(req: EmbedRequest):
    """
    Embeds documents. No instruction prefix is added.
    """
    model_name = req.model or settings.default_model_type
    # For documents, we pass no prompt
    embeddings = await get_embeddings(req.texts, model_name)
    return EmbedResponse(embeddings=embeddings, model=model_name)

@app.post("/query", response_model=EmbedResponse, dependencies=[Depends(get_api_key)])
async def embed_query(req: EmbedRequest):
    """
    Embeds queries. Adds the instruction prefix via the 'prompt' argument.
    """
    model_name = req.model or settings.default_model_type
    if model_name not in settings.models:
         raise HTTPException(status_code=400, detail=f"Model '{model_name}' config not found.")
    
    config = settings.models[model_name]
    task_description = req.task or config.default_task
    
    # Construct the prompt string (e.g., "Instruct: ...\nQuery: ")
    # sentence-transformers will prepend this to every text in the batch
    prompt = config.query_instruction_template.format(task=task_description)
    
    embeddings = await get_embeddings(req.texts, model_name, prompt=prompt)
    return EmbedResponse(embeddings=embeddings, model=model_name)

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(models.keys())}
