from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config import settings
from services.embedder import embedder
from services.vector_store import chat_store, semantic_store
from services.sync_to_embed import start_scheduler
from routers import embed, chat, semantic

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.load_models()
    
    await chat_store.create_index()
    await semantic_store.create_index()
    
    sync_task = asyncio.create_task(start_scheduler())
    
    yield
    
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    
    await chat_store.close()
    await semantic_store.close()

app = FastAPI(title="RAG & Embedding API", version="1.0.0", lifespan=lifespan)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if key == settings.api_key.get_secret_value():
        return key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API Key"
    )

app.include_router(embed.router, dependencies=[Security(verify_api_key)])
app.include_router(chat.router, dependencies=[Security(verify_api_key)])
app.include_router(semantic.router, dependencies=[Security(verify_api_key)])

@app.get("/health")
def health():
    return {"status": "ok"}
