from contextlib import asynccontextmanager
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import settings
from app.services.embedder import embedder
from app.services.vector_store import chat_store, semantic_store
from app.routers import embed, chat, semantic

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    embedder.load_models()
    
    # Ensure indices exist
    await chat_store.create_index()
    await semantic_store.create_index()
    
    yield
    
    await chat_store.close()
    await semantic_store.close()

app = FastAPI(title="RAG & Embedding API", version="1.0.0", lifespan=lifespan)

# Security
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if key == settings.api_key.get_secret_value():
        return key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API Key"
    )

# Include Routers (Protected)
app.include_router(embed.router, dependencies=[Security(verify_api_key)])
app.include_router(chat.router, dependencies=[Security(verify_api_key)])
app.include_router(semantic.router, dependencies=[Security(verify_api_key)])

@app.get("/health")
def health():
    return {"status": "ok"}
