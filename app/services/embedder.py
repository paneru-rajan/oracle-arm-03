import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool
from config import settings

logger = logging.getLogger(__name__)

class EmbedderService:
    def __init__(self):
        self.models = {}
    
    def load_models(self):
        for name, config in settings.models.items():
            if config.enabled:
                logger.info(f"Loading model: {name} ({config.repo_id})...")
                try:
                    model = SentenceTransformer(
                        config.repo_id, 
                        trust_remote_code=config.trust_remote_code
                    )
                    if config.max_seq_length:
                        model.max_seq_length = config.max_seq_length
                    self.models[name] = model
                    logger.info(f"Model {name} loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load model {name}: {e}")
                    if name == settings.default_model_type:
                        raise RuntimeError(f"Critical: Failed to load default model '{name}'") from e

    def _encode_sync(self, model_name: str, texts: List[str], prompt: Optional[str] = None) -> List[List[float]]:
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not loaded")
        
        embeddings = model.encode(texts, prompt=prompt, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed(self, texts: List[str], model_name: str, prompt: Optional[str] = None) -> List[List[float]]:
        return await run_in_threadpool(self._encode_sync, model_name, texts, prompt)

embedder = EmbedderService()
