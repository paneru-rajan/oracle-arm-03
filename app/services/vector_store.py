import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from elasticsearch import AsyncElasticsearch
from app.config import settings

logger = logging.getLogger(__name__)

class BaseVectorStore(ABC):
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.client = AsyncElasticsearch(
            settings.es_host,
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=False
        )

    async def close(self):
        await self.client.close()

    @abstractmethod
    async def create_index(self, dim: int = 1024):
        pass

class ChatVectorStore(BaseVectorStore):
    async def create_index(self, dim: int = 1024):
        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info(f"Creating Chat index: {self.index_name}")
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        # Strict schema for Chat
                        "timestamp": {"type": "date", "format": "epoch_millis"},
                        "inbox_id": {"type": "keyword"},
                        "property_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "question": {"type": "text"}, # Indexed for text search too?
                        "answer": {"type": "text"}
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)

    async def index(self, text: str, vector: list, question: str, answer: str, 
                   inbox_id: str, property_id: str, user_id: str, timestamp: int):
        doc = {
            "text": text,
            "embedding": vector,
            "question": question,
            "answer": answer,
            "inbox_id": inbox_id,
            "property_id": property_id,
            "user_id": user_id,
            "timestamp": timestamp
        }
        # ID strategy: Maybe inbox_id + timestamp? Or auto-generate.
        # Let's auto-generate for now unless user specifies ID requirement for Chat.
        await self.client.index(index=self.index_name, document=doc)

    async def search(self, vector: list, top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None,
                    range_filters: Optional[Dict[str, Dict[str, Any]]] = None):
        knn_query = {
            "field": "embedding",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(100, top_k * 10)
        }

        filter_clauses = []

        # 1. Exact Match (Filters)
        if filters:
            for key, value in filters.items():
                # Now we target top-level keyword fields
                filter_clauses.append({"term": {key: value}})
        
        # 2. Range Match (Timestamp)
        if range_filters:
            for key, ranges in range_filters.items():
                filter_clauses.append({"range": {key: ranges}})

        if filter_clauses:
             knn_query["filter"] = filter_clauses if len(filter_clauses) > 1 else filter_clauses[0]

        query = {
            "knn": knn_query,
            "_source": ["text", "question", "answer", "inbox_id", "property_id", "user_id", "timestamp"]
        }
        
        resp = await self.client.search(index=self.index_name, body=query)
        
        results = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            results.append({
                "score": hit["_score"],
                "text": source.get("text", ""),
                "question": source.get("question", ""),
                "answer": source.get("answer", ""),
                "inbox_id": source.get("inbox_id", ""),
                "property_id": source.get("property_id", ""),
                "user_id": source.get("user_id", ""),
                "timestamp": source.get("timestamp", 0)
            })
        return results

class SemanticVectorStore(BaseVectorStore):
    async def create_index(self, dim: int = 1024):
        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info(f"Creating Semantic index: {self.index_name}")
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "timestamp": {"type": "date", "format": "epoch_millis"}, 
                        "alias": {"type": "keyword"},
                        "chunk_id": {"type": "integer"},
                        "tag": {"type": "keyword"},
                        "content_type": {"type": "keyword"}
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)

    async def index(self, chunk: str, vector: list, alias: str, chunk_id: int, 
                   timestamp: int, tag: List[str], content_type: str):
        
        doc_id = f"{alias}_{chunk_id}"
        doc = {
            "text": chunk,
            "embedding": vector,
            "alias": alias,
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "tag": tag,
            "content_type": content_type
        }
        await self.client.index(index=self.index_name, document=doc, id=doc_id)

    async def search(self, vector: list, top_k: int = 5, 
                    range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                    terms_filters: Optional[Dict[str, List[Any]]] = None,
                    include_embeddings: bool = False):
        
        knn_query = {
            "field": "embedding",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(100, top_k * 10)
        }

        filter_clauses = []

        if terms_filters:
            for key, values in terms_filters.items():
                if values:
                    filter_clauses.append({"terms": {key: values}})

        if range_filters:
            for key, ranges in range_filters.items():
                filter_clauses.append({"range": {key: ranges}})

        if filter_clauses:
             knn_query["filter"] = filter_clauses if len(filter_clauses) > 1 else filter_clauses[0]

        fields = ["text", "alias", "chunk_id", "timestamp", "tag", "content_type"]
        if include_embeddings:
            fields.append("embedding")

        query = {
            "knn": knn_query,
            "_source": fields
        }
        
        resp = await self.client.search(index=self.index_name, body=query)
        
        results = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            results.append({
                "score": hit["_score"],
                "text": source.get("text", ""),
                "alias": source.get("alias"),
                "chunk_id": source.get("chunk_id"),
                "timestamp": source.get("timestamp"),
                "tag": source.get("tag", []),
                "content_type": source.get("content_type"),
                "embedding": source.get("embedding") if include_embeddings else None
            })
        return results

# Instances
chat_store = ChatVectorStore(index_name=settings.chat_index)
semantic_store = SemanticVectorStore(index_name=settings.semantic_index)