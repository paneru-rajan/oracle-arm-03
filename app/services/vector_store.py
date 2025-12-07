import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from elasticsearch import AsyncElasticsearch, helpers
from config import settings

logger = logging.getLogger(__name__)

class BaseVectorStore(ABC):
    def __init__(self, index_name: str):
        self.index_name = index_name
        logger.info(f"Connecting to Elasticsearch at {settings.es_host}")
        self.client = AsyncElasticsearch(
            settings.es_host,
            basic_auth=(settings.es_username, settings.es_password)
        )

    async def close(self):
        await self.client.close()

    @abstractmethod
    async def create_index(self, dim: Optional[int] = None):
        pass

class ChatVectorStore(BaseVectorStore):
    async def create_index(self, dim: Optional[int] = None):
        if dim is None:
            dim = settings.models[settings.default_model_type].embedding_dim

        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info(f"Creating Chat index: {self.index_name} with dim {dim}")
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "message_id": {"type": "keyword"},
                        "question": {"type": "text", "index": False},
                        "answer": {"type": "text", "index": False},
                        "guest_id": {"type": "keyword"},
                        "property_id": {"type": "keyword"},
                        "host_id": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)
            
    async def get_latest_timestamp(self) -> int:
        """
        Retrieves the latest 'updated_at' timestamp from the index.
        Returns 0 if the index is empty or doesn't exist.
        """
        try:
            if not await self.client.indices.exists(index=self.index_name):
                return 0
                
            resp = await self.client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "max_timestamp": {"max": {"field": "updated_at"}}
                    }
                }
            )
            max_ts = resp["aggregations"]["max_timestamp"]["value"]
            return int(max_ts) if max_ts else 0
        except Exception as e:
            logger.error(f"Error fetching latest timestamp: {e}")
            return 0

    async def bulk_index(self, docs: List[Dict[str, Any]]):
        """
        Bulk indexes a list of documents.
        docs: List of dicts containing the document fields.
        Expected to have '_id' key for specific doc ID.
        """
        actions = [
            {
                "_index": self.index_name,
                "_id": doc.pop("_id", None),
                "_source": doc
            }
            for doc in docs
        ]
        try:
            success, failed = await helpers.async_bulk(self.client, actions, refresh=True)
            logger.info(f"Bulk indexed {success} documents. Failed: {failed}")
            return success
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise e

    async def index(self, vector: list, question: str, answer: str, **kwargs):
        doc = {
            "embedding": vector,
            "question": question,
            "answer": answer,
            **kwargs
        }
        doc_id = kwargs.get("message_id")
        await self.client.index(index=self.index_name, document=doc, id=doc_id)

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

        if filters:
            for key, value in filters.items():
                if value is not None:
                    filter_clauses.append({"term": {key: value}})
        
        if range_filters:
            for key, ranges in range_filters.items():
                filter_clauses.append({"range": {key: ranges}})

        if filter_clauses:
             knn_query["filter"] = filter_clauses if len(filter_clauses) > 1 else filter_clauses[0]

        query = {
            "knn": knn_query,
            "_source": True
        }
        
        resp = await self.client.search(index=self.index_name, body=query)
        
        results = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            source["score"] = hit["_score"]
            results.append(source)
        return results

class SemanticVectorStore(BaseVectorStore):
    async def create_index(self, dim: Optional[int] = None):
        if dim is None:
            dim = settings.models[settings.default_model_type].embedding_dim

        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info(f"Creating Semantic index: {self.index_name} with dim {dim}")
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
                        "id": {"type": "keyword"},
                        "chunk_id": {"type": "integer"},
                        "tag": {"type": "keyword"},
                        "type": {"type": "keyword"}
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)

    async def index(self, chunk: str, vector: list, id: str, chunk_id: int, 
                   timestamp: int, tag: List[str], type: str):
        
        doc_id = f"{id}_{chunk_id}"
        doc = {
            "text": chunk,
            "embedding": vector,
            "id": id,
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "tag": tag,
            "type": type
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

        fields = ["text", "id", "chunk_id", "timestamp", "tag", "type"]
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
                "id": source.get("id"),
                "chunk_id": source.get("chunk_id"),
                "timestamp": source.get("timestamp"),
                "tag": source.get("tag", []),
                "type": source.get("type"),
                "embedding": source.get("embedding") if include_embeddings else None
            })
        return results

# Instances
chat_store = ChatVectorStore(index_name=settings.chat_index)
semantic_store = SemanticVectorStore(index_name=settings.semantic_index)