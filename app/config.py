from typing import Dict, Optional
from pydantic import BaseModel, model_validator, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelConfig(BaseModel):
    repo_id: str
    max_seq_length: int = 1024
    embedding_dim: int = 1024
    enabled: bool = True
    trust_remote_code: bool = False
    query_instruction_template: str 
    default_task: str = "Given a web search query, retrieve relevant passages that answer the query"

class Settings(BaseSettings):
    # Security
    api_key: SecretStr = Field(alias="API_KEY")
    
    # ElasticSearch
    es_host: str = Field(default="http://es:9200", alias="ELASTICSEARCH_HOST")
    es_username: str = Field(default="elastic", alias="ELASTICSEARCH_USERNAME")
    es_password: str = Field(default="changeme", alias="ELASTIC_PASSWORD")
    
    # Indices
    chat_index: str = "chat-index"
    semantic_index: str = "semantic-index"

    # Database & Sync
    database_url: str = Field(alias="DATABASE_URL", default="")
    sync_interval_minutes: int = Field(alias="SYNC_INTERVAL_MINUTES", default=60)

    # Models
    default_model_type: str = "qwen"
    models: Dict[str, ModelConfig] = {
        "qwen": ModelConfig(
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            query_instruction_template="Instruct: {task}\nQuery: ",
            max_seq_length=1024,
            trust_remote_code=False
        )
    }

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    @model_validator(mode='after')
    def check_default_model_exists(self) -> 'Settings':
        if self.default_model_type not in self.models:
            raise ValueError(f"Default model '{self.default_model_type}' not found in configured models: {list(self.models.keys())}")
        return self

settings = Settings()
