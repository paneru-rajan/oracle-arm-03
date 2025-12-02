from typing import Dict
from pydantic import BaseModel, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelConfig(BaseModel):
    repo_id: str
    max_seq_length: int = 1024
    enabled: bool = True
    trust_remote_code: bool = False
    # Template for the prompt argument. 
    # The sentence-transformer will prepend this string to the input text.
    # So if template is "Instruct: {task}\nQuery: ", the input becomes "Instruct: {task}\nQuery: {text}"
    query_instruction_template: str 
    default_task: str = "Given a web search query, retrieve relevant passages that answer the query"

class Settings(BaseSettings):
    default_model_type: str = "qwen"
    api_key: SecretStr
    
    models: Dict[str, ModelConfig] = {
        "qwen": ModelConfig(
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            query_instruction_template="Instruct: {task}\nQuery: ",
            max_seq_length=1024,
            trust_remote_code=False
        ),
        "gemma": ModelConfig(
            repo_id="tencent/KaLM-Embedding-Gemma3-12B-2511",
            query_instruction_template="Instruct: {task}\nQuery: ",
            max_seq_length=1024,
            trust_remote_code=True
        )
    }

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    @model_validator(mode='after')
    def check_default_model_exists(self) -> 'Settings':
        if self.default_model_type not in self.models:
            raise ValueError(f"Default model '{self.default_model_type}' not found in configured models: {list(self.models.keys())}")
        return self

settings = Settings()
