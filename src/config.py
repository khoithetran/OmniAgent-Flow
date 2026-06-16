from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env.

    Only the variables that the slimmed-down stack needs are kept:

    - ``REDIS_*`` for chat history and pending_crawl markers
    - ``SESSION_TTL_SECONDS`` for chat history TTL
    - ``PENDING_CRAWL_TTL_SECONDS`` for the entity-detection marker
    - ``OPENAI_*`` for embeddings and chat completions
    - ``QDRANT_*`` for the vector store
    - ``RAG_*`` for chunk size, top-k, and embedding dimensionality
    - ``TELEGRAM_*`` for the bot token and webhook secret
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_name: str = Field(default="OmniAgent Flow", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS")
    pending_crawl_ttl_seconds: int = Field(
        default=300, alias="PENDING_CRAWL_TTL_SECONDS"
    )

    # LLM response cache
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        default="omniagent_knowledge", alias="QDRANT_COLLECTION"
    )

    # RAG
    rag_embedding_size: int = Field(default=1536, alias="RAG_EMBEDDING_SIZE")
    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")

    # Telegram
    telegram_bot_token: SecretStr | None = Field(
        default=None, alias="TELEGRAM_BOT_TOKEN"
    )
    telegram_webhook_secret_token: SecretStr | None = Field(
        default=None, alias="TELEGRAM_WEBHOOK_SECRET_TOKEN"
    )
    telegram_timeout_seconds: float = Field(
        default=10.0, alias="TELEGRAM_TIMEOUT_SECONDS"
    )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def openai_api_key_value(self) -> str | None:
        if self.openai_api_key is None:
            return None
        value = self.openai_api_key.get_secret_value().strip()
        return value or None

    @property
    def telegram_bot_token_value(self) -> str | None:
        if self.telegram_bot_token is None:
            return None
        value = self.telegram_bot_token.get_secret_value().strip()
        return value or None

    @property
    def telegram_webhook_secret_token_value(self) -> str | None:
        if self.telegram_webhook_secret_token is None:
            return None
        value = self.telegram_webhook_secret_token.get_secret_value().strip()
        return value or None

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
