from functools import lru_cache
from urllib.parse import quote

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="OmniAgent Flow", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    webhook_verify_token: str = Field(
        default="change-me",
        alias="WEBHOOK_VERIFY_TOKEN",
    )
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        alias="CELERY_BROKER_URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        alias="CELERY_RESULT_BACKEND",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "OPENAI-API-KEY",
            "openai-api-key",
            "openai_api_key",
        ),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "OPENAI_MODEL",
            "OPENAI-MODEL",
            "openai-model",
            "openai_model",
        ),
    )
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        default="omniagent_knowledge",
        alias="QDRANT_COLLECTION",
    )
    rag_embedding_size: int = Field(default=384, alias="RAG_EMBEDDING_SIZE")
    rag_dense_weight: float = Field(default=0.65, alias="RAG_DENSE_WEIGHT")
    rag_bm25_weight: float = Field(default=0.35, alias="RAG_BM25_WEIGHT")
    rag_candidate_limit: int = Field(default=20, alias="RAG_CANDIDATE_LIMIT")
    rag_bm25_corpus_limit: int = Field(default=500, alias="RAG_BM25_CORPUS_LIMIT")
    rag_enable_reranker: bool = Field(default=False, alias="RAG_ENABLE_RERANKER")
    rag_reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        alias="RAG_RERANKER_MODEL",
    )
    hubspot_access_token: SecretStr | None = Field(
        default=None,
        alias="HUBSPOT_ACCESS_TOKEN",
    )
    hubspot_base_url: str = Field(
        default="https://api.hubapi.com",
        alias="HUBSPOT_BASE_URL",
    )
    hubspot_sync_enabled: bool = Field(default=False, alias="HUBSPOT_SYNC_ENABLED")
    hubspot_timeout_seconds: float = Field(default=10.0, alias="HUBSPOT_TIMEOUT_SECONDS")
    postgres_dsn_override: SecretStr | None = Field(
        default=None,
        alias="POSTGRES_DSN",
    )
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str | None = Field(default=None, alias="POSTGRES_USER")
    postgres_password: SecretStr | None = Field(
        default=None,
        alias="POSTGRES_PASSWORD",
    )
    postgres_db: str | None = Field(default=None, alias="POSTGRES_DB")
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS")

    @model_validator(mode="after")
    def validate_postgres_configuration(self) -> "Settings":
        if self.postgres_dsn_override:
            return self

        missing_variables: list[str] = []
        if not self.postgres_user or not self.postgres_user.strip():
            missing_variables.append("POSTGRES_USER")
        if (
            self.postgres_password is None
            or not self.postgres_password.get_secret_value()
        ):
            missing_variables.append("POSTGRES_PASSWORD")
        if not self.postgres_db or not self.postgres_db.strip():
            missing_variables.append("POSTGRES_DB")

        if missing_variables:
            missing = ", ".join(missing_variables)
            raise ValueError(f"Missing PostgreSQL configuration: {missing}")

        return self

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def openai_api_key_value(self) -> str | None:
        if self.openai_api_key is None:
            return None

        api_key = self.openai_api_key.get_secret_value().strip()
        return api_key or None

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def hubspot_access_token_value(self) -> str | None:
        if self.hubspot_access_token is None:
            return None

        access_token = self.hubspot_access_token.get_secret_value().strip()
        return access_token or None

    @property
    def postgres_dsn(self) -> str:
        if self.postgres_dsn_override:
            return self.postgres_dsn_override.get_secret_value()

        if (
            self.postgres_user is None
            or self.postgres_password is None
            or self.postgres_db is None
        ):
            raise RuntimeError("PostgreSQL configuration has not been validated")

        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        database = quote(self.postgres_db, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{database}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
