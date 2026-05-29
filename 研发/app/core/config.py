import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from pydantic import field_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import validator as field_validator
    PYDANTIC_V2 = False

from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent


def _parse_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    truthy = {"1", "true", "yes", "on", "debug", "dev", "development"}
    falsy = {"0", "false", "no", "off", "release", "prod", "production"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    raise ValueError(f"{field_name} must be a boolean-like value, got: {value}")


class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Legal RAG System")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")

    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "legal_system")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)

    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "legal_knowledge")
    MILVUS_INSERT_BATCH_SIZE: int = int(os.getenv("MILVUS_INSERT_BATCH_SIZE", 64))

    MINERU_API_URL: str = os.getenv("MINERU_API_URL", "https://mineru.net")
    MINERU_API_KEY: str = os.getenv("MINERU_API_KEY", "")
    MINERU_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("MINERU_REQUEST_TIMEOUT_SECONDS", 180))
    MINERU_PDF_FALLBACK_MIN_CHARS: int = int(os.getenv("MINERU_PDF_FALLBACK_MIN_CHARS", 1200))
    MINERU_POLL_INTERVAL_SECONDS: int = int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", 2))
    MINERU_LIGHTWEIGHT_MAX_FILE_MB: int = int(os.getenv("MINERU_LIGHTWEIGHT_MAX_FILE_MB", 10))
    PDF_TABLE_TRIGGER_SCORE: int = int(os.getenv("PDF_TABLE_TRIGGER_SCORE", 8))
    PDF_FORMULA_TRIGGER_SCORE: int = int(os.getenv("PDF_FORMULA_TRIGGER_SCORE", 6))
    DOC_TABLE_TRIGGER_SCORE: int = int(os.getenv("DOC_TABLE_TRIGGER_SCORE", 8))
    DOC_FORMULA_TRIGGER_SCORE: int = int(os.getenv("DOC_FORMULA_TRIGGER_SCORE", 6))

    EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "D:/八维/zg3/bge-base-zh-v1.5")
    RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH", "D:/八维/zg3/bge-reranker-base")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K: int = int(os.getenv("TOP_K", 8))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", 4))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    QUERY_MAX_LENGTH: int = int(os.getenv("QUERY_MAX_LENGTH", 128))
    QUERY_REWRITE_MAX_CHARS: int = int(os.getenv("QUERY_REWRITE_MAX_CHARS", 120))
    RAG_FAST_TOP_K: int = int(os.getenv("RAG_FAST_TOP_K", 4))
    RAG_SEARCH_MULTIPLIER: int = int(os.getenv("RAG_SEARCH_MULTIPLIER", 1))
    RAG_ENABLE_RERANK: bool = _parse_bool(os.getenv("RAG_ENABLE_RERANK", "false"), "RAG_ENABLE_RERANK")
    RAG_CONTEXT_MAX_CHARS: int = int(os.getenv("RAG_CONTEXT_MAX_CHARS", 1200))
    CITATION_CONTENT_MAX_CHARS: int = int(os.getenv("CITATION_CONTENT_MAX_CHARS", 1200))
    RAG_RETRIEVAL_TIMEOUT_SECONDS: int = int(os.getenv("RAG_RETRIEVAL_TIMEOUT_SECONDS", 18))

    SHORT_TERM_MAX_LEN: int = int(os.getenv("SHORT_TERM_MAX_LEN", 20))
    LONG_TERM_THRESHOLD: int = int(os.getenv("LONG_TERM_THRESHOLD", 15))
    MEMORY_EXPIRE_DAYS: int = int(os.getenv("MEMORY_EXPIRE_DAYS", 7))

    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
    VLLM_URL: str = os.getenv("VLLM_URL", "http://localhost:8000")
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "Qwen/Qwen2-7B-Instruct")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1800))
    SIMPLE_CHAT_MAX_TOKENS: int = int(os.getenv("SIMPLE_CHAT_MAX_TOKENS", 600))
    LLM_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", 900))

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-bfacca39db984321b92f519351e6b8ec")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    DOUBAO_API_KEY: str = os.getenv("DOUBAO_API_KEY", "")
    DOUBAO_BASE_URL: str = os.getenv("DOUBAO_BASE_URL", "")

    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "deepseek")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "deepseek-chat")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 1))
    DEBUG: bool = os.getenv("DEBUG", "False")
    CHAT_TIMEOUT_SECONDS: int = int(os.getenv("CHAT_TIMEOUT_SECONDS", 55))
    SOFT_RESPONSE_TARGET_SECONDS: int = int(os.getenv("SOFT_RESPONSE_TARGET_SECONDS", 30))
    STREAM_HEARTBEAT_SECONDS: int = int(os.getenv("STREAM_HEARTBEAT_SECONDS", 15))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(ROOT_DIR / "优化" / "logs" / "legal_rag.log"))

    class Config:
        env_file = str(ROOT_DIR / ".env")
        case_sensitive = True
        extra = "ignore"

    if PYDANTIC_V2:
        @field_validator("DEBUG", mode="before")
        @classmethod
        def validate_debug(cls, value):
            return _parse_bool(value, "DEBUG")
    else:
        @field_validator("DEBUG", pre=True)
        def validate_debug(cls, value):
            return _parse_bool(value, "DEBUG")


settings = Settings()
