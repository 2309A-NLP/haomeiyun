from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Prospectus RAG System", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_pdf_path: Path = Field(
        default=Path("./data/raw/武汉兴图新科电子股份有限公司招股说明书.pdf"),
        alias="RAW_PDF_PATH",
    )
    uploads_dir: Path = Field(default=Path("./data/uploads"), alias="UPLOADS_DIR")
    processed_pdf_path: Path = Field(
        default=Path("./data/processed/cleaned_upload.pdf"),
        alias="PROCESSED_PDF_PATH",
    )
    processed_text_path: Path = Field(
        default=Path("./data/processed/cleaned_upload.txt"),
        alias="PROCESSED_TEXT_PATH",
    )
    processed_docs_dir: Path = Field(
        default=Path("./data/processed/documents"),
        alias="PROCESSED_DOCS_DIR",
    )
    document_registry_path: Path = Field(
        default=Path("./data/processed/document_registry.json"),
        alias="DOCUMENT_REGISTRY_PATH",
    )
    parsed_chunks_path: Path = Field(
        default=Path("./data/processed/xingtu_chunks.json"),
        alias="PARSED_CHUNKS_PATH",
    )
    seed_qa_path: Path = Field(
        default=Path("./data/seed/xingtu_manual_qa.json"),
        alias="SEED_QA_PATH",
    )

    vector_backend: str = Field(default="inmemory", alias="VECTOR_BACKEND")
    vector_collection: str = Field(default="xingtu_prospectus", alias="VECTOR_COLLECTION")
    vector_top_k: int = Field(default=12, alias="VECTOR_TOP_K")
    retrieval_rrf_k: int = Field(default=60, alias="RETRIEVAL_RRF_K")
    rerank_top_k: int = Field(default=6, alias="RERANK_TOP_K")
    similarity_threshold: float = Field(default=0.18, alias="SIMILARITY_THRESHOLD")
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_database: str = Field(default="default", alias="MILVUS_DATABASE")
    milvus_consistency_level: str = Field(default="Bounded", alias="MILVUS_CONSISTENCY_LEVEL")
    milvus_dense_weight: float = Field(default=0.55, alias="MILVUS_DENSE_WEIGHT")
    milvus_sparse_weight: float = Field(default=0.45, alias="MILVUS_SPARSE_WEIGHT")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    redis_key_prefix: str = Field(default="prospectus_rag", alias="REDIS_KEY_PREFIX")
    redis_session_ttl_seconds: int = Field(default=86400, alias="REDIS_SESSION_TTL_SECONDS")

    embedding_provider: str = Field(default="bge", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(default="BAAI/bge-large-zh-v1.5", alias="EMBEDDING_MODEL_NAME")
    embedding_model_path: str = Field(default="", alias="EMBEDDING_MODEL_PATH")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")

    rerank_provider: str = Field(default="bge", alias="RERANK_PROVIDER")
    rerank_model_name: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL_NAME")
    rerank_model_path: str = Field(default="", alias="RERANK_MODEL_PATH")

    llm_provider: str = Field(default="disabled", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_timeout_seconds: int = Field(default=12, alias="LLM_TIMEOUT_SECONDS")
    vlm_enabled: bool = Field(default=True, alias="VLM_ENABLED")
    vlm_model: str = Field(default="Qwen/Qwen3-VL-32B-Instruct", alias="VLM_MODEL")
    vlm_base_url: str = Field(default="https://api.siliconflow.cn/v1", alias="VLM_BASE_URL")
    vlm_api_key: str = Field(default="", alias="VLM_API_KEY")
    vlm_timeout_seconds: int = Field(default=20, alias="VLM_TIMEOUT_SECONDS")
    vlm_max_pages: int = Field(default=3, alias="VLM_MAX_PAGES")
    vlm_render_scale: float = Field(default=1.8, alias="VLM_RENDER_SCALE")

    chunk_size: int = Field(default=0, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=0, alias="CHUNK_OVERLAP")
    answer_context_max_chars: int = Field(default=2400, alias="ANSWER_CONTEXT_MAX_CHARS")
    soft_response_target_seconds: float = Field(default=3.0, alias="SOFT_RESPONSE_TARGET_SECONDS")


settings = Settings()
