from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from .api import chat, knowledge, role, user
from .core.config import settings
from .models.database import init_db
from .utils.logger import logger

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Legal RAG System...")
    init_db()
    logger.info("Database initialized")
    yield
    from .rag.pipeline import LegalRAGPipeline
    from .services.llm_service import LLMService

    LegalRAGPipeline().close()
    await LLMService().aclose()
    logger.info("Shutting down Legal RAG System...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(knowledge.router, prefix=settings.API_V1_STR)
app.include_router(role.router, prefix=settings.API_V1_STR)
app.include_router(user.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/system/info")
async def system_info():
    import torch
    from .rag.embedding import BGEEmbedder
    from .rag.rerank import BGEReranker

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    embedder = BGEEmbedder()
    reranker = BGEReranker()

    return {
        "system": {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "debug": settings.DEBUG,
        },
        "llm": {
            "provider": settings.DEFAULT_LLM_PROVIDER,
            "model": settings.DEFAULT_LLM_MODEL,
            "use_local": settings.USE_LOCAL_LLM,
            "local_model": settings.LOCAL_MODEL if settings.USE_LOCAL_LLM else None,
            "vllm_url": settings.VLLM_URL if settings.USE_LOCAL_LLM else None,
            "available_providers": _get_available_llm_providers(),
        },
        "embedding": {
            "model_name": "BGE-M3",
            "model_path": settings.EMBEDDING_MODEL_PATH,
            "dimension": settings.EMBEDDING_DIMENSION,
            "device": device,
        },
        "reranker": {
            "model_name": "BGE-Reranker",
            "model_path": settings.RERANKER_MODEL_PATH,
            "device": device,
        },
        "rag": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K,
            "rerank_top_k": settings.RERANK_TOP_K,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        },
        "memory": {
            "short_term_max": settings.SHORT_TERM_MAX_LEN,
            "long_term_threshold": settings.LONG_TERM_THRESHOLD,
            "expire_days": settings.MEMORY_EXPIRE_DAYS,
        },
        "databases": {
            "mysql": _check_mysql_status(),
            "redis": _check_redis_status(),
            "milvus": _check_milvus_status(),
        },
        "models_loaded": {
            "embedder": bool(embedder),
            "reranker": bool(reranker),
        },
    }


@app.get("/system/roles")
async def list_available_roles():
    from .core.prompts.lawyer_prompts import ROLE_PROMPTS, ROLE_SPECIALTIES

    roles = []
    for role_id, prompt in ROLE_PROMPTS.items():
        roles.append(
            {
                "role_id": role_id,
                "specialties": ROLE_SPECIALTIES.get(role_id, "综合法律领域"),
                "has_prompt": bool(prompt),
                "prompt_preview": prompt[:200] + "..." if prompt else None,
            }
        )

    return {
        "total": len(roles),
        "roles": roles,
    }


def _get_available_llm_providers():
    providers = []
    if settings.DEEPSEEK_API_KEY:
        providers.append("deepseek")
    if settings.OPENAI_API_KEY:
        providers.append("openai")
    if settings.SILICONFLOW_API_KEY:
        providers.append("siliconflow")
    if settings.QWEN_API_KEY:
        providers.append("qwen")
    if settings.DOUBAO_API_KEY:
        providers.append("doubao")
    if settings.USE_LOCAL_LLM:
        providers.append("local")
    return providers


def _check_mysql_status():
    try:
        from .models.database import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "connected"
    except Exception as exc:
        return f"disconnected: {str(exc)[:50]}"


def _check_redis_status():
    try:
        import redis

        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            socket_connect_timeout=2,
        )
        client.ping()
        return "connected"
    except Exception as exc:
        return f"disconnected: {str(exc)[:50]}"


def _check_milvus_status():
    try:
        from pymilvus import connections, utility

        connections.connect(
            alias="check",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=2,
        )
        version = utility.get_server_version(using="check")
        connections.disconnect("check")
        version_text = str(version)
        if version_text.startswith("v"):
            return f"connected ({version_text})"
        return f"connected (v{version_text})"
    except Exception as exc:
        return f"disconnected: {str(exc)[:50]}"


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
    )
