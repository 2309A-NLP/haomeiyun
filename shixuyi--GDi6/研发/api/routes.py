from __future__ import annotations

from functools import lru_cache
from urllib.parse import unquote

from fastapi import APIRouter, Header, HTTPException, Request

from ..core.logging import get_logger
from ..models.schemas import (
    AskRequest,
    AskResponse,
    BenchmarkSummary,
    DocumentOption,
    IngestResponse,
    UploadResponse,
)
from ..services.benchmark_service import BenchmarkService
from ..services.document_service import DocumentService
from ..services.rag_service import ProspectusRAGService

router = APIRouter()
logger = get_logger(__name__)


def _truncate_text(value: str, limit: int = 120) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    return DocumentService()


@lru_cache(maxsize=1)
def get_rag_service() -> ProspectusRAGService:
    return ProspectusRAGService()


@lru_cache(maxsize=1)
def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService()


@router.get("/health")
def health() -> dict:
    rag_service = get_rag_service()
    return {"status": "healthy", "index_ready": rag_service.retriever.ready()}


@router.get("/documents", response_model=list[DocumentOption])
def list_documents() -> list[DocumentOption]:
    document_service = get_document_service()
    records = document_service.load_documents()
    return [
        DocumentOption(
            document_id=item.document_id,
            document_label=item.document_label,
            company_name=item.company_name,
            filename=item.filename,
            chunk_count=item.chunk_count,
            uploaded_at=item.uploaded_at,
        )
        for item in records
    ]


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    document_service = get_document_service()
    rag_service = get_rag_service()
    benchmark_service = get_benchmark_service()
    result = document_service.ingest()
    rag_service.refresh_index()
    benchmark_service.rag.refresh_index()
    return result


@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    request: Request,
    x_filename: str | None = Header(default=None),
    x_document_label: str | None = Header(default=None),
    content_type: str | None = Header(default=None),
) -> UploadResponse:
    document_service = get_document_service()
    rag_service = get_rag_service()
    benchmark_service = get_benchmark_service()

    filename = unquote(x_filename) if x_filename else "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        logger.warning("PDF upload rejected: filename=%s reason=invalid_extension", filename)
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    payload = await request.body()
    if not payload:
        logger.warning("PDF upload rejected: filename=%s reason=empty_payload", filename)
        raise HTTPException(status_code=400, detail="上传内容为空")

    logger.info(
        "PDF upload started: filename=%s content_type=%s size_bytes=%s",
        filename,
        content_type or "application/pdf",
        len(payload),
    )

    try:
        document_record, ingest_result = document_service.ingest_uploaded_pdf_bytes(
            filename=filename,
            content=payload,
            document_label=unquote(x_document_label).strip() if x_document_label else None,
        )
        rag_service.refresh_index()
        benchmark_service.rag.refresh_index()
        uploaded_path = document_service.get_document_pdf_path(document_record.document_id, prefer_raw=True)
        size_bytes = uploaded_path.stat().st_size if uploaded_path and uploaded_path.exists() else 0
        logger.info(
            "PDF upload completed: filename=%s document_id=%s document_label=%s company_name=%s saved_path=%s size_bytes=%s chunks=%s saved_to=%s used_seed_fallback=%s",
            document_record.filename,
            document_record.document_id,
            document_record.document_label,
            document_record.company_name,
            uploaded_path,
            size_bytes,
            ingest_result.chunks,
            ingest_result.saved_to,
            ingest_result.used_seed_fallback,
        )
    except Exception:
        logger.exception("PDF upload failed: filename=%s", filename)
        raise

    upload_message = (
        f"PDF 处理完成：已清洗文本并完成索引构建，当前文档《{document_record.document_label}》"
        f"共生成 {ingest_result.chunks} 个切块，后续提问将优先基于该文档回答。"
    )

    return UploadResponse(
        document_id=document_record.document_id,
        document_label=document_record.document_label,
        company_name=document_record.company_name,
        filename=document_record.filename,
        content_type=content_type or "application/pdf",
        size_bytes=size_bytes,
        message=upload_message,
        ingest=ingest_result,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, raw_request: Request) -> AskResponse:
    rag_service = get_rag_service()
    resolved_session_id = request.session_id or f"client:{raw_request.client.host if raw_request.client else 'local'}"
    effective_request = request if request.session_id else request.model_copy(update={"session_id": resolved_session_id})
    logger.info(
        "Question started: question=%s top_k=%s compare_plain_llm=%s debug_mode=%s session_id=%s",
        _truncate_text(effective_request.question),
        effective_request.top_k,
        effective_request.compare_plain_llm,
        effective_request.debug_mode,
        effective_request.session_id,
    )
    try:
        response = rag_service.ask(effective_request)
    except Exception:
        logger.exception("Question failed: question=%s", _truncate_text(effective_request.question))
        raise

    logger.info(
        "Question completed: question=%s latency_ms=%s source=%s snippets=%s answer=%s session_id=%s",
        _truncate_text(effective_request.question),
        response.latency_ms,
        response.source,
        len(response.related_snippets),
        _truncate_text(response.answer, 180),
        response.session_id,
    )
    return response


@router.get("/benchmark", response_model=BenchmarkSummary)
def benchmark() -> BenchmarkSummary:
    benchmark_service = get_benchmark_service()
    return benchmark_service.run()
