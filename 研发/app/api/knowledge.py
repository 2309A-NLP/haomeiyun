from __future__ import annotations

import base64
import importlib.util
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..models.database import KnowledgeDocument, get_db
from ..models.schemas import KnowledgeDocumentResponse, KnowledgeQueryRequest, KnowledgeUploadJsonRequest
from ..services.document_ingestion import DocumentIngestionService, SUPPORTED_EXTENSIONS
from ..utils.logger import logger
from ..vector_store.milvus_client import MilvusClient

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
PROJECT_DIR = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_DIR / "data" / "knowledge_uploads"
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None


if MULTIPART_AVAILABLE:
    @router.post("/upload-file")
    async def upload_document_file(
        title: str = Form(...),
        doc_type: str = Form(...),
        legal_field: str = Form(...),
        source: str | None = Form(None),
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
        try:
            file_bytes = await file.read()
        finally:
            await file.close()

        return await _ingest_uploaded_bytes(
            db=db,
            title=title,
            doc_type=doc_type,
            legal_field=legal_field,
            source=source,
            filename=file.filename or "upload.bin",
            file_bytes=file_bytes,
        )
else:
    @router.post("/upload-file")
    async def upload_document_file_unavailable():
        raise HTTPException(
            status_code=503,
            detail=(
                'File upload requires "python-multipart" in the current Python environment. '
                'Please install project dependencies or start the project with .venv.'
            ),
        )


@router.post("/upload-file-json")
async def upload_document_file_json(
    request: KnowledgeUploadJsonRequest,
    db: Session = Depends(get_db),
):
    try:
        file_bytes = base64.b64decode(request.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Base64 file content") from exc

    return await _ingest_uploaded_bytes(
        db=db,
        title=request.title,
        doc_type=request.doc_type,
        legal_field=request.legal_field,
        source=request.source,
        filename=request.filename,
        file_bytes=file_bytes,
    )


@router.post("/query")
async def query_knowledge(
    request: KnowledgeQueryRequest,
    milvus: MilvusClient = Depends(),
):
    try:
        legal_field = request.legal_field.value if request.legal_field else None
        results = milvus.bm25_search(
            query=request.query,
            top_k=request.top_k,
            legal_field=legal_field,
            knowledge_type="law",
        )
        return {"results": results}
    except Exception as exc:
        logger.error("Knowledge query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents", response_model=list[KnowledgeDocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return (
        db.query(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    milvus: MilvusClient = Depends(),
):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        if doc.title:
            safe_title = doc.title.replace('"', '\\"')
            milvus.delete_by_filter(f'document_title == "{safe_title}"')
    except Exception as exc:
        logger.warning("Failed to delete document from Milvus: %s", exc)

    try:
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except OSError as exc:
        logger.warning("Failed to remove local file %s: %s", doc.file_path, exc)

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}


def _normalize_upload_title(title: str | None, filename: str | None, source: str | None) -> str:
    raw_title = (title or "").strip()
    fallback = Path(source or filename or "untitled").stem.strip() or "untitled"
    if not raw_title:
        return fallback
    if raw_title.count("?") >= 3 or "\ufffd" in raw_title:
        return fallback
    return raw_title


async def _ingest_uploaded_bytes(
    *,
    db: Session,
    title: str,
    doc_type: str,
    legal_field: str,
    source: str | None,
    filename: str,
    file_bytes: bytes,
):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_ROOT / f"{uuid.uuid4().hex}{suffix}"
    with saved_path.open("wb") as target:
        target.write(file_bytes)

    service = DocumentIngestionService(db)
    return await service.ingest_uploaded_document(
        title=_normalize_upload_title(title, filename, source),
        doc_type=doc_type.strip() or "general",
        legal_field=legal_field.strip() or "general",
        source=(source or filename or saved_path.name).strip(),
        file_path=saved_path,
        original_filename=filename or saved_path.name,
    )
