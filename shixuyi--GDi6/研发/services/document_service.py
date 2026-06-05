from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from ..core.config import settings
from ..core.logging import logger
from ..ingest.pdf_parser import ProspectusPDFParser
from ..models.schemas import DocumentChunk, DocumentRecord, IngestResponse
from .pdf_processing_service import PDFProcessingService


class DocumentService:
    def __init__(self) -> None:
        self.parser = ProspectusPDFParser()
        self.processor = PDFProcessingService()

    def ingest(self) -> IngestResponse:
        self._ensure_directories()
        if not settings.raw_pdf_path.exists():
            logger.warning("PDF not found, using seed QA knowledge only")
            self._save_chunks([], settings.parsed_chunks_path)
            return IngestResponse(
                source="seed-only",
                chunks=0,
                saved_to=str(settings.parsed_chunks_path),
                used_seed_fallback=True,
            )

        _record, ingest_result = self._ingest_pdf_file(
            source_pdf=settings.raw_pdf_path,
            filename=settings.raw_pdf_path.name,
            document_label=settings.raw_pdf_path.stem,
            copy_to_uploads=False,
        )
        return ingest_result

    def ingest_uploaded_pdf_bytes(
        self,
        filename: str,
        content: bytes,
        document_label: str | None = None,
    ) -> tuple[DocumentRecord, IngestResponse]:
        self._ensure_directories()
        safe_name = Path(filename).name or "uploaded.pdf"
        upload_path = self._unique_upload_path(safe_name)
        upload_path.write_bytes(content)
        return self._ingest_pdf_file(upload_path, safe_name, document_label=document_label, copy_to_uploads=False)

    def save_uploaded_pdf(self, filename: str, file_obj) -> Path:
        self._ensure_directories()
        safe_name = Path(filename).name or "uploaded.pdf"
        target_path = self._unique_upload_path(safe_name)
        with target_path.open("wb") as output:
            shutil.copyfileobj(file_obj, output)
        logger.info("Saved uploaded PDF to %s", target_path)
        return target_path

    def save_uploaded_pdf_bytes(self, filename: str, content: bytes) -> Path:
        self._ensure_directories()
        safe_name = Path(filename).name or "uploaded.pdf"
        target_path = self._unique_upload_path(safe_name)
        target_path.write_bytes(content)
        logger.info("Saved uploaded PDF bytes to %s", target_path)
        return target_path

    def process_pdf_for_ingest(
        self,
        input_pdf: Path | None = None,
        output_pdf: Path | None = None,
        output_text: Path | None = None,
    ) -> Path:
        started = time.perf_counter()
        self._ensure_directories()
        source_pdf = Path(input_pdf or settings.raw_pdf_path)
        target_pdf = Path(output_pdf or settings.processed_pdf_path)
        target_text = Path(output_text or settings.processed_text_path)
        if not source_pdf.exists():
            raise FileNotFoundError(source_pdf)
        logger.info(
            "PDF preprocess started: source_pdf=%s output_pdf=%s output_text=%s",
            source_pdf,
            target_pdf,
            target_text,
        )
        stats = self.processor.process(source_pdf, target_pdf, target_text)
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "PDF preprocess completed: source_pdf=%s pages=%s paragraphs=%s lines=%s latency_ms=%s",
            source_pdf,
            stats.get("pages"),
            stats.get("paragraphs"),
            stats.get("lines"),
            latency_ms,
        )
        return target_pdf

    def get_active_pdf_path(self, prefer_raw: bool = False) -> Path | None:
        documents = self.load_documents()
        if not documents:
            return None
        return self.get_document_pdf_path(documents[-1].document_id, prefer_raw=prefer_raw)

    def get_document_pdf_path(self, document_id: str | None, prefer_raw: bool = False) -> Path | None:
        if not document_id:
            return self.get_active_pdf_path(prefer_raw=prefer_raw) if self.load_documents() else None
        record = self.find_document(document_id=document_id)
        if record is None:
            return None
        raw_path = Path(record.raw_pdf_path)
        processed_path = Path(record.processed_pdf_path)
        if prefer_raw and raw_path.exists():
            return raw_path
        if processed_path.exists():
            return processed_path
        if raw_path.exists():
            return raw_path
        return None

    def load_chunks(self, document_id: str | None = None) -> list[DocumentChunk]:
        if document_id:
            record = self.find_document(document_id=document_id)
            if record is None:
                return []
            chunks_path = Path(record.chunks_path)
            if not chunks_path.exists():
                return []
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            return [DocumentChunk.model_validate(item) for item in raw]

        if not settings.parsed_chunks_path.exists():
            return []
        raw = json.loads(settings.parsed_chunks_path.read_text(encoding="utf-8"))
        return [DocumentChunk.model_validate(item) for item in raw]

    def load_documents(self) -> list[DocumentRecord]:
        if not settings.document_registry_path.exists():
            return []
        raw = json.loads(settings.document_registry_path.read_text(encoding="utf-8"))
        return [DocumentRecord.model_validate(item) for item in raw]

    def find_document(
        self,
        document_id: str | None = None,
        document_label: str | None = None,
        company_name: str | None = None,
    ) -> DocumentRecord | None:
        normalized_id = (document_id or "").strip()
        normalized_label = self._normalize_text_key(document_label)
        normalized_company = self._normalize_text_key(company_name)
        for record in self.load_documents():
            if normalized_id and record.document_id == normalized_id:
                return record
            if normalized_label and self._normalize_text_key(record.document_label) == normalized_label:
                return record
            if normalized_company and self._normalize_text_key(record.company_name) == normalized_company:
                return record
        return None

    def load_seed_qa(self) -> list[dict]:
        if not settings.seed_qa_path.exists():
            return []
        return json.loads(settings.seed_qa_path.read_text(encoding="utf-8"))

    def _ingest_pdf_file(
        self,
        source_pdf: Path,
        filename: str,
        document_label: str | None = None,
        copy_to_uploads: bool = True,
    ) -> tuple[DocumentRecord, IngestResponse]:
        started = time.perf_counter()
        self._ensure_directories()

        safe_name = Path(filename).name or "uploaded.pdf"
        requested_label = (document_label or Path(safe_name).stem).strip() or Path(safe_name).stem
        document_id = self._build_document_id(requested_label)
        document_dir = settings.processed_docs_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)

        if copy_to_uploads:
            upload_path = self._unique_upload_path(safe_name)
            shutil.copyfile(source_pdf, upload_path)
            source_pdf = upload_path

        raw_pdf_path = document_dir / "source.pdf"
        processed_pdf_path = document_dir / "cleaned.pdf"
        processed_text_path = document_dir / "cleaned.txt"
        chunks_path = document_dir / "chunks.json"

        shutil.copyfile(source_pdf, raw_pdf_path)
        self.process_pdf_for_ingest(raw_pdf_path, processed_pdf_path, processed_text_path)

        parse_source = processed_text_path if processed_text_path.exists() else raw_pdf_path
        paragraphs = self.parser.parse(parse_source, requested_label)
        chunks = self.parser.chunk_pages(paragraphs, settings.chunk_size, settings.chunk_overlap)
        company_name = self._extract_company_name(processed_text_path, fallback=requested_label)
        final_label = company_name or requested_label
        enriched_chunks = self._enrich_chunks(
            chunks=chunks,
            document_id=document_id,
            document_label=final_label,
            company_name=company_name,
            filename=safe_name,
        )
        self._save_chunks(enriched_chunks, chunks_path)

        record = DocumentRecord(
            document_id=document_id,
            document_label=final_label,
            company_name=company_name,
            filename=safe_name,
            raw_pdf_path=str(raw_pdf_path),
            processed_pdf_path=str(processed_pdf_path),
            processed_text_path=str(processed_text_path),
            chunks_path=str(chunks_path),
            uploaded_at=time.time(),
            chunk_count=len(enriched_chunks),
        )
        self._upsert_document_record(record)
        self._rebuild_combined_chunks()

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Document ingest completed: document_id=%s label=%s company=%s chunks=%s latency_ms=%s",
            record.document_id,
            record.document_label,
            record.company_name,
            len(enriched_chunks),
            latency_ms,
        )
        return (
            record,
            IngestResponse(
                source=str(processed_text_path),
                chunks=len(enriched_chunks),
                saved_to=str(settings.parsed_chunks_path),
                used_seed_fallback=False,
            ),
        )

    def _enrich_chunks(
        self,
        chunks: list[DocumentChunk],
        document_id: str,
        document_label: str,
        company_name: str,
        filename: str,
    ) -> list[DocumentChunk]:
        enriched: list[DocumentChunk] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "document_id": document_id,
                    "document_label": document_label,
                    "company_name": company_name,
                    "filename": filename,
                }
            )
            enriched.append(
                chunk.model_copy(
                    update={
                        "chunk_id": f"{document_id}:{chunk.chunk_id}",
                        "source": document_label,
                        "title": document_label,
                        "document_id": document_id,
                        "document_label": document_label,
                        "company_name": company_name,
                        "metadata": metadata,
                    }
                )
            )
        return enriched

    def _extract_company_name(self, processed_text_path: Path, fallback: str = "") -> str:
        if not processed_text_path.exists():
            return fallback
        text = processed_text_path.read_text(encoding="utf-8", errors="ignore")
        sample = "\n".join(text.splitlines()[:220])
        patterns = (
            r"公司名称[:：]\s*([\u4e00-\u9fa5A-Za-z0-9()（）·\-]{4,80}?(?:股份有限公司|有限责任公司|有限公司))",
            r"中文名称[:：]\s*([\u4e00-\u9fa5A-Za-z0-9()（）·\-]{4,80}?(?:股份有限公司|有限责任公司|有限公司))",
            r"^([\u4e00-\u9fa5A-Za-z0-9()（）·\-]{4,80}?(?:股份有限公司|有限责任公司|有限公司))$",
        )
        for pattern in patterns:
            match = next(iter(re.finditer(pattern, sample, re.MULTILINE)), None)
            if match:
                return str(match.group(1)).strip()

        candidates = re.findall(
            r"([\u4e00-\u9fa5A-Za-z0-9()（）·\-]{4,80}?(?:股份有限公司|有限责任公司|有限公司))",
            text,
        )
        filtered = [
            item.strip()
            for item in candidates
            if len(item.strip()) >= 8 and "招股说明书" not in item and "关于" not in item[:2]
        ]
        if filtered:
            counts = Counter(filtered)
            best = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0][0]
            return best

        if "兴图新科" in sample:
            return "武汉兴图新科电子股份有限公司"
        if "力源信息" in sample:
            return "武汉力源信息技术股份有限公司"
        return fallback

    def _rebuild_combined_chunks(self) -> None:
        all_chunks: list[DocumentChunk] = []
        for record in self.load_documents():
            chunks_path = Path(record.chunks_path)
            if not chunks_path.exists():
                continue
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            all_chunks.extend(DocumentChunk.model_validate(item) for item in raw)
        self._save_chunks(all_chunks, settings.parsed_chunks_path)
        logger.info("Rebuilt combined chunks index: documents=%s chunks=%s", len(self.load_documents()), len(all_chunks))

    def _upsert_document_record(self, record: DocumentRecord) -> None:
        records = self.load_documents()
        updated: list[DocumentRecord] = []
        replaced = False
        for item in records:
            if item.document_id == record.document_id:
                updated.append(record)
                replaced = True
            else:
                updated.append(item)
        if not replaced:
            updated.append(record)
        payload = [item.model_dump() for item in sorted(updated, key=lambda item: item.uploaded_at)]
        settings.document_registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_chunks(self, chunks: list[DocumentChunk], save_path: Path) -> None:
        payload = [chunk.model_dump() for chunk in chunks]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_directories(self) -> None:
        required_dirs = (
            settings.data_dir,
            settings.uploads_dir,
            settings.raw_pdf_path.parent,
            settings.processed_pdf_path.parent,
            settings.processed_text_path.parent,
            settings.processed_docs_dir,
            settings.parsed_chunks_path.parent,
            settings.seed_qa_path.parent,
            settings.document_registry_path.parent,
        )
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        if not settings.document_registry_path.exists():
            settings.document_registry_path.write_text("[]\n", encoding="utf-8")

    def _unique_upload_path(self, filename: str) -> Path:
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".pdf"
        timestamp = int(time.time() * 1000)
        return settings.uploads_dir / f"{stem}-{timestamp}{suffix}"

    def _build_document_id(self, label: str) -> str:
        normalized = self._normalize_text_key(label) or "document"
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"doc-{digest}"

    def _normalize_text_key(self, value: str | None) -> str:
        return "".join(
            ch
            for ch in (value or "").strip()
            if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum()
        ).lower()
