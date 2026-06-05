from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EntityBundle(BaseModel):
    company: str | None = None
    years: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    intent: Literal["fact", "comparison", "definition", "amount", "fallback"] = "fact"
    normalized_query: str
    disambiguation: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: EntityBundle = Field(default_factory=EntityBundle)


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    document_id: str = ""
    document_label: str = ""
    company_name: str = ""
    page: int | None = None
    text: str
    title: str = ""
    keywords: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    document_id: str | None = None
    top_k: int | None = None
    compare_plain_llm: bool = True
    debug_mode: bool = False


class AskDebugInfo(BaseModel):
    forced_multimodal: bool = False
    multimodal_attempted: bool = False
    logical_pages: list[int] = Field(default_factory=list)
    render_pages: list[int] = Field(default_factory=list)
    multimodal_raw_answer: str | None = None


class AskResponse(BaseModel):
    question: str
    session_id: str | None = None
    resolved_question: str | None = None
    analysis: QueryAnalysis
    answer: str
    source: str
    document_id: str | None = None
    document_label: str | None = None
    company_name: str | None = None
    related_snippets: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    latency_ms: int
    pdf_answer: str | None = None
    plain_llm_answer: str | None = None
    debug_info: AskDebugInfo | None = None


class IngestResponse(BaseModel):
    source: str
    chunks: int
    saved_to: str
    used_seed_fallback: bool


class UploadResponse(BaseModel):
    document_id: str
    document_label: str
    company_name: str
    filename: str
    content_type: str
    size_bytes: int
    message: str
    ingest: IngestResponse


class DocumentRecord(BaseModel):
    document_id: str
    document_label: str
    company_name: str
    filename: str
    raw_pdf_path: str
    processed_pdf_path: str
    processed_text_path: str
    chunks_path: str
    uploaded_at: float
    chunk_count: int = 0


class DocumentOption(BaseModel):
    document_id: str
    document_label: str
    company_name: str
    filename: str
    chunk_count: int = 0
    uploaded_at: float = 0.0


class ConversationState(BaseModel):
    session_id: str
    last_question: str = ""
    last_resolved_question: str = ""
    last_company: str | None = None
    last_document_id: str | None = None
    last_document_label: str | None = None
    last_indicators: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    id: int
    question: str
    expected_answer: str
    retrieval_terms: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    id: int
    question: str
    expected_answer: str
    predicted_answer: str
    matched: bool
    source: str


class BenchmarkSummary(BaseModel):
    total: int
    correct: int
    avg_latency_ms: float
    results: list[BenchmarkResult]
