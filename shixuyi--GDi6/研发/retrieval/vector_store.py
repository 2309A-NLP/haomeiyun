from __future__ import annotations

import json
from functools import lru_cache

from ..core.config import settings
from ..core.logging import logger
from ..models.schemas import DocumentChunk


class MilvusVectorStore:
    def __init__(self) -> None:
        self.collection_name = settings.vector_collection
        self.dimension = settings.embedding_dimension
        self._ready = False

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk count and vector count must match")

        client = self._get_client()
        self._recreate_collection(client)

        rows = [self._build_row(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=False)]
        if not rows:
            self._ready = False
            return

        batch_size = 128
        for start in range(0, len(rows), batch_size):
            client.insert(collection_name=self.collection_name, data=rows[start : start + batch_size])

        client.flush(collection_name=self.collection_name)
        try:
            client.load_collection(collection_name=self.collection_name)
        except Exception as exc:  # pragma: no cover - depends on local Milvus runtime
            logger.warning("Milvus load_collection failed: %s", exc)

        self._ready = True
        logger.info("Saved %s chunks to Milvus collection %s", len(rows), self.collection_name)

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if not self.ready():
            return []

        del query_text
        client = self._get_client()
        filter_expr = self._build_filter_expr(document_ids)
        raw_results = client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=[
                "chunk_id",
                "source",
                "document_id",
                "document_label",
                "company_name",
                "page",
                "title",
                "text",
                "keywords",
                "metadata",
            ],
            filter=filter_expr,
        )

        hits = raw_results[0] if raw_results and isinstance(raw_results, list) else raw_results
        results: list[tuple[DocumentChunk, float]] = []
        for hit in hits or []:
            entity = self._extract_entity(hit)
            if not entity:
                continue
            results.append((self._to_chunk(entity), self._extract_score(hit)))
        return results

    def ready(self) -> bool:
        if self._ready:
            return True
        try:
            client = self._get_client()
            if not client.has_collection(collection_name=self.collection_name):
                return False
            stats = client.get_collection_stats(collection_name=self.collection_name)
            self._ready = int(stats.get("row_count", 0)) > 0
            return self._ready
        except Exception as exc:  # pragma: no cover - depends on local Milvus runtime
            logger.warning("Milvus ready check failed: %s", exc)
            return False

    def available(self) -> bool:
        try:
            self._server_version()
            return True
        except Exception as exc:  # pragma: no cover - depends on local Milvus runtime
            logger.warning("Milvus availability check failed: %s", exc)
            return False

    def _recreate_collection(self, client) -> None:
        if client.has_collection(collection_name=self.collection_name):
            client.drop_collection(collection_name=self.collection_name)

        api = self._milvus_api()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=api["DataType"].VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="source", datatype=api["DataType"].VARCHAR, max_length=512)
        schema.add_field(field_name="document_id", datatype=api["DataType"].VARCHAR, max_length=128)
        schema.add_field(field_name="document_label", datatype=api["DataType"].VARCHAR, max_length=512)
        schema.add_field(field_name="company_name", datatype=api["DataType"].VARCHAR, max_length=512)
        schema.add_field(field_name="page", datatype=api["DataType"].INT64)
        schema.add_field(field_name="title", datatype=api["DataType"].VARCHAR, max_length=512)
        text_field_args = {
            "field_name": "text",
            "datatype": api["DataType"].VARCHAR,
            "max_length": 65535,
        }
        if self._supports_full_text():
            text_field_args.update(
                {
                    "enable_match": True,
                    "enable_analyzer": True,
                    "analyzer_params": {"tokenizer": "jieba"},
                }
            )
        schema.add_field(**text_field_args)
        schema.add_field(field_name="keywords", datatype=api["DataType"].VARCHAR, max_length=8192)
        schema.add_field(field_name="metadata", datatype=api["DataType"].VARCHAR, max_length=16384)
        schema.add_field(field_name="dense_vector", datatype=api["DataType"].FLOAT_VECTOR, dim=self.dimension)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={},
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=settings.milvus_consistency_level,
        )

    def _build_row(self, chunk: DocumentChunk, vector: list[float]) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "document_id": chunk.document_id,
            "document_label": chunk.document_label,
            "company_name": chunk.company_name,
            "page": int(chunk.page or 0),
            "title": chunk.title,
            "text": chunk.text,
            "keywords": json.dumps(chunk.keywords, ensure_ascii=False),
            "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
            "dense_vector": vector,
        }

    def _to_chunk(self, entity: dict) -> DocumentChunk:
        try:
            keywords = json.loads(entity.get("keywords", "[]"))
        except Exception:
            keywords = []
        try:
            metadata = json.loads(entity.get("metadata", "{}"))
        except Exception:
            metadata = {}

        page = entity.get("page")
        return DocumentChunk(
            chunk_id=str(entity.get("chunk_id", "")),
            source=str(entity.get("source", "")),
            document_id=str(entity.get("document_id", "")),
            document_label=str(entity.get("document_label", "")),
            company_name=str(entity.get("company_name", "")),
            page=int(page) if page not in (None, "") else None,
            text=str(entity.get("text", "")),
            title=str(entity.get("title", "")),
            keywords=keywords if isinstance(keywords, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def _build_filter_expr(self, document_ids: list[str] | None) -> str | None:
        if not document_ids:
            return None
        quoted = [json.dumps(item, ensure_ascii=False) for item in document_ids if item]
        if not quoted:
            return None
        return f"document_id in [{', '.join(quoted)}]"

    def _extract_entity(self, hit) -> dict:
        if isinstance(hit, dict):
            entity = hit.get("entity") or hit.get("fields")
            return entity if isinstance(entity, dict) else hit

        entity = getattr(hit, "entity", None) or getattr(hit, "fields", None)
        if isinstance(entity, dict):
            return entity
        if hasattr(entity, "to_dict"):
            return entity.to_dict()
        return {}

    def _extract_score(self, hit) -> float:
        if isinstance(hit, dict):
            return float(hit.get("distance", hit.get("score", 0.0)))
        return float(getattr(hit, "distance", getattr(hit, "score", 0.0)))

    def _get_client(self):
        api = self._milvus_api()
        return api["MilvusClient"](
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_database,
        )

    @lru_cache(maxsize=1)
    def _server_version(self) -> str:
        api = self._milvus_api()
        alias = "codex_milvus"
        api["connections"].connect(alias=alias, uri=settings.milvus_uri, token=settings.milvus_token or None)
        return str(api["utility"].get_server_version(using=alias))

    def _supports_full_text(self) -> bool:
        version = self._server_version().lstrip("v")
        parts = [int(part) for part in version.split(".")[:3] if part.isdigit()]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3]) >= (2, 5, 0)

    @lru_cache(maxsize=1)
    def _milvus_api(self) -> dict:
        try:
            from pymilvus import DataType, MilvusClient, connections, utility
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("pymilvus is required for Milvus vector storage") from exc

        return {
            "DataType": DataType,
            "MilvusClient": MilvusClient,
            "connections": connections,
            "utility": utility,
        }
