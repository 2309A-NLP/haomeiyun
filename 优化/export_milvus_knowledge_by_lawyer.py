from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR / "研发"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.vector_store.milvus_client import MilvusClient


ROLE_LEGAL_FIELD_MAP = {
    "lawyer": None,
    "criminal_lawyer": "criminal",
    "labor_lawyer": "labor",
    "family_lawyer": "family",
    "contract_lawyer": "contract",
}


def fetch_documents(client: MilvusClient, legal_field: str | None) -> list[dict]:
    client._ensure_connection()
    expr = f'legal_field == "{legal_field}"' if legal_field else 'knowledge_type == "law"'
    iterator = client.collection.query_iterator(
        batch_size=500,
        limit=-1,
        expr=expr,
        output_fields=[
            "id",
            "content",
            "source",
            "article_number",
            "legal_field",
            "knowledge_type",
            "document_title",
        ],
    )

    documents: list[dict] = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for item in batch:
                if legal_field is None and item.get("knowledge_type") != "law":
                    continue
                documents.append(
                    {
                        "id": item.get("id"),
                        "source": item.get("source", ""),
                        "article_number": item.get("article_number", ""),
                        "legal_field": item.get("legal_field", ""),
                        "knowledge_type": item.get("knowledge_type", ""),
                        "document_title": item.get("document_title", ""),
                        "content": item.get("content", ""),
                    }
                )
    finally:
        iterator.close()

    return documents


def main() -> None:
    output_dir = ROOT_DIR / "优化" / "exports" / "milvus_lawyer_knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = MilvusClient()
    summary: dict[str, int] = {}

    for role_name, legal_field in ROLE_LEGAL_FIELD_MAP.items():
        documents = fetch_documents(client, legal_field)
        output_path = output_dir / f"{role_name}.json"
        output_path.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[role_name] = len(documents)
        print(f"{role_name}: {len(documents)} -> {output_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
