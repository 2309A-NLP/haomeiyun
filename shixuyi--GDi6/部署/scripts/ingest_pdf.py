from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.document_service import DocumentService
from app.services.rag_service import ProspectusRAGService


def main() -> None:
    document_service = DocumentService()
    result = document_service.ingest()
    ProspectusRAGService().refresh_index()
    print(f"source={result.source}")
    print(f"chunks={result.chunks}")
    print(f"saved_to={result.saved_to}")
    print(f"used_seed_fallback={result.used_seed_fallback}")


if __name__ == "__main__":
    main()
