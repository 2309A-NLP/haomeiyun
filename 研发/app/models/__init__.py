# app/models/__init__.py
from .database import (
    Base,
    User,
    Role,
    ChatSession,
    KnowledgeDocument,
    get_db,
    init_db,
    SessionLocal,
    engine
)
from .schemas import (
    LegalField,
    MessageRole,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    KnowledgeQueryRequest,
    KnowledgeDocumentResponse,
    RoleCreateRequest,
    RoleResponse
)

__all__ = [
    # database
    "Base",
    "User",
    "Role",
    "ChatSession",
    "KnowledgeDocument",
    "get_db",
    "init_db",
    "SessionLocal",
    "engine",
    # schemas
    "LegalField",
    "MessageRole",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "KnowledgeQueryRequest",
    "KnowledgeDocumentResponse",
    "RoleCreateRequest",
    "RoleResponse",
]
