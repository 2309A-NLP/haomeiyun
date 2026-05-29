# app/services/memory_service.py
import json
from datetime import datetime
from typing import Dict, List, Optional

import redis

from ..core.config import settings
from ..rag.embedding import BGEEmbedder
from ..utils.logger import logger
from ..vector_store.milvus_client import MilvusClient


class MemoryService:
    """Manage short-term chat context, full history, and session metadata."""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.short_term_max = settings.SHORT_TERM_MAX_LEN
        self.long_term_threshold = settings.LONG_TERM_THRESHOLD
        self.expire_seconds = settings.MEMORY_EXPIRE_DAYS * 86400
        self.embedder = BGEEmbedder()
        self.milvus_client = MilvusClient()

    def _get_key(self, user_id: str, session_id: str) -> str:
        return f"legal:chat:{user_id}:{session_id}"

    def _get_history_key(self, user_id: str, session_id: str) -> str:
        return f"legal:chat_history:{user_id}:{session_id}"

    def _get_session_meta_key(self, user_id: str, session_id: str) -> str:
        return f"legal:chat_meta:{user_id}:{session_id}"

    def _get_summary_key(self, user_id: str, session_id: str) -> str:
        return f"legal:summary:{user_id}:{session_id}"

    def _get_long_term_state_key(self, user_id: str, session_id: str) -> str:
        return f"legal:long_term_saved:{user_id}:{session_id}"

    def _build_preview(self, content: str, fallback: str = "新会话") -> str:
        text = (content or "").strip()
        if not text:
            return fallback
        preview = text[:50]
        if len(text) > 50:
            preview += "..."
        return preview

    def create_session(self, user_id: str, session_id: str) -> Dict:
        """Create an empty persisted session entry."""
        now = datetime.now().isoformat()
        meta = {
            "session_id": session_id,
            "last_message": "新会话",
            "user_preview": "新会话",
            "last_time": now,
            "created_at": now,
        }
        self.redis_client.set(
            self._get_session_meta_key(user_id, session_id),
            json.dumps(meta, ensure_ascii=False),
            ex=self.expire_seconds,
        )
        return meta

    async def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict]] = None,
    ) -> None:
        """Save a message to short-term context, full history, and session metadata."""
        recent_key = self._get_key(user_id, session_id)
        history_key = self._get_history_key(user_id, session_id)
        meta_key = self._get_session_meta_key(user_id, session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "citations": citations or [],  # type: ignore[var-annotated]
        }
        payload = json.dumps(message, ensure_ascii=False)
        preview = self._build_preview(content)

        existing_meta_raw = self.redis_client.get(meta_key)
        existing_meta = json.loads(existing_meta_raw) if existing_meta_raw else {}
        meta = {
            "session_id": session_id,
            "last_message": preview,
            "user_preview": preview if role == "user" else existing_meta.get("user_preview", "新会话"),
            "last_time": message["timestamp"],
            "created_at": existing_meta.get("created_at", message["timestamp"]),
        }

        pipe = self.redis_client.pipeline()
        pipe.lpush(recent_key, payload)
        pipe.ltrim(recent_key, 0, self.short_term_max - 1)
        pipe.expire(recent_key, self.expire_seconds)
        pipe.rpush(history_key, payload)
        pipe.expire(history_key, self.expire_seconds)
        pipe.set(meta_key, json.dumps(meta, ensure_ascii=False), ex=self.expire_seconds)
        pipe.execute()

        logger.debug("Saved message to %s and %s, role=%s", recent_key, history_key, role)

        if role == "assistant" and self.check_long_term_trigger(user_id, session_id):
            self._sync_long_term_memory(user_id, session_id)

    def get_recent_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int = None,
    ) -> List[Dict]:
        """Get recent messages for model context."""
        recent_key = self._get_key(user_id, session_id)
        limit = limit or self.short_term_max
        messages = self.redis_client.lrange(recent_key, 0, limit - 1)
        return [json.loads(item) for item in reversed(messages)]

    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict]:
        """Get full saved history for a session."""
        history_key = self._get_history_key(user_id, session_id)
        messages = self.redis_client.lrange(history_key, 0, -1)
        if messages:
            return [json.loads(item) for item in messages]
        return self.get_recent_messages(user_id, session_id)

    def get_conversation_context(
        self,
        user_id: str,
        session_id: str,
        max_turns: int = 10,
    ) -> str:
        """Format recent messages for model context."""
        messages = self.get_recent_messages(user_id, session_id, max_turns)
        context_parts = []
        for msg in messages:
            role_name = "用户" if msg["role"] == "user" else "律师"
            context_parts.append(f"{role_name}: {msg['content']}")
        return "\n".join(context_parts)

    def clear_memory(self, user_id: str, session_id: str) -> None:
        """Delete recent context, full history, and session metadata."""
        self.redis_client.delete(self._get_key(user_id, session_id))
        self.redis_client.delete(self._get_history_key(user_id, session_id))
        self.redis_client.delete(self._get_session_meta_key(user_id, session_id))
        self.redis_client.delete(self._get_long_term_state_key(user_id, session_id))
        logger.info("Cleared memory for %s:%s", user_id, session_id)

    def get_session_list(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user, preferring user prompts as previews."""
        sessions_by_id: Dict[str, Dict] = {}

        meta_pattern = f"legal:chat_meta:{user_id}:*"
        for key in self.redis_client.scan_iter(match=meta_pattern):
            raw = self.redis_client.get(key)
            if not raw:
                continue
            meta = json.loads(raw)
            session_id = meta.get("session_id") or key.split(":")[-1]
            sessions_by_id[session_id] = {
                "session_id": session_id,
                "last_message": meta.get("user_preview") or meta.get("last_message", "新会话"),
                "last_time": meta.get("last_time", meta.get("created_at", datetime.now().isoformat())),
            }

        pattern = f"legal:chat:{user_id}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            session_id = key.split(":")[-1]
            last_msg = self.redis_client.lindex(key, 0)
            if not last_msg:
                continue
            msg_data = json.loads(last_msg)
            preview = self._build_preview(msg_data.get("content", ""))
            existing = sessions_by_id.get(session_id, {})
            sessions_by_id[session_id] = {
                "session_id": session_id,
                "last_message": existing.get("last_message", preview),
                "last_time": msg_data["timestamp"],
            }

        return sorted(sessions_by_id.values(), key=lambda item: item["last_time"], reverse=True)

    def _sync_long_term_memory(self, user_id: str, session_id: str) -> None:
        """Persist enough conversation messages into vector memory when needed."""
        current_length = self.redis_client.llen(self._get_key(user_id, session_id))
        saved_count = self.redis_client.get(self._get_long_term_state_key(user_id, session_id))
        saved_count = int(saved_count) if saved_count and saved_count.isdigit() else 0

        if current_length <= saved_count:
            return

        messages = self.get_recent_messages(user_id, session_id, self.long_term_threshold)
        if not messages:
            return

        try:
            content_texts = [f"{msg['role']}: {msg['content']}" for msg in messages]
            embed_result = self.embedder.encode(content_texts, return_sparse=False)
            dense_vectors = [vec.tolist() for vec in embed_result["dense_vecs"]]

            self.milvus_client.insert(
                dense_vectors=dense_vectors,
                contents=content_texts,
                sources=[f"chat:{user_id}:{session_id}"] * len(content_texts),
                article_numbers=[""] * len(content_texts),
                legal_fields=["chat"] * len(content_texts),
                knowledge_types=["conversation"] * len(content_texts),
                document_titles=[f"chat_{user_id}_{session_id}"] * len(content_texts),
            )
            self.redis_client.set(
                self._get_long_term_state_key(user_id, session_id),
                current_length,
                ex=self.expire_seconds,
            )
            logger.info("Saved %s long-term chat messages for %s:%s", len(messages), user_id, session_id)
        except Exception as exc:
            logger.error("Failed to save long-term memory for %s:%s: %s", user_id, session_id, exc)

    def check_long_term_trigger(self, user_id: str, session_id: str) -> bool:
        """Check whether the session has enough messages for vector memory sync."""
        recent_key = self._get_key(user_id, session_id)
        return self.redis_client.llen(recent_key) >= self.long_term_threshold
