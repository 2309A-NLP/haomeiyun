from __future__ import annotations

import json

from ..core.config import settings
from ..core.logging import logger
from ..models.schemas import ConversationState


class ConversationService:
    def __init__(self) -> None:
        self._client = None
        self._redis_available = False
        self._memory_store: dict[str, ConversationState] = {}
        self._init_client()

    def get_state(self, session_id: str | None) -> ConversationState | None:
        if not session_id:
            return None
        if not self._redis_available or self._client is None:
            return self._memory_store.get(session_id)

        try:
            payload = self._client.get(self._key(session_id))
        except Exception as exc:  # pragma: no cover - depends on Redis runtime
            logger.warning("Redis get failed for session %s: %s", session_id, exc)
            return None
        if not payload:
            return None

        try:
            return ConversationState.model_validate(json.loads(payload))
        except Exception as exc:
            logger.warning("Invalid Redis conversation payload for session %s: %s", session_id, exc)
            return None

    def save_state(self, state: ConversationState) -> None:
        if not state.session_id:
            return
        if not self._redis_available or self._client is None:
            self._memory_store[state.session_id] = state
            return

        try:
            self._client.setex(
                self._key(state.session_id),
                settings.redis_session_ttl_seconds,
                json.dumps(state.model_dump(), ensure_ascii=False),
            )
        except Exception as exc:  # pragma: no cover - depends on Redis runtime
            logger.warning("Redis save failed for session %s: %s", state.session_id, exc)

    def _key(self, session_id: str) -> str:
        return f"{settings.redis_key_prefix}:session:{session_id}"

    def _init_client(self) -> None:
        try:
            from redis import Redis
        except ImportError:
            logger.warning("redis package not installed; falling back to in-process conversation memory")
            return

        try:
            client = Redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
        except Exception as exc:  # pragma: no cover - depends on Redis runtime
            logger.warning("Redis unavailable at %s: %s; falling back to in-process conversation memory", settings.redis_url, exc)
            return

        self._client = client
        self._redis_available = True
        logger.info("Redis conversation memory enabled: url=%s", settings.redis_url)
