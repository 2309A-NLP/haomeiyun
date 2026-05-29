import time
from typing import Any, AsyncGenerator, Optional

import httpx

try:
    from openai import AsyncOpenAI
    OPENAI_IMPORT_ERROR = None
except ImportError as exc:
    AsyncOpenAI = Any  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc

from ..core.config import settings
from ..utils.logger import logger


class LLMService:
    """Unified async LLM client wrapper with provider fallback."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False) and not getattr(self, "_closed", False):
            return

        self.provider = settings.DEFAULT_LLM_PROVIDER
        self.model = settings.DEFAULT_LLM_MODEL
        self.clients: dict[str, AsyncOpenAI] = {}
        self.http_clients: dict[str, httpx.AsyncClient] = {}
        self._init_clients()
        self._initialized = True
        self._closed = False

    def _init_clients(self) -> None:
        timeout = settings.LLM_REQUEST_TIMEOUT_SECONDS

        self._register_client(
            name="ollama",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=timeout,
            log_message="Ollama client initialized at http://localhost:11434/v1",
        )

        if settings.DEEPSEEK_API_KEY:
            self._register_client(
                name="deepseek",
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                timeout=timeout,
                log_message="DeepSeek client initialized",
            )

        if settings.OPENAI_API_KEY:
            self._register_client(
                name="openai",
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                timeout=timeout,
                log_message="OpenAI client initialized",
            )

        if settings.SILICONFLOW_API_KEY:
            self._register_client(
                name="siliconflow",
                api_key=settings.SILICONFLOW_API_KEY,
                base_url=settings.SILICONFLOW_BASE_URL,
                timeout=timeout,
                log_message="SiliconFlow client initialized",
            )

        if settings.QWEN_API_KEY:
            self._register_client(
                name="qwen",
                api_key=settings.QWEN_API_KEY,
                base_url=settings.QWEN_BASE_URL,
                timeout=timeout,
                log_message="Qwen client initialized",
            )

        if settings.DOUBAO_API_KEY and settings.DOUBAO_BASE_URL:
            self._register_client(
                name="doubao",
                api_key=settings.DOUBAO_API_KEY,
                base_url=settings.DOUBAO_BASE_URL,
                timeout=timeout,
                log_message="Doubao client initialized",
            )

    def _register_client(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        timeout: int,
        log_message: str,
    ) -> None:
        if OPENAI_IMPORT_ERROR is not None:
            logger.warning(
                "Skipping %s client initialization because the openai package is unavailable: %s",
                name,
                OPENAI_IMPORT_ERROR,
            )
            return

        try:
            http_client = httpx.AsyncClient(timeout=timeout)
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                http_client=http_client,
            )
            self.clients[name] = client
            self.http_clients[name] = http_client
            logger.info(log_message)
        except Exception as exc:
            logger.warning("Failed to initialize %s client: %s", name, exc)

    def _resolve_provider(self, provider: Optional[str]) -> str:
        selected = provider or self.provider
        if selected in self.clients:
            return selected
        if "ollama" in self.clients:
            logger.warning("Provider %s unavailable, falling back to ollama", selected)
            return "ollama"
        if self.clients:
            fallback = next(iter(self.clients))
            logger.warning("Provider %s unavailable, falling back to %s", selected, fallback)
            return fallback
        if OPENAI_IMPORT_ERROR is not None:
            raise ValueError(
                "No available LLM provider because the openai package is not installed in the current "
                "Python environment. Please install project dependencies or start the project with .venv."
            )
        raise ValueError("No available LLM provider. Please check configuration.")

    def _provider_fallback_order(self, preferred: str) -> list[str]:
        ordered = []
        if preferred in self.clients:
            ordered.append(preferred)
        for name in self.clients:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        provider: str = None,
        model: str = None,
    ) -> str:
        selected_provider = self._resolve_provider(provider)
        selected_model = model or self.model
        messages = self._build_messages(prompt, system_prompt)
        last_error = None

        for provider_name in self._provider_fallback_order(selected_provider):
            client = self.clients[provider_name]
            try:
                start_time = time.time()
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = response.choices[0].message.content or ""
                elapsed = time.time() - start_time
                logger.info(
                    "LLM generation completed in %.2fs using %s/%s",
                    elapsed,
                    provider_name,
                    selected_model,
                )
                return result
            except Exception as exc:
                last_error = exc
                logger.error("LLM generation failed via %s/%s: %s", provider_name, selected_model, exc)
                continue

        raise last_error or RuntimeError("No LLM provider succeeded.")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        provider: str = None,
        model: str = None,
    ) -> AsyncGenerator[str, None]:
        selected_provider = self._resolve_provider(provider)
        selected_model = model or self.model
        messages = self._build_messages(prompt, system_prompt)
        last_error = None

        for provider_name in self._provider_fallback_order(selected_provider):
            client = self.clients[provider_name]
            try:
                stream = await client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except Exception as exc:
                last_error = exc
                logger.error("Stream generation failed via %s/%s: %s", provider_name, selected_model, exc)
                continue

        raise last_error or RuntimeError("No LLM provider succeeded.")

    async def aclose(self) -> None:
        if getattr(self, "_closed", False):
            return

        for client in self.http_clients.values():
            await client.aclose()

        self.clients.clear()
        self.http_clients.clear()
        self._closed = True
