from __future__ import annotations

import base64
from typing import Sequence

from openai import OpenAI

from ..core.config import settings
from ..core.logging import logger


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower().strip()
        self.enabled = bool(settings.llm_api_key and self.provider != "disabled")
        self.client = None
        self.vlm_enabled = bool(settings.vlm_enabled and settings.vlm_api_key)
        self.vlm_client = None
        if self.enabled:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
                timeout=settings.llm_timeout_seconds,
            )
        if self.vlm_enabled:
            self.vlm_client = OpenAI(
                api_key=settings.vlm_api_key,
                base_url=settings.vlm_base_url or None,
                timeout=settings.vlm_timeout_seconds,
            )

    def answer(self, prompt: str, system_prompt: str | None = None) -> str | None:
        if not self.enabled or self.client is None:
            return None

        try:
            if self.provider == "deepseek":
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt or "你是一个严谨的金融文档问答助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                choice = response.choices[0] if response.choices else None
                return choice.message.content.strip() if choice and choice.message and choice.message.content else None

            response = self.client.responses.create(
                model=settings.llm_model,
                input=prompt if not system_prompt else f"{system_prompt}\n\n{prompt}",
            )
            return getattr(response, "output_text", None)
        except Exception as exc:
            logger.warning("LLM request failed for provider=%s model=%s: %s", self.provider, settings.llm_model, exc)
            return None

    def answer_with_images(
        self,
        prompt: str,
        image_payloads: Sequence[bytes],
        system_prompt: str | None = None,
    ) -> str | None:
        if not self.vlm_enabled or self.vlm_client is None or not image_payloads:
            return None

        try:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for image_bytes in image_payloads:
                encoded = base64.b64encode(image_bytes).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                )

            response = self.vlm_client.chat.completions.create(
                model=settings.vlm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or "你是一个严谨的中文文档视觉问答助手。",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
            )
            choice = response.choices[0] if response.choices else None
            return choice.message.content.strip() if choice and choice.message and choice.message.content else None
        except Exception as exc:
            logger.warning("VLM request failed for model=%s: %s", settings.vlm_model, exc)
            return None
