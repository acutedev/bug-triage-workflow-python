"""OpenAI provider adapter for classifier prompts.

This adapter is intentionally thin: it sends the classifier prompt to OpenAI and
returns the model's text output. The classifier module remains responsible for
JSON parsing and Pydantic validation.
"""

from __future__ import annotations

from typing import Any

from src.config import AppConfig
from src.logging_config import get_logger

logger = get_logger("openai_provider")


class OpenAIClassifierClient:
    """Callable OpenAI adapter compatible with classify_bug_report."""

    def __init__(self, config: AppConfig, *, client: Any | None = None) -> None:
        if config.llm_provider != "openai":
            raise ValueError("OpenAIClassifierClient requires LLM_PROVIDER=openai")

        if not config.llm_api_key:
            raise ValueError("LLM_API_KEY is required for OpenAIClassifierClient")

        self._config = config
        self._client = client or self._build_default_client(config.llm_api_key)

    @staticmethod
    def _build_default_client(api_key: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to use OpenAIClassifierClient"
            ) from exc

        return OpenAI(api_key=api_key)

    def __call__(self, prompt: str) -> str:
        logger.info(
            "OpenAI classification request started",
            extra={
                "executor": "OpenAIClassifierClient",
                "model": self._config.llm_model,
            },
        )

        try:
            response = self._client.responses.create(
                model=self._config.llm_model,
                input=prompt,
                temperature=0,
            )
        except Exception:
            logger.exception(
                "OpenAI classification request failed",
                extra={
                    "executor": "OpenAIClassifierClient",
                    "model": self._config.llm_model,
                },
            )
            raise

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("OpenAI response did not include non-empty output_text")

        logger.info(
            "OpenAI classification request completed",
            extra={
                "executor": "OpenAIClassifierClient",
                "model": self._config.llm_model,
            },
        )

        return output_text
