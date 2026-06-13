"""OpenAI provider construction for bug triage classification.

The primary integration creates a native Microsoft Agent Framework ``Agent``
backed by ``OpenAIChatClient``.
"""

from __future__ import annotations

from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from src.classifier import CLASSIFIER_AGENT_INSTRUCTIONS
from src.models import TriageClassification

from src.config import AppConfig
from src.logging_config import get_logger

logger = get_logger("openai_provider")

OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0
OPENAI_MAX_RETRIES = 2


def build_classifier_agent(
    config: AppConfig,
    *,
    client: Any | None = None,
) -> Agent:
    """Create the native MAF agent used as the workflow classifier node."""
    if config.llm_provider != "openai":
        raise ValueError("Classifier agent requires LLM_PROVIDER=openai")

    if not config.llm_api_key and client is None:
        raise ValueError("LLM_API_KEY is required for the classifier agent")

    if client is None:
        async_client = AsyncOpenAI(
            api_key=config.llm_api_key,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
        chat_client = OpenAIChatClient(
            model=config.llm_model,
            async_client=async_client,
        )
    else:
        chat_client = client

    logger.info(
        "Native classifier agent created",
        extra={
            "executor": "classifier_agent",
            "model": config.llm_model,
        },
    )

    return Agent(
        client=chat_client,
        name="classifier_agent",
        instructions=CLASSIFIER_AGENT_INSTRUCTIONS,
        default_options={"response_format": TriageClassification},
    )
