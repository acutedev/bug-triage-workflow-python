"""Configuration loading for the bug triage workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass

_SUPPORTED_LLM_PROVIDERS = {"openai"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    llm_provider: str
    llm_api_key: str
    llm_model: str
    human_approval_enabled: bool


def _load_dotenv_if_available() -> None:
    """Load `.env` values when python-dotenv is installed.

    The project should not require python-dotenv just to import configuration in
    tests, so this dependency is optional.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _parse_bool(value: str, *, variable_name: str) -> bool:
    normalized_value = value.strip().lower()

    if normalized_value in _TRUE_VALUES:
        return True

    if normalized_value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"{variable_name} must be a boolean value like true/false, yes/no, or 1/0"
    )


def load_config(*, require_api_key: bool = True) -> AppConfig:
    """Load and validate workflow configuration from environment variables."""
    _load_dotenv_if_available()

    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if llm_provider not in _SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_LLM_PROVIDERS))
        raise ValueError(f"LLM_PROVIDER must be one of: {supported}")

    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    if require_api_key and not llm_api_key:
        raise ValueError("LLM_API_KEY is required")

    llm_model = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
    if not llm_model:
        raise ValueError("LLM_MODEL cannot be blank")

    human_approval_enabled = _parse_bool(
        os.getenv("HUMAN_APPROVAL_ENABLED", "true"),
        variable_name="HUMAN_APPROVAL_ENABLED",
    )

    return AppConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        human_approval_enabled=human_approval_enabled,
    )
