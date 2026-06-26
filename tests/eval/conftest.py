import os

import pytest


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv()


@pytest.fixture(autouse=True)
def require_llm_api_key():
    if not os.environ.get("LLM_API_KEY", "").strip():
        pytest.skip("Live evals require LLM_API_KEY to be set. No API call was attempted.")
