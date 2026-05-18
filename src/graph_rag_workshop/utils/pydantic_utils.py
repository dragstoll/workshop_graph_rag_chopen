"""Utility functions for working with Pydantic AI models and providers."""

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from graph_rag_workshop.settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)


def get_ollama_model(
    model_name: str | None = None,
    base_url: str | None = None,
) -> OpenAIChatModel:
    """Create a Pydantic AI model backed by the local Ollama OpenAI API.

    Args:
        model_name: Optional name of the Ollama model to use. If not provided, it will be read from the OLLAMA_MODEL environment variable or default to DEFAULT_OLLAMA_MODEL.
        base_url: Optional base URL for the Ollama API. If not provided, it will be read from the OLLAMA_BASE_URL environment variable or default to DEFAULT_OLLAMA_BASE_URL.

    Returns:
        An instance of OpenAIChatModel configured to use the specified Ollama model and API base URL.
    """
    return OpenAIChatModel(
        model_name=model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        provider=OllamaProvider(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ),
    )
