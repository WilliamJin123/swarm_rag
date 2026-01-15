"""
Factory for creating LLM providers based on configuration.

Uses UniversalLLMProvider for provider-agnostic LLM access through
keycycle's MultiProviderWrapper.
"""
from typing import Optional
import logging

from .provider import LLMProvider
from ..types.config import EvolutionConfigDict

logger = logging.getLogger(__name__)


# Known providers supported by keycycle
KNOWN_PROVIDERS = [
    "cerebras",
    "openai",
    "groq",
    "anthropic",
    "together",
    "fireworks",
    "deepseek",
]


class LLMProviderFactory:
    """
    Factory for creating LLM providers based on configuration.

    Uses UniversalLLMProvider which supports any provider through keycycle's
    MultiProviderWrapper unified API.
    """

    @classmethod
    def create(cls, config: EvolutionConfigDict) -> Optional[LLMProvider]:
        """
        Create an LLM provider based on configuration.

        Returns None if LLM mutation is not configured.

        Args:
            config: Evolution configuration dictionary
                - llm_provider: Provider name (e.g., "cerebras", "openai", "groq")
                - llm_model: Model ID (e.g., "zai-glm-4.7", "gpt-4o-mini")
                - llm_env_path: Path to .env file with API keys

        Returns:
            LLMProvider instance or None if not configured
        """
        # Check if LLM mutation is enabled
        mutation_strategy = config.get("mutation_strategy")
        if mutation_strategy != "llm_mutation":
            return None

        # Import here to avoid circular imports
        from .providers.universal import UniversalLLMProvider

        provider_name = config.get("llm_provider", "cerebras")
        model = config.get("llm_model", "zai-glm-4.7")
        env_path = config.get("llm_env_path", ".env")

        logger.info(
            f"Creating UniversalLLMProvider: provider={provider_name}, model={model}"
        )

        return UniversalLLMProvider(
            provider=provider_name,
            model=model,
            env_path=env_path
        )

    @classmethod
    def available_providers(cls) -> list:
        """Return list of known provider names supported by keycycle."""
        return KNOWN_PROVIDERS.copy()
