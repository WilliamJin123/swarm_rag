"""
Factory for creating LLM providers based on configuration.

Uses UniversalLLMProvider for provider-agnostic LLM access through
keycycle's MultiProviderWrapper.
"""
from typing import Optional, Union, Dict, Any
import logging

from .provider import LLMProvider
from ..types.config import EvolutionConfig

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
    def create(
        cls, config: Union[EvolutionConfig, Dict[str, Any]]
    ) -> Optional[LLMProvider]:
        """
        Create an LLM provider based on configuration.

        Returns None if LLM mutation is not configured.

        Args:
            config: Evolution configuration (EvolutionConfig or legacy dict)

        Returns:
            LLMProvider instance or None if not configured
        """
        # Handle both new EvolutionConfig and legacy dict
        if isinstance(config, EvolutionConfig):
            llm_enabled = config.llm.enabled
            mutation_strategy = config.genetic.mutation_strategy
            provider_name = config.llm.provider
            model = config.llm.model
            env_path = config.llm.env_path
        else:
            # Legacy dict support
            llm_enabled = config.get("llm_enabled", False)
            mutation_strategy = config.get("mutation_strategy", "")
            provider_name = config.get("llm_provider", "cerebras")
            model = config.get("llm_model", "zai-glm-4.7")
            env_path = config.get("llm_env_path", ".env")

        # Check if LLM mutation is enabled
        if not llm_enabled and mutation_strategy != "llm_mutation":
            return None

        # Import here to avoid circular imports
        from .providers.universal import UniversalLLMProvider

        logger.info(
            f"Creating UniversalLLMProvider: provider={provider_name}, model={model}"
        )

        return UniversalLLMProvider(
            provider=provider_name, model=model, env_path=env_path
        )

    @classmethod
    def available_providers(cls) -> list:
        """Return list of known provider names supported by keycycle."""
        return KNOWN_PROVIDERS.copy()
