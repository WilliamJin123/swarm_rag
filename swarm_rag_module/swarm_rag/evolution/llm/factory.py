"""
Factory for creating LLM clients from configuration.
"""
from typing import Optional, Union, Dict, Any
import logging

from .client import LLMClient, SUPPORTED_PROVIDERS
from ..types.config import EvolutionConfig

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """
    Factory for creating LLMClient instances from configuration.
    """

    @classmethod
    def create(
        cls, config: Union[EvolutionConfig, Dict[str, Any]]
    ) -> Optional[LLMClient]:
        """
        Create an LLMClient from configuration.

        Returns None if LLM is not configured/enabled.

        Args:
            config: Evolution configuration (EvolutionConfig or legacy dict)

        Returns:
            LLMClient instance or None if not configured
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

        # Check if LLM is enabled
        if not llm_enabled and mutation_strategy != "llm_mutation":
            return None

        logger.info(f"Creating LLMClient: provider={provider_name}, model={model}")

        return LLMClient.from_config(
            provider=provider_name,
            model=model,
            env_path=env_path,
        )

    @classmethod
    def available_providers(cls) -> list:
        """Return list of supported provider names."""
        return SUPPORTED_PROVIDERS.copy()


# Backwards compatibility alias
LLMProviderFactory = LLMClientFactory
