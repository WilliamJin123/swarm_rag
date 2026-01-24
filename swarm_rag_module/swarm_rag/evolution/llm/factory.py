"""
Factory for creating LLM clients from configuration.
"""
from typing import Optional
import logging

from .client import LLMClient, SUPPORTED_PROVIDERS
from ..types.config import EvolutionConfig

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """
    Factory for creating LLMClient instances from configuration.
    """

    @classmethod
    def create(cls, config: EvolutionConfig) -> Optional[LLMClient]:
        """
        Create an LLMClient from configuration.

        Returns None if LLM is not configured/enabled.

        Args:
            config: Evolution configuration

        Returns:
            LLMClient instance or None if not configured
        """
        llm_enabled = config.llm.enabled
        mutation_strategy = config.genetic.mutation_strategy

        # Check if LLM is enabled
        if not llm_enabled and mutation_strategy != "llm_mutation":
            return None

        logger.info(f"Creating LLMClient: provider={config.llm.provider}, model={config.llm.model}")

        return LLMClient.from_config(
            provider=config.llm.provider,
            model=config.llm.model,
            env_path=config.llm.env_path,
        )

    @classmethod
    def available_providers(cls) -> list:
        """Return list of supported provider names."""
        return SUPPORTED_PROVIDERS.copy()


# Backwards compatibility alias
LLMProviderFactory = LLMClientFactory
