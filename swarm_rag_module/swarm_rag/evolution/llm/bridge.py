"""
LLM Bridge - Single entry point for all LLM features.

This module provides a unified interface to the LLM subsystem.
When LLM is disabled, all factory methods return null implementations
that do nothing but satisfy the API.

Usage:
    from swarm_rag.evolution.llm.bridge import LLMBridge

    # Initialize once
    bridge = LLMBridge.initialize(config)

    # Check if LLM is active
    if bridge.is_enabled():
        mutator = bridge.get_mutator()
        journal = bridge.create_journal()
"""
from typing import Optional, Any, TYPE_CHECKING

from .protocols import NullJournal, NullTracker

if TYPE_CHECKING:
    from ..types.config import LLMConfig, CreativeModeConfig


class LLMBridge:
    """
    Single entry point for all LLM features.

    Provides factory methods that return real or null implementations
    based on whether LLM is enabled.
    """

    _instance: Optional["LLMBridge"] = None
    _initialized: bool = False

    def __init__(self):
        self._enabled: bool = False
        self._client: Optional[Any] = None
        self._mutator: Optional[Any] = None
        self._config: Optional["LLMConfig"] = None
        self._creative_config: Optional["CreativeModeConfig"] = None

    @classmethod
    def get_instance(cls) -> "LLMBridge":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(
        cls,
        llm_config: Optional["LLMConfig"] = None,
        creative_config: Optional["CreativeModeConfig"] = None,
    ) -> "LLMBridge":
        """
        Initialize the LLM bridge with configuration.

        Args:
            llm_config: LLM configuration (from EvolutionConfig.llm)
            creative_config: Creative mode configuration

        Returns:
            Initialized LLMBridge instance
        """
        bridge = cls.get_instance()

        if llm_config is None or not llm_config.enabled:
            bridge._enabled = False
            bridge._client = None
            bridge._mutator = None
            cls._initialized = True
            return bridge

        # Lazy import to avoid loading LLM dependencies when disabled
        try:
            from .factory import LLMClientFactory
            from .constrained_executor import ThreeTierMutator, ParameterBounds

            bridge._config = llm_config
            bridge._creative_config = creative_config

            # Create client
            bridge._client = LLMClientFactory.create_from_env(
                provider=llm_config.provider,
                model=llm_config.model,
                env_path=llm_config.env_path,
            )

            # Create mutator
            bridge._mutator = ThreeTierMutator(
                client=bridge._client,
                bounds=ParameterBounds(),
                creative_mode_config=creative_config,
            )

            bridge._enabled = True

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to initialize LLM bridge: {e}. Running without LLM."
            )
            bridge._enabled = False
            bridge._client = None
            bridge._mutator = None

        cls._initialized = True
        return bridge

    @classmethod
    def reset(cls) -> None:
        """Reset the bridge (useful for testing)."""
        cls._instance = None
        cls._initialized = False

    def is_enabled(self) -> bool:
        """Check if LLM is active."""
        return self._enabled

    def get_mutator(self) -> Optional[Any]:
        """
        Get the three-tier mutator.

        Returns:
            ThreeTierMutator if LLM is enabled, None otherwise
        """
        return self._mutator

    def get_client(self) -> Optional[Any]:
        """
        Get the LLM client.

        Returns:
            LLMClient if LLM is enabled, None otherwise
        """
        return self._client

    def create_journal(self, max_records: int = 10000) -> Any:
        """
        Create an evolution journal.

        Returns:
            EvolutionJournal if LLM is enabled, NullJournal otherwise
        """
        if not self._enabled:
            return NullJournal()

        from .evolution_journal import EvolutionJournal
        return EvolutionJournal(max_records=max_records)

    def create_tracker(
        self,
        enabled: bool = True,
        sample_rate: float = 0.1,
        sampling_mode: str = "priority",
        **kwargs
    ) -> Any:
        """
        Create a decision tracker.

        Args:
            enabled: Whether tracking is active
            sample_rate: Fraction of decisions to capture
            sampling_mode: "uniform" or "priority"
            **kwargs: Additional tracker options

        Returns:
            DecisionTracker if LLM is enabled, NullTracker otherwise
        """
        if not self._enabled:
            return NullTracker(enabled=False)

        from .decision_tracker import DecisionTracker
        return DecisionTracker(
            enabled=enabled,
            sample_rate=sample_rate,
            sampling_mode=sampling_mode,
            **kwargs
        )

    def load_journal_from_dict(self, data: dict, max_records: int = 10000) -> Any:
        """
        Load a journal from checkpoint data.

        Args:
            data: Serialized journal data
            max_records: Maximum records to keep

        Returns:
            EvolutionJournal if LLM enabled, NullJournal otherwise
        """
        if not self._enabled:
            return NullJournal()

        from .evolution_journal import EvolutionJournal
        return EvolutionJournal.from_dict(data, max_records=max_records)


# Convenience functions for module-level access
def is_llm_enabled() -> bool:
    """Check if LLM is enabled."""
    return LLMBridge.get_instance().is_enabled()


def get_mutator() -> Optional[Any]:
    """Get the LLM mutator if enabled."""
    return LLMBridge.get_instance().get_mutator()


def create_journal(max_records: int = 10000) -> Any:
    """Create a journal (real or null based on LLM state)."""
    return LLMBridge.get_instance().create_journal(max_records)


def create_tracker(enabled: bool = True, sample_rate: float = 0.1, **kwargs) -> Any:
    """Create a tracker (real or null based on LLM state)."""
    return LLMBridge.get_instance().create_tracker(enabled, sample_rate, **kwargs)
