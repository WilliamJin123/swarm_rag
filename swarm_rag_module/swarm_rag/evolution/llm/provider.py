"""
LLM Provider abstraction for genome refinement.

Provides a clean interface for LLM-guided mutations with:
- Retry logic with exponential backoff
- Circuit breaker pattern for repeated failures
- Graceful fallback to standard mutation
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol
import logging
import time

from ..types.genome import Genome
from ..types.config import EvolutionContext

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM mutation."""
    diagnosis: str
    proposed_changes: Dict[str, Any]
    raw_response: str
    success: bool
    error: Optional[str] = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def refine_genome(
        self,
        genome: Genome,
        context: EvolutionContext
    ) -> LLMResponse:
        """
        Request genome refinement from LLM.

        Args:
            genome: Genome to refine
            context: Evolution context

        Returns:
            LLMResponse with diagnosis and proposed changes
        """
        ...


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers with common functionality.

    Features:
    - Retry logic with exponential backoff
    - Circuit breaker pattern for repeated failures
    - Configurable parameters
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        circuit_threshold: int = 5
    ):
        """
        Initialize the base provider.

        Args:
            max_retries: Maximum retry attempts per call
            retry_delay: Base delay between retries (exponential backoff)
            circuit_threshold: Number of failures before circuit opens
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_threshold = circuit_threshold

    def refine_genome(self, genome: Genome, context: EvolutionContext) -> LLMResponse:
        """
        Wraps the actual LLM call with retry and circuit breaker logic.

        Args:
            genome: Genome to refine
            context: Evolution context

        Returns:
            LLMResponse with results or error information
        """
        # Check circuit breaker
        if self._circuit_open:
            return LLMResponse(
                diagnosis="Circuit breaker open",
                proposed_changes={},
                raw_response="",
                success=False,
                error="Too many failures, circuit breaker triggered"
            )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(genome, context)
                self._failure_count = 0  # Reset on success
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(sleep_time)

        # All retries exhausted
        self._failure_count += 1
        if self._failure_count >= self._circuit_threshold:
            self._circuit_open = True
            logger.error(
                f"Circuit breaker triggered after {self._failure_count} failures"
            )

        return LLMResponse(
            diagnosis="LLM call failed after retries",
            proposed_changes={},
            raw_response="",
            success=False,
            error=f"Max retries exceeded: {last_error}"
        )

    @abstractmethod
    def _call_llm(self, genome: Genome, context: EvolutionContext) -> LLMResponse:
        """
        Actual LLM call. Implemented by specific providers.

        Args:
            genome: Genome to refine
            context: Evolution context

        Returns:
            LLMResponse with results

        Raises:
            Exception: If the LLM call fails
        """
        pass

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._circuit_open = False
        self._failure_count = 0
        logger.info("Circuit breaker reset")

    @property
    def is_available(self) -> bool:
        """Check if provider is available (circuit not open)."""
        return not self._circuit_open
