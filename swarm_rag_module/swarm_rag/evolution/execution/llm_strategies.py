"""
LLM-based genetic operators.

Provides mutation strategies that utilize LLM guidance for genome refinement.
"""
import logging



from ..types.genome import Genome
from ..types.config import EvolutionContext
from ...interfaces.enums import GeneticKey

from .strategies import GeneticRegistry

from ..llm.utils import apply_llm_edits
from ..llm.optimizer import LLMOptimizer
from ..llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMStrategies:
    """
    Genetic operators that utilize an LLM.

    Supports both:
    - New provider interface: ctx.llm_provider (LLMProvider protocol)
    - Legacy interface: ctx.llm_optimizer (LLMOptimizer class)

    Falls back to standard mutation if LLM is unavailable or fails.
    """

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.LLM_MUTATION)
    def llm_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Mutates a genome by asking an LLM to refine it based on performance metrics.

        Falls back to standard mutation if:
        - LLM provider/optimizer not available
        - LLM call fails
        - LLM proposes no valid changes

        Args:
            genome: Genome to mutate (already a copy)
            ctx: Evolution context with LLM provider

        Returns:
            Mutated genome
        """
        # Try new provider interface first, then legacy optimizer
        provider: BaseLLMProvider = getattr(ctx, 'llm_provider', None)
        optimizer: LLMOptimizer = getattr(ctx, 'llm_optimizer', None)

        if provider is None and optimizer is None:
            logger.warning(
                "LLM Mutation requested but no provider/optimizer found. "
                "Falling back to expression_tree_mutation."
            )
            return _fallback_mutation(genome, ctx)

        try:
            if provider is not None:
                # Use new provider interface
                response = provider.refine_genome(genome, ctx)

                if not response.success:
                    logger.warning(
                        f"LLM provider failed for {genome.id}: {response.error}. "
                        "Falling back to standard mutation."
                    )
                    return _fallback_mutation(genome, ctx)

                # Apply changes from provider response
                edits = {
                    "diagnosis": response.diagnosis,
                    "proposed_changes": response.proposed_changes
                }
                applied = apply_llm_edits(genome, edits)

            else:
                # Use legacy optimizer interface
                response = optimizer.refine_genome(
                    genome=genome,
                    evolution_context=ctx
                )
                applied = apply_llm_edits(genome, response)

            if not applied:
                logger.info(
                    f"LLM proposed no valid changes for {genome.id}. "
                    "Applying standard mutation."
                )
                return _fallback_mutation(genome, ctx)

        except Exception as e:
            logger.error(f"LLM Mutation failed for {genome.id}: {e}")
            return _fallback_mutation(genome, ctx)

        return genome


def _fallback_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    """Apply fallback standard mutation."""
    return GeneticRegistry.get_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)(genome, ctx)
