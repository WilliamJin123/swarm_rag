"""
LLM-based genetic operators.

Provides mutation strategies that utilize LLM guidance for genome refinement.

Implements a three-tier architecture:
- Tier 1: Strategic Oracle (archive-level steering, called periodically)
- Tier 2: Tactical Advisor (per-genome diagnosis and intent prescription)
- Tier 3: Constrained Executor (intent-to-mutation translation)
"""
import logging
from typing import Optional, Any

from ..types.genome import Genome
from ..types.config import EvolutionContext
from ...interfaces.enums import GeneticKey

from .strategies import GeneticRegistry

from ..llm.utils import apply_llm_edits
from ..llm.provider import BaseLLMProvider
from ..llm.intents import StrategicDirective, MutationPrescription, MutationIntent
from ..llm.evolution_journal import EvolutionJournal, MutationRecord
from ..llm.constrained_executor import ConstrainedExecutor, ThreeTierMutator, ParameterBounds

logger = logging.getLogger(__name__)


# Global three-tier mutator instance (initialized on first use)
_three_tier_mutator: Optional[ThreeTierMutator] = None


def get_three_tier_mutator(ctx: EvolutionContext) -> Optional[ThreeTierMutator]:
    """
    Get or create the three-tier mutator instance.

    Args:
        ctx: Evolution context with LLM provider

    Returns:
        ThreeTierMutator or None if LLM not available
    """
    global _three_tier_mutator

    if _three_tier_mutator is not None:
        return _three_tier_mutator

    provider = ctx.llm_provider
    if provider is None:
        return None

    # Get LLM wrapper and model from provider
    if hasattr(provider, 'wrapper') and hasattr(provider, 'model'):
        wrapper = provider.wrapper
        model = provider.model
    else:
        logger.warning("Provider does not expose wrapper/model for three-tier architecture")
        return None

    # Get parameter bounds from config if available
    bounds = None
    if hasattr(ctx, 'config') and hasattr(ctx.config, 'genetic'):
        genetic = ctx.config.genetic
        if hasattr(genetic, 'param_ranges'):
            pr = genetic.param_ranges
            bounds = ParameterBounds(
                n_agents=getattr(pr, 'n_agents', (5, 30)),
                steps=getattr(pr, 'steps', (4, 12)),
                decay=getattr(pr, 'decay', (0.85, 0.99)),
                initial_pool_size=getattr(pr, 'initial_pool_size', (10, 50)),
                start_subset=getattr(pr, 'start_subset', (5, 15)),
                drop_zone_inc=getattr(pr, 'drop_zone_inc', (0.05, 0.2)),
            )

    _three_tier_mutator = ThreeTierMutator(wrapper, model, bounds)
    return _three_tier_mutator


def reset_three_tier_mutator():
    """Reset the global mutator (for testing or reconfiguration)."""
    global _three_tier_mutator
    _three_tier_mutator = None


class LLMStrategies:
    """
    Genetic operators that utilize an LLM.

    Uses the three-tier architecture:
    1. Strategic Oracle: High-level evolution steering (called periodically)
    2. Tactical Advisor: Per-genome diagnosis and intent prescription
    3. Constrained Executor: Safe, validated mutations

    Falls back to standard mutation if LLM is unavailable or fails.
    """

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.LLM_MUTATION)
    def llm_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Mutates a genome using the three-tier LLM architecture.

        Falls back to standard mutation if:
        - LLM provider not available
        - Three-tier system fails
        - No valid changes proposed

        Args:
            genome: Genome to mutate (already a copy)
            ctx: Evolution context with LLM provider and journal

        Returns:
            Mutated genome
        """
        # Try three-tier mutation first
        mutator = get_three_tier_mutator(ctx)

        if mutator is None:
            logger.warning(
                "Three-tier mutator not available. "
                "Falling back to expression_tree_mutation."
            )
            return _fallback_mutation(genome, ctx)

        try:
            # Get journal from context if available
            journal = getattr(ctx, 'evolution_journal', None)

            # Store parent fitness for journal
            parent_fitness = genome.fitness.quality_score

            # Perform three-tier mutation
            result, prescription = mutator.mutate(genome, journal)

            if not result.any_changes():
                if prescription.primary_intent == MutationIntent.NO_CHANGE:
                    logger.debug(f"Three-tier: no change needed for {genome.id}")
                    # Apply small jitter for diversity
                    executor = ConstrainedExecutor()
                    executor.apply_jitter(genome, magnitude=0.05)
                else:
                    logger.info(
                        f"Three-tier proposed no valid changes for {genome.id}. "
                        "Applying fallback mutation."
                    )
                    return _fallback_mutation(genome, ctx)

            # Record mutation in journal (outcome will be updated after evaluation)
            if journal is not None:
                parent_id = getattr(genome, '_parent_id', None)
                record = journal.record_mutation(
                    generation=ctx.generation,
                    genome_id=genome.id,
                    parent_genome_id=parent_id,
                    diagnosis=prescription.diagnosis,
                    intent=prescription.primary_intent,
                    target_component=prescription.target_component,
                    confidence=prescription.confidence,
                    param_changes=result.param_changes,
                    strategy_changes=result.strategy_changes,
                    fitness_before=parent_fitness,
                )
                # Store record reference on genome for later outcome update
                genome._mutation_record = record

            logger.debug(
                f"Three-tier mutation for {genome.id}: "
                f"intent={prescription.primary_intent.value}, "
                f"params_changed={len(result.param_changes)}, "
                f"strategies_changed={len(result.strategy_changes)}"
            )

        except Exception as e:
            logger.error(f"Three-tier mutation failed for {genome.id}: {e}")
            return _fallback_mutation(genome, ctx)

        return genome

    @staticmethod
    def update_strategic_directive(
        ctx: EvolutionContext,
        archive_stats: dict,
        population: list,
    ) -> Optional[StrategicDirective]:
        """
        Update the strategic directive (call periodically from orchestrator).

        Args:
            ctx: Evolution context
            archive_stats: Stats from archive.stats()
            population: Current archive population

        Returns:
            New StrategicDirective or None if not available
        """
        mutator = get_three_tier_mutator(ctx)
        if mutator is None:
            return None

        journal = getattr(ctx, 'evolution_journal', None)

        from ..llm.strategic_oracle import build_strategic_context

        strategic_ctx = build_strategic_context(
            archive_stats=archive_stats,
            journal=journal,
            generation=ctx.generation,
            total_generations=getattr(ctx.config, 'n_generations', 100),
            population=population,
        )

        return mutator.update_directive(strategic_ctx, journal)

    @staticmethod
    def should_update_oracle(ctx: EvolutionContext) -> bool:
        """
        Check if the strategic oracle should be updated.

        Args:
            ctx: Evolution context

        Returns:
            True if oracle should be called
        """
        mutator = get_three_tier_mutator(ctx)
        if mutator is None:
            return False

        journal = getattr(ctx, 'evolution_journal', None)
        stagnation = journal.get_stagnation_count() if journal else 0

        return mutator.should_update_oracle(ctx.generation, stagnation)


def _fallback_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
    """Apply fallback standard mutation."""
    return GeneticRegistry.get_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)(genome, ctx)


# Legacy support: Keep the old provider-based mutation available
class LegacyLLMStrategies:
    """
    Legacy LLM mutation using direct provider calls.

    This is the old approach where the LLM directly generates expression strings.
    Kept for backwards compatibility but not recommended.
    """

    @staticmethod
    def llm_mutation_legacy(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Legacy mutation using direct LLM expression generation.

        Args:
            genome: Genome to mutate
            ctx: Evolution context

        Returns:
            Mutated genome
        """
        provider: BaseLLMProvider = ctx.llm_provider

        if provider is None:
            return _fallback_mutation(genome, ctx)

        try:
            response = provider.refine_genome(genome, ctx)

            if not response.success:
                return _fallback_mutation(genome, ctx)

            edits = {
                "diagnosis": response.diagnosis,
                "proposed_changes": response.proposed_changes,
            }
            applied = apply_llm_edits(genome, edits)

            if not applied:
                return _fallback_mutation(genome, ctx)

        except Exception as e:
            logger.error(f"Legacy LLM Mutation failed for {genome.id}: {e}")
            return _fallback_mutation(genome, ctx)

        return genome
