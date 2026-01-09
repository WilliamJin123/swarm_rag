import logging
import random
from typing import Dict, Any


from ..types.genome import Genome
from ..types.config import EvolutionContext
from ...interfaces.enums import GeneticKey
from .strategies import GeneticRegistry
from ..llm.optimizer import LLMOptimizer
from ..llm.utils import apply_llm_edits


logger = logging.getLogger(__name__)

class LLMStrategies:
    """
    Genetic operators that utilize an LLM.
    Requires 'llm_optimizer' to be present in EvolutionContext.
    """

    @staticmethod
    @GeneticRegistry.register_mutation(GeneticKey.LLM_MUTATION)
    def llm_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        """
        Mutates a genome by asking an LLM to refine it based on its performance metrics.
        Falls back to standard mutation if LLM fails or is not available.
        """
        # 1. Check for Optimizer
        optimizer: LLMOptimizer = getattr(ctx, 'llm_optimizer', None)
        if not optimizer:
            logger.warning("LLM Mutation requested but no optimizer found in context. Falling back to expression_tree_mutation.")
            return GeneticRegistry.get_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)(genome, ctx)

        # 2. Prepare Child (Copy)
        # We mutate a copy, not in-place (standard contract for mutation ops usually returns new or modified copy)
        # But looking at existing mutation ops, they often modify in place if they were passed a copy from crossover.
        # However, `loop.py` usually does: child = p1.copy(); child = mutation(child).
        # So 'genome' here is already a fresh copy we can modify.
        
        # 3. Call LLM
        try:
            response = optimizer.refine_genome(
                genome=genome, 
                evolution_context=ctx
            )
            
            # 4. Apply Changes
            applied = apply_llm_edits(genome, response)
            
            if not applied:
                # If LLM proposed no valid changes, maybe add a small jitter?
                logger.info(f"LLM proposed no changes for {genome.id}. Applying small jitter.")
                return GeneticRegistry.get_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)(genome, ctx)
                
        except Exception as e:
            logger.error(f"LLM Mutation failed for {genome.id}: {e}")
            # Fallback
            return GeneticRegistry.get_mutation(GeneticKey.EXPRESSION_TREE_MUTATION)(genome, ctx)

        return genome

