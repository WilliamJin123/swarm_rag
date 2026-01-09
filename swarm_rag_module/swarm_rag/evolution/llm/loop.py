from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..types.genome import Genome
from ..types.config import EvolutionContext
from ..execution.strategies import GeneticRegistry
from .optimizer import LLMOptimizer, MockLLMOptimizer
from .parsers import ExpressionParser
from .utils import genome_to_json_context

logger = logging.getLogger(__name__)

class LLMEvolutionLoop:
    """
    Evolution Loop that uses an LLM (or Mock) to refine individuals 
    instead of random mutation/crossover.
    """
    def __init__(self, context: EvolutionContext, optimizer: LLMOptimizer = None):
        self.context = context
        # Default to Mock if not provided (for now)
        self.optimizer = optimizer or MockLLMOptimizer()
        self.selection_fn = GeneticRegistry.get_selection(context.config["selection_strategy"])
        
        # Concurrency for LLM calls
        self.max_workers = context.config.get("llm_concurrency", 4)

    def step(self, population: List[Genome]) -> List[Genome]:
        """
        Produces the NEXT generation using LLM-based refinement.
        """
        self.context.population = population
        self.context.generation += 1
        
        # 1. Elitism (Always keep the best)
        population.sort(key=lambda g: g.fitness, reverse=True)
        elite_count = int(self.context.config['population_size'] * self.context.config['elite_fraction'])
        offspring = population[:elite_count]
        
        # 2. Selection for Refinement
        needed = self.context.config['population_size'] - len(offspring)
        
        # We select parents to "coach". 
        # Note: We can select the same parent multiple times to try different refinements?
        # For now, let's assume we select 'needed' parents.
        parents = self.selection_fn(self.context, k=needed)
        
        logger.info(f"LLM Loop: Refining {len(parents)} agents with {self.max_workers} workers.")

        # 3. Parallel Refinement
        refined_genomes = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each parent
            future_to_parent = {
                executor.submit(self._refine_single_agent, p): p 
                for p in parents
            }
            
            for future in as_completed(future_to_parent):
                parent = future_to_parent[future]
                try:
                    child = future.result()
                    if child:
                        refined_genomes.append(child)
                    else:
                        # Fallback if refinement fails: keep parent or random mutation?
                        # Let's keep parent for safety
                        refined_genomes.append(parent.copy(new_id=f"{parent.id}_failed_refine"))
                except Exception as exc:
                    logger.error(f"Agent {parent.id} refinement generated an exception: {exc}")
                    refined_genomes.append(parent.copy(new_id=f"{parent.id}_error"))

        offspring.extend(refined_genomes)
        
        # Ensure population size is exact
        return offspring[:self.context.config['population_size']]

    def _refine_single_agent(self, parent: Genome) -> Genome:
        """
        Worker function:
        1. Copy Parent
        2. Call Optimizer
        3. Apply Changes
        4. Return Child
        """
        # Create child with new ID
        child_id = f"gen_{self.context.generation}_{parent.id.split('_')[-1]}" 
        # Note: ID generation strategy might need to be more robust to avoid collisions
        # if using simple split. Let's append a random suffix or just standard naming.
        import uuid
        child_id = f"gen_{self.context.generation}_{str(uuid.uuid4())[:8]}"
        
        child = parent.copy(new_id=child_id)
        
        # Call LLM (or Mock)
        # We pass 'parent' to refine_genome logic, but we apply changes to 'child'
        # Actually, let's pass the child so the context (ID) matches if needed, 
        # though metrics are from parent. 
        # Better: Pass parent, apply to child.
        
        response = self.optimizer.refine_genome(
            genome=parent, # Pass parent to analyze metrics
            evolution_context=self.context
        )
        
        # Apply Changes
        self._apply_edits(child, response)
        
        # Reset fitness/metrics since it changed
        child.evaluated = False
        child.metrics = {}
        # child.fitness is already copied but should be considered "stale" or "unknown"
        # The engine will re-evaluate it.
        
        return child

    def _apply_edits(self, genome: Genome, edits: Dict[str, Any]):
        """
        Parses the JSON response and updates the Genome object in-place.
        """
        if "proposed_changes" not in edits:
            return

        changes = edits["proposed_changes"]
        
        # 1. Params
        if "params" in changes:
            for key, val in changes["params"].items():
                if key in genome.params:
                    # Type safety check could go here
                    genome.params[key] = val
        
        # 2. Strategies
        if "strategies" in changes:
            for key, expr_str in changes["strategies"].items():
                if not expr_str: continue
                
                try:
                    # Parse string -> ExpressionNode
                    new_tree = ExpressionParser.parse(expr_str)
                    genome.strategies[key] = new_tree
                    # Clear cache since logic changed
                    genome.clear_cache()
                except Exception as e:
                    logger.warning(f"Failed to parse strategy '{key}' for {genome.id}: {e}")
                    # Keep original strategy if parsing fails
