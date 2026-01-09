from typing import Any, Dict, TypedDict
import logging
from ..types.genome import Genome, SwarmParams
from .parsers import ExpressionParser

logger = logging.getLogger(__name__)

class GenomePerformance(TypedDict):
    quality_score: float
    cost_score: float
    stability_score: float
    recall_at_20: float
    hit_at_1: float
    latency: float
    complexity: int

class GenomeConfig(TypedDict):
    params: SwarmParams
    strategies: Dict[str, str]
    group_ratios: Dict[str, float]

class GenomeLLMContext(TypedDict):
    id: str
    performance: GenomePerformance
    current_config: GenomeConfig

def genome_to_json_context(genome: Genome) -> GenomeLLMContext:
    """
    Serializes a Genome into the format expected by the LLM.
    Focuses on Performance Metrics and Configuration (Params + Strategies).
    """
    
    # 1. Performance
    performance: GenomePerformance = {
        "quality_score": genome.fitness.quality_score,
        "cost_score": genome.fitness.cost_score,
        "stability_score": genome.fitness.stability_score,
        "recall_at_20": genome.metrics.get("Recall@20", 0.0),
        "hit_at_1": genome.metrics.get("Hit@1", 0.0),
        "latency": genome.latency,
        "complexity": genome.complexity()
    }
    
    # 2. Configuration
    # Convert expression trees to their string representation
    strategy_strings: Dict[str, str] = {}
    for name, tree in genome.strategies.items():
        strategy_strings[name] = tree.to_string()
        
    config: GenomeConfig = {
        "params": genome.params,
        "strategies": strategy_strings,
        "group_ratios": genome.group_ratios
    }
    
    return {
        "id": genome.id,
        "performance": performance,
        "current_config": config
    }

def apply_llm_edits(genome: Genome, edits: Dict[str, Any]) -> bool:
    """
    Parses the JSON response from an LLM and updates the Genome object in-place.
    Returns True if any changes were made.
    """
    if "proposed_changes" not in edits:
        return False

    changes = edits["proposed_changes"]
    modified = False
    
    # 1. Params
    if "params" in changes:
        for key, val in changes["params"].items():
            if key in genome.params:
                # Basic type check to prevent breaking schemas
                current_type = type(genome.params[key])
                try:
                    if current_type == int:
                        val = int(val)
                    elif current_type == float:
                        val = float(val)
                    
                    if genome.params[key] != val:
                        genome.params[key] = val
                        modified = True
                except (ValueError, TypeError):
                    pass # Ignore invalid types
    
    # 2. Strategies
    if "strategies" in changes:
        for key, expr_str in changes["strategies"].items():
            if not expr_str: continue
            
            # Verify key exists
            if key not in genome.strategies:
                continue

            try:
                # Parse string -> ExpressionNode
                new_tree = ExpressionParser.parse(expr_str)
                genome.strategies[key] = new_tree
                genome.clear_cache()
                modified = True
            except Exception as e:
                logger.warning(f"Failed to parse strategy '{key}' for {genome.id}: {e}")
    
    return modified
