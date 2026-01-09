from typing import Any, Dict, TypedDict
from ..types.genome import Genome, SwarmParams

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
