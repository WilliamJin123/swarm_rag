from typing import Any, Dict
from ..types.genome import Genome

def genome_to_json_context(genome: Genome) -> Dict[str, Any]:
    """
    Serializes a Genome into the format expected by the LLM.
    Focuses on Performance Metrics and Configuration (Params + Strategies).
    """
    
    # 1. Performance
    performance = {
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
    strategy_strings = {}
    for name, tree in genome.strategies.items():
        strategy_strings[name] = tree.to_string()
        
    config = {
        "params": genome.params,
        "strategies": strategy_strings,
        "group_ratios": genome.group_ratios
    }
    
    return {
        "id": genome.id,
        "performance": performance,
        "current_config": config
    }
