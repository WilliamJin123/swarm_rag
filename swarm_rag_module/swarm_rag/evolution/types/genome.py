from typing import Dict, Callable, Any, List, Set, TypedDict, Tuple, Optional, Literal
from dataclasses import dataclass, field
import torch
import random

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from .fitness_results import FitnessResult
from .config import WeightTensors, MutationSigmas

from .expressions import ExpressionNode

from ...core.heuristics import HeuristicContext, HeuristicRegistry
from ...interfaces.types import AgentGroupConfig


# =============================================================================
# Fixed and Evolvable Parameter Definitions
# =============================================================================

# Fixed parameters - removed from evolution to reduce search space
# These rarely impact results significantly and are best left at sensible defaults
FIXED_PARAMS: Dict[str, Any] = {
    "drop_zone_inc": 0.05,  # Rarely impacts results significantly
    "start_subset": 10,     # 10 starting nodes is usually sufficient
}


class CompiledStrategies(TypedDict, total=False):
    """
    Type definition for the compiled function cache.
    total=False means keys can be missing (e.g. if 'deposit' isn't compiled yet).
    """
    movement: Callable
    ranking: Callable
    deposit: Callable

class SwarmParams(TypedDict):
    """
    Defines the contract for Swarm Hyperparameters.
    Acts as a Dictionary at runtime, but provides IDE autocompletion.
    """
    n_agents: int
    steps: int
    drop_zone_inc: float
    decay: float
    initial_pool_size: int
    start_subset: int
    

DEFAULT_PARAMS: SwarmParams = {
    "n_agents": 20,
    "steps": 4,
    "decay": 0.5,
    "drop_zone_inc": 0.05,
    "initial_pool_size": 30,
    "start_subset": 10,
}

@dataclass
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.

    Supports dual-mode evolution:
    - "expression_tree": Nonlinear symbolic expressions (expressive, default)
    - "weighted_sum": Linear heuristic combinations (fast, GPU-optimized)
    """
    id: str
    mutation_rate: float = 0.1

    # === Mode Selection ===
    mode: Literal["weighted_sum", "expression_tree"] = "expression_tree"

    # === Shared Fields ===
    params: SwarmParams = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    group_ratios: Dict[str, float] = field(default_factory=dict)

    # === Expression Tree Mode ===
    strategies: Dict[str, ExpressionNode] = field(default_factory=dict)

    # === Weighted Sum Mode ===
    weight_tensors: Optional[WeightTensors] = None

    # === Self-Adaptive Mutation (ES-style) ===
    mutation_sigmas: MutationSigmas = field(default_factory=MutationSigmas)

    # === Evaluation Results ===
    fitness: FitnessResult = field(default_factory=lambda: FitnessResult())

    # Recall@20, Hit@1, Hit@5, MRR, etc.
    metrics: Dict[str, float] = field(default_factory=dict)
    latency: float = 0.0
    evaluated: bool = False

    _compiled_cache: CompiledStrategies = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Ensure consistent state after initialization."""
        self.normalize_ratios()

    def __hash__(self):
        """Allows Genome to be used in sets or as dict keys."""
        return hash(self.id)

    def __eq__(self, other):
        """Genomes with the same ID are considered the same object."""
        if not isinstance(other, Genome):
            return False
        return self.id == other.id

    def __getstate__(self):
        """
        Custom pickling logic.
        EXCLUDES the compiled cache (functions) because they cannot be pickled.
        We only save the 'strategies' (trees) and 'params' (data).
        """
        state = self.__dict__.copy()
        # Remove the un-picklable cache
        state['_compiled_cache'] = {} 
        return state

    def __setstate__(self, state):
        """
        Restores state and re-initializes the empty cache.
        Handles backward compatibility for older checkpoints.
        """
        self.__dict__.update(state)
        # Ensure cache exists
        if '_compiled_cache' not in self.__dict__:
            self._compiled_cache = {}

        # Backward compatibility: set defaults for new fields
        if 'mode' not in self.__dict__:
            self.mode = "expression_tree"
        if 'weight_tensors' not in self.__dict__:
            self.weight_tensors = None
        if 'mutation_sigmas' not in self.__dict__:
            self.mutation_sigmas = MutationSigmas()

        self.normalize_ratios()

    def complexity(self) -> int:
        """
        Compute genome complexity.

        For expression_tree mode: Sum of the size of all expression trees.
        For weighted_sum mode: Total number of weight parameters.
        """
        if self.mode == "weighted_sum" and self.weight_tensors is not None:
            return self.weight_tensors.total_params
        return sum(tree.size() for tree in self.strategies.values())
    
    def normalize_ratios(self):
        """Ensures group_ratios sum to 1.0."""
        if not self.group_ratios:
            return

        n = len(self.group_ratios)
        if n == 0:  # Additional safety check
            return

        total = sum(self.group_ratios.values())
        if total <= 1e-9:
            # If all are zero, distribute evenly
            for k in self.group_ratios:
                self.group_ratios[k] = 1.0 / n
        else:
            for k in self.group_ratios:
                self.group_ratios[k] /= total
    
    def clear_cache(self):
        """Must be called after any mutation."""
        self._compiled_cache.clear()

    def get_kwargs(self) -> Dict[str, Any]:
        """
        Returns a dictionary ready to be unpacked into SwarmRetriever.retrieve().
        Merges params and compiled strategies.
        """
        # This will fail if not compiled, handled by Compiler/Evaluator
        if not self._compiled_cache and self.strategies:
            raise RuntimeError(f"Genome {self.id} accessed before compilation.")
            
        return {**self.params}
    
    def copy(self, new_id: str = None) -> 'Genome':
        """
        Deep copy ensures mutations don't bleed into parents/elites.
        """
        # Manually copy fitness to avoid reference sharing
        new_fitness = FitnessResult(
            quality_score=self.fitness.quality_score,
            stability_score=self.fitness.stability_score,
        )

        target_id = new_id if new_id is not None else f"{self.id}_copy"

        return Genome(
            id=target_id,
            mutation_rate=self.mutation_rate,
            mode=self.mode,
            params=self.params.copy(),
            group_ratios=self.group_ratios.copy(),
            strategies={k: v.copy() for k, v in self.strategies.items()},
            weight_tensors=self.weight_tensors.copy() if self.weight_tensors else None,
            mutation_sigmas=self.mutation_sigmas.copy(),
            fitness=new_fitness,
            metrics=self.metrics.copy(),
            latency=self.latency,
            evaluated=False,  # Children must be re-evaluated
            _compiled_cache={}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Genome into a plain Python dictionary.

        This is essential for saving genomes to JSON, logging, or other
        forms of data interchange. It intelligently handles nested objects
        and excludes non-serializable components like the compiled cache.

        Returns:
            A dictionary representation of the genome, ready for JSON serialization.
        """
        d = self.__dict__.copy()
        # Exclude the non-serializable compiled cache
        d.pop('_compiled_cache', None)

        # Serialize nested, complex objects.
        if 'strategies' in d and d['strategies']:
            d['strategies'] = {key: node.to_dict() for key, node in d['strategies'].items()}

        if 'fitness' in d and hasattr(d['fitness'], 'to_dict'):
            d['fitness'] = d['fitness'].to_dict()

        # Serialize weight_tensors (weighted_sum mode)
        if 'weight_tensors' in d and d['weight_tensors'] is not None:
            d['weight_tensors'] = d['weight_tensors'].to_dict()

        # Serialize mutation_sigmas
        if 'mutation_sigmas' in d and hasattr(d['mutation_sigmas'], 'to_dict'):
            d['mutation_sigmas'] = d['mutation_sigmas'].to_dict()

        return d
    
    def pretty_print(self, printer: Callable[[str], Any] = print):
        """
        Prints a human-readable summary of the genome using the provided printer.

        Args:
            printer: Function to handle output (e.g., print, logger.info). Defaults to print.
        """
        border = "=" * 60
        separator = "-" * 60

        printer(border)
        printer(f"GENOME REPORT: {self.id}")
        printer(f"Mode: {self.mode}")
        printer(border)

        # 1. Fitness & Metrics
        printer(f"Fitness Score: {self.fitness}")
        if self.metrics:
            printer("Detailed Metrics:")
            # Sort metrics for consistent output
            for k, v in sorted(self.metrics.items()):
                printer(f"  • {k:<15} : {v:.4f}")
        printer(separator)

        # 2. Hyperparameters
        printer("Hyperparameters:")
        max_len = max(len(k) for k in self.params.keys()) if self.params else 0
        for k, v in sorted(self.params.items()):
            printer(f"  • {k:<{max_len}} : {v}")

        # 3. Group Ratios
        if self.group_ratios:
            printer(separator)
            printer("Agent Group Ratios:")
            for k, v in sorted(self.group_ratios.items()):
                printer(f"  • {k:<5} : {v:.2%} ({v:.4f})")

        # 4a. Strategies (Expression Tree Mode)
        if self.mode == "expression_tree" and self.strategies:
            printer(separator)
            printer("Evolved Strategies (Expression Trees):")
            for name, tree in sorted(self.strategies.items()):
                # Clean up the name for display
                display_name = name.replace("evolved_", "").replace("_", " ").title()
                printer(f"  [ {display_name} ]")
                printer(f"    {tree.to_string()}")
                printer("")  # Empty line between strategies for readability

        # 4b. Weight Tensors (Weighted Sum Mode)
        if self.mode == "weighted_sum" and self.weight_tensors is not None:
            printer(separator)
            printer("Weight Tensors (Weighted Sum):")
            wt = self.weight_tensors
            printer(f"  Movement weights shape: {tuple(wt.movement_weights.shape)}")
            printer(f"  Deposit weights shape:  {tuple(wt.deposit_weights.shape)}")
            printer(f"  Ranking weights shape:  {tuple(wt.ranking_weights.shape)}")
            printer(f"  Total parameters:       {wt.total_params}")

            # Show actual weights for small tensors
            if wt.n_groups <= 3:
                printer("")
                for g in range(wt.n_groups):
                    printer(f"  Group {g} movement: {wt.movement_weights[g].tolist()}")
                    printer(f"  Group {g} deposit:  {wt.deposit_weights[g].tolist()}")
                printer(f"  Ranking weights: {wt.ranking_weights.tolist()}")
                printer(f"  Ranking bias: {wt.ranking_bias:.4f}")

        # 5. Mutation Sigmas (if self-adaptive)
        if self.mutation_sigmas:
            printer(separator)
            printer("Mutation Sigmas (Self-Adaptive):")
            printer(f"  • weight_sigma:    {self.mutation_sigmas.weight_sigma:.4f}")
            printer(f"  • bias_sigma:      {self.mutation_sigmas.bias_sigma:.4f}")
            printer(f"  • ratio_sigma:     {self.mutation_sigmas.ratio_sigma:.4f}")
            printer(f"  • hyperparam_sigma:{self.mutation_sigmas.hyperparam_sigma:.4f}")

        printer(border)

class GenomeCompiler:
    """
    Dynamically compiles a Genome's symbolic trees into executable logic.
    Decoupled from specific strategy names (works for 'movement', 'ranking', or anything new).
    """

    def compile(self, genome: Genome) -> Dict[str, Any]:
        """
        Converts a Genome into the specific kwargs required by SwarmRetriever.
        """
        valid_suffixes = {'movement', 'deposit', 'ranking'}
        for key in genome.strategies:
            if key == 'ranking': 
                continue
            
            # Split 'g0_movement' -> prefix='g0', suffix='movement'
            if "_" in key:
                prefix, suffix = key.rsplit('_', 1)
                if suffix not in valid_suffixes:
                    raise ValueError(f"Genome contains illegal strategy key: '{key}'. Suffix '{suffix}' not in {valid_suffixes}")
            else:
                raise ValueError(f"Genome contains malformed strategy key: '{key}' (expected format 'gN_type')")

        compiled_funcs = {}
        for name, expr_tree in genome.strategies.items():
            if name in genome._compiled_cache:
                compiled_func = genome._compiled_cache[name]
            else:
                compiled_func = self._compile_tree(expr_tree)
                genome._compiled_cache[name] = compiled_func
            compiled_funcs[name] = compiled_func

        ranking_strategies = {}
        if "ranking" in compiled_funcs:
             ranking_strategies["evolved_ranking"] = (compiled_funcs["ranking"], 1.0)    
        
        agent_groups = []
        total_agents = genome.params['n_agents']

        # Use the explicit group_ratios dict, sorted by key
        sorted_groups = sorted(genome.group_ratios.keys()) # ['g0', 'g1', ...]
        
        if sorted_groups:
            ratios = [genome.group_ratios[g] for g in sorted_groups]
            total_ratio = sum(ratios) 
            if total_ratio <= 1e-9:
                total_ratio = 1.0
            
            counts = [int(round(total_agents * (r / total_ratio))) for r in ratios]

            # Fix rounding remainder (dump into first group)
            if counts:
                current_sum = sum(counts)
                if current_sum < total_agents:
                    # Add deficit to first group
                    counts[0] += (total_agents - current_sum)
                elif current_sum > total_agents:
                    # Remove surplus from first group (rare with round, but possible)
                    counts[0] -= (current_sum - total_agents)
            
            for i, group_key in enumerate(sorted_groups):
                if counts[i] <= 0: continue
                
                # Extract index from "g0", "g1"
                idx_suffix = group_key[1:] 
                mov_key = f"{group_key}_movement"
                dep_key = f"{group_key}_deposit"
                
                group_config: AgentGroupConfig = {
                    "count": counts[i],
                    "movement_strategies": { f"evolved_mov_{idx_suffix}": (compiled_funcs.get(mov_key), 1.0) },
                    "deposit_strategies": { f"evolved_dep_{idx_suffix}": (compiled_funcs.get(dep_key), 1.0) },
                }
                agent_groups.append(group_config)

        # Return keyword arguments for retrieve()
        return {
            **genome.params, 
            "agent_groups": agent_groups,
            "ranking_strategies": ranking_strategies,
        }



    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles a single expression tree into a lambda.
        """
        raw_features = self._extract_features(expr_tree)
        required_features = sorted([str(f.value) if hasattr(f, 'value') else str(f) for f in raw_features])

        getters: List[Callable[[HeuristicContext], Any]] = []
        for name in required_features:
            # Handle Context Attributes (Fast Path)
            if name == 'degree':
                getters.append(lambda ctx: ctx.node_degrees)
            elif name == 'pheromone':
                getters.append(lambda ctx: ctx.pheromone_values)
            elif name == 'max_pheromone':
                getters.append(lambda ctx : ctx.max_pheromone)
            elif name == 'avg_degree':
                getters.append(lambda ctx: ctx.avg_degree)
            elif name == 'votes':
                getters.append(lambda ctx: ctx.votes)
            else:
                try:
                    func = HeuristicRegistry.get(name) 
                    getters.append(func)
                except KeyError:
                    print(f"Warning: Feature '{name}' not in registry. Assuming raw data.")
                    getters.append(lambda ctx, n=name: 0.0)
        
        # Signature: lambda arg1, arg2: ...
        compiled_lambda = expr_tree.compile(arg_names=required_features)
        
        def strategy_wrapper(ctx: HeuristicContext) -> float:
            args = [getter(ctx) for getter in getters]
            raw_scores = compiled_lambda(*args)
            if isinstance(raw_scores, torch.Tensor):
                return torch.nan_to_num(raw_scores, nan=0.0, posinf=10.0, neginf=-10.0)
            return raw_scores if raw_scores == raw_scores else 0.0  # Simple NaN check for scalars
        
        return strategy_wrapper

    def _extract_features(self, node: ExpressionNode) -> Set[str]:
        features = set()
        if node.type == 'feature':
            features.add(node.value)
        for child in node.children:
            features.update(self._extract_features(child))
        return features


# =============================================================================
# Helper Functions
# =============================================================================

def create_random_genome(genome_id: str = None) -> Genome:
    """
    Create a new genome with random evolvable params and fixed params.

    Uses FIXED_PARAMS for parameters that are removed from evolution,
    and samples evolvable parameters from SwarmParamRanges (the single source of truth).

    Args:
        genome_id: Optional ID for the genome (auto-generated if None)

    Returns:
        Newly created Genome with randomized evolvable params
    """
    import uuid
    from .config import SwarmParamRanges

    # Start with fixed values
    params = dict(FIXED_PARAMS)

    # Add evolvable params sampled from SwarmParamRanges (single source of truth)
    ranges = SwarmParamRanges()
    for name, (low, high) in ranges.to_evolvable_dict().items():
        if isinstance(low, int) and isinstance(high, int):
            params[name] = random.randint(low, high)
        else:
            params[name] = random.uniform(low, high)

    # Generate genome ID if not provided
    if genome_id is None:
        genome_id = f"random_{uuid.uuid4().hex[:8]}"

    return Genome(
        id=genome_id,
        params=params,
        group_ratios={},
        strategies={},
        evaluated=False,
    )