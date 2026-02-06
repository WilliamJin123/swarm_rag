from typing import Dict, Callable, Any, List, Set, Tuple, Optional, Iterator, Union
from dataclasses import dataclass, field
import math
import torch
import random

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from .fitness_results import FitnessResult
from .config import WeightTensors, MutationSigmas, GenomeMode

from .expressions import ExpressionNode

from ...core.heuristics import HeuristicContext, HeuristicRegistry
from ...interfaces.types import AgentGroupConfig
# Import shared constants from centralized source
from ...utils.constants import SwarmParams, DEFAULT_SWARM_PARAMS


# =============================================================================
# Fixed and Evolvable Parameter Definitions
# =============================================================================

# Fixed parameters - removed from evolution to reduce search space
# These rarely impact results significantly and are best left at sensible defaults
FIXED_PARAMS: Dict[str, Any] = {
    "drop_zone_inc": 0.05,  # Rarely impacts results significantly
    "start_subset": 10,     # 10 starting nodes is usually sufficient
}


class CompiledStrategies:
    """
    Type definition for the compiled function cache.
    Using class instead of TypedDict for backward compatibility.
    """
    movement: Callable
    ranking: Callable
    deposit: Callable


# =============================================================================
# EvaluationMetrics: Typed replacement for Dict[str, float] (Audit 2.4)
# =============================================================================

class EvaluationMetrics:
    """
    Typed metrics container that replaces Dict[str, float] for genome.metrics.

    Has explicit fields for known metrics while supporting arbitrary extra keys
    for backward compatibility (e.g., 'variance', 'complexity', 'node_results').

    Supports full dict-like access: [], .get(), .items(), .keys(), .values(),
    iteration, len(), bool(), and copy(). This ensures all existing code that
    treats genome.metrics as a dict continues to work.
    """

    __slots__ = (
        'hit_at_1', 'hit_at_5', 'hit_at_10', 'hit_at_20',
        'recall_at_1', 'recall_at_5', 'recall_at_10', 'recall_at_20',
        'mrr', 'latency', 'diversity_node_types', 'diversity_count',
        '_extra',
    )

    _KEY_TO_ATTR = {
        'Hit@1': 'hit_at_1', 'Hit@5': 'hit_at_5',
        'Hit@10': 'hit_at_10', 'Hit@20': 'hit_at_20',
        'Recall@1': 'recall_at_1', 'Recall@5': 'recall_at_5',
        'Recall@10': 'recall_at_10', 'Recall@20': 'recall_at_20',
        'MRR': 'mrr', 'latency': 'latency',
        'Diversity_Node_Types': 'diversity_node_types',
        'Diversity_Count': 'diversity_count',
    }

    _ATTR_TO_KEY = {v: k for k, v in _KEY_TO_ATTR.items()}

    def __init__(self, data: Dict[str, Any] = None, **kwargs):
        self.hit_at_1: float = 0.0
        self.hit_at_5: float = 0.0
        self.hit_at_10: float = 0.0
        self.hit_at_20: float = 0.0
        self.recall_at_1: float = 0.0
        self.recall_at_5: float = 0.0
        self.recall_at_10: float = 0.0
        self.recall_at_20: float = 0.0
        self.mrr: float = 0.0
        self.latency: float = 0.0
        self.diversity_node_types: float = 0.0
        self.diversity_count: float = 0.0
        self._extra: Dict[str, Any] = {}
        if data:
            self.update(data)
        if kwargs:
            self.update(kwargs)

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            attr = self._KEY_TO_ATTR.get(key)
            if attr is not None:
                object.__setattr__(self, attr, value)
            else:
                self._extra[key] = value

    def __getitem__(self, key: str) -> Any:
        attr = self._KEY_TO_ATTR.get(key)
        if attr is not None:
            return getattr(self, attr)
        if key in self._extra:
            return self._extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        attr = self._KEY_TO_ATTR.get(key)
        if attr is not None:
            object.__setattr__(self, attr, value)
        else:
            self._extra[key] = value

    def __contains__(self, key: str) -> bool:
        attr = self._KEY_TO_ATTR.get(key)
        if attr is not None:
            return True
        return key in self._extra

    def __delitem__(self, key: str) -> None:
        if key in self._extra:
            del self._extra[key]
        elif key in self._KEY_TO_ATTR:
            object.__setattr__(self, self._KEY_TO_ATTR[key], 0.0)
        else:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        result = []
        for dict_key, attr in self._KEY_TO_ATTR.items():
            if getattr(self, attr) != 0.0:
                result.append(dict_key)
        result.extend(self._extra.keys())
        return result

    def values(self) -> List[Any]:
        result = []
        for dict_key, attr in self._KEY_TO_ATTR.items():
            if getattr(self, attr) != 0.0:
                result.append(getattr(self, attr))
        result.extend(self._extra.values())
        return result

    def items(self) -> List[Tuple[str, Any]]:
        result = []
        for dict_key, attr in self._KEY_TO_ATTR.items():
            val = getattr(self, attr)
            if val != 0.0:
                result.append((dict_key, val))
        result.extend(self._extra.items())
        return result

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        count = sum(1 for attr in self._KEY_TO_ATTR.values() if getattr(self, attr) != 0.0)
        return count + len(self._extra)

    def __bool__(self) -> bool:
        for attr in self._KEY_TO_ATTR.values():
            if getattr(self, attr) != 0.0:
                return True
        return bool(self._extra)

    def clear(self) -> None:
        for attr in self._KEY_TO_ATTR.values():
            object.__setattr__(self, attr, 0.0)
        self._extra.clear()

    def copy(self) -> 'EvaluationMetrics':
        new = EvaluationMetrics()
        for attr in self._KEY_TO_ATTR.values():
            object.__setattr__(new, attr, getattr(self, attr))
        new._extra = self._extra.copy()
        return new

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for dict_key, attr in self._KEY_TO_ATTR.items():
            val = getattr(self, attr)
            if val != 0.0:
                result[dict_key] = val
        result.update(self._extra)
        return result

    def __repr__(self) -> str:
        return f"EvaluationMetrics({self.to_dict()})"

    def __eq__(self, other) -> bool:
        if isinstance(other, EvaluationMetrics):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        return cls(data=data)


def _coerce_metrics(value):
    """Coerce a value to EvaluationMetrics. Accepts dict, EvaluationMetrics, or None."""
    if isinstance(value, EvaluationMetrics):
        return value
    if isinstance(value, dict):
        return EvaluationMetrics(data=value)
    if value is None:
        return EvaluationMetrics()
    raise TypeError(f"Cannot coerce {type(value).__name__} to EvaluationMetrics")


# =============================================================================
# CompiledGenome: Typed return from GenomeCompiler.compile() (Audit 2.1)
# =============================================================================

@dataclass
class CompiledGenome:
    """
    Typed result from GenomeCompiler.compile().

    Replaces the untyped Dict[str, Any] return value with explicit fields.
    Supports dict-like access for backward compatibility.
    """
    agent_groups: List[AgentGroupConfig]
    ranking_strategies: Dict[str, Tuple[Callable, float]]
    params: SwarmParams

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.params)
        result['agent_groups'] = self.agent_groups
        result['ranking_strategies'] = self.ranking_strategies
        return result

    def __getitem__(self, key: str) -> Any:
        if key == 'agent_groups':
            return self.agent_groups
        if key == 'ranking_strategies':
            return self.ranking_strategies
        if key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in ('agent_groups', 'ranking_strategies') or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return list(self.params.keys()) + ['agent_groups', 'ranking_strategies']

    def values(self):
        return list(self.params.values()) + [self.agent_groups, self.ranking_strategies]

    def items(self):
        result = list(self.params.items())
        result.append(('agent_groups', self.agent_groups))
        result.append(('ranking_strategies', self.ranking_strategies))
        return result

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.params) + 2

    def copy(self) -> Dict[str, Any]:
        return self.to_dict()


# Re-export DEFAULT_PARAMS for backward compatibility
# The single source of truth is in utils.constants.DEFAULT_SWARM_PARAMS
DEFAULT_PARAMS: SwarmParams = DEFAULT_SWARM_PARAMS

@dataclass
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.

    Supports dual-mode evolution:
    - GenomeMode.EXPRESSION_TREE: Nonlinear symbolic expressions (expressive, default)
    - GenomeMode.WEIGHTED_SUM: Linear heuristic combinations (fast, GPU-optimized)
    """
    id: str
    mutation_rate: float = 0.1

    # === Mode Selection ===
    mode: GenomeMode = GenomeMode.EXPRESSION_TREE

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
    # Now uses EvaluationMetrics (dict-compatible) instead of plain Dict[str, float].
    # Assignment from a plain dict is auto-coerced via __setattr__.
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    latency: float = 0.0
    evaluated: bool = False

    _compiled_cache: CompiledStrategies = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Ensure consistent state after initialization."""
        # Coerce metrics from dict to EvaluationMetrics if needed
        if not isinstance(self.metrics, EvaluationMetrics):
            object.__setattr__(self, 'metrics', _coerce_metrics(self.metrics))
        self.normalize_ratios()

    def __setattr__(self, name: str, value) -> None:
        """
        Custom attribute setter for two purposes:

        1. Auto-coerce plain dicts assigned to 'metrics' into EvaluationMetrics (Audit 2.4).
        2. Auto-invalidate _compiled_cache when 'strategies' is reassigned (Audit 3.3).
        """
        if name == 'metrics' and not isinstance(value, EvaluationMetrics):
            value = _coerce_metrics(value)
        if name == 'strategies':
            # Invalidate compiled cache when strategies are replaced
            try:
                cache = object.__getattribute__(self, '_compiled_cache')
                if isinstance(cache, dict) and cache:
                    cache.clear()
            except AttributeError:
                pass  # During __init__, _compiled_cache may not exist yet
        object.__setattr__(self, name, value)

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
        # Use object.__setattr__ to bypass our custom __setattr__ during restore
        for key, val in state.items():
            object.__setattr__(self, key, val)

        # Ensure cache exists
        if '_compiled_cache' not in state:
            object.__setattr__(self, '_compiled_cache', {})

        # Backward compatibility: set defaults for new fields
        if 'mode' not in state:
            object.__setattr__(self, 'mode', GenomeMode.EXPRESSION_TREE)
        elif isinstance(self.mode, str):
            object.__setattr__(self, 'mode', GenomeMode(self.mode))
        if 'weight_tensors' not in state:
            object.__setattr__(self, 'weight_tensors', None)
        if 'mutation_sigmas' not in state:
            object.__setattr__(self, 'mutation_sigmas', MutationSigmas())

        # Coerce legacy dict metrics to EvaluationMetrics
        if 'metrics' in state and not isinstance(self.metrics, EvaluationMetrics):
            object.__setattr__(self, 'metrics', _coerce_metrics(self.metrics))
        elif 'metrics' not in state:
            object.__setattr__(self, 'metrics', EvaluationMetrics())

        self.normalize_ratios()

    def complexity(self) -> int:
        """
        Compute genome complexity.

        For expression_tree mode: Sum of the size of all expression trees.
        For weighted_sum mode: Total number of weight parameters.
        """
        if self.mode == GenomeMode.WEIGHTED_SUM and self.weight_tensors is not None:
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

    def set_evaluation_result(
        self,
        fitness: FitnessResult,
        metrics: Union[Dict[str, float], 'EvaluationMetrics'],
        latency: float = 0.0,
        evaluated: bool = True,
    ) -> None:
        """
        Atomically set all evaluation results at once (Audit 3.1).

        Instead of setting genome.fitness, genome.metrics, genome.latency,
        and genome.evaluated individually (risking partial updates), this
        method sets them all in one call.

        Args:
            fitness: The computed FitnessResult.
            metrics: Evaluation metrics (dict or EvaluationMetrics).
            latency: Average per-query latency in seconds.
            evaluated: Whether the genome has been fully evaluated (default True).
        """
        self.fitness = fitness
        self.metrics = metrics  # Auto-coerced by __setattr__
        self.latency = latency
        self.evaluated = evaluated

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

        # Serialize mode enum to string for JSON compatibility
        if 'mode' in d and isinstance(d['mode'], GenomeMode):
            d['mode'] = d['mode'].value

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

        # Serialize EvaluationMetrics to plain dict
        if 'metrics' in d and isinstance(d['metrics'], EvaluationMetrics):
            d['metrics'] = d['metrics'].to_dict()

        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """
        Deserialize a Genome from a plain Python dictionary (Audit 2.5).

        Symmetric with to_dict(). Handles enum deserialization and nested
        object reconstruction consistently.

        Args:
            data: Dictionary as produced by to_dict() or JSON deserialization.

        Returns:
            A fully reconstructed Genome instance.
        """
        d = data.copy()

        # --- Reconstruct enums ---
        mode_val = d.pop('mode', GenomeMode.EXPRESSION_TREE.value)
        if isinstance(mode_val, str):
            mode = GenomeMode(mode_val)
        elif isinstance(mode_val, GenomeMode):
            mode = mode_val
        else:
            mode = GenomeMode.EXPRESSION_TREE

        # --- Reconstruct nested objects ---

        # Strategies: dict of ExpressionNode dicts -> ExpressionNode objects
        strategies_data = d.pop('strategies', {})
        strategies = {}
        if strategies_data:
            for key, node_data in strategies_data.items():
                if isinstance(node_data, dict):
                    strategies[key] = _expression_node_from_dict(node_data)
                elif isinstance(node_data, ExpressionNode):
                    strategies[key] = node_data

        # Fitness
        fitness_data = d.pop('fitness', None)
        if fitness_data is not None:
            if isinstance(fitness_data, dict):
                fitness = FitnessResult.from_dict(fitness_data)
            elif isinstance(fitness_data, FitnessResult):
                fitness = fitness_data
            else:
                fitness = FitnessResult()
        else:
            fitness = FitnessResult()

        # Weight tensors
        wt_data = d.pop('weight_tensors', None)
        if wt_data is not None:
            if isinstance(wt_data, dict):
                weight_tensors = WeightTensors.from_dict(wt_data)
            elif isinstance(wt_data, WeightTensors):
                weight_tensors = wt_data
            else:
                weight_tensors = None
        else:
            weight_tensors = None

        # Mutation sigmas
        ms_data = d.pop('mutation_sigmas', None)
        if ms_data is not None:
            if isinstance(ms_data, dict):
                mutation_sigmas = MutationSigmas.from_dict(ms_data)
            elif isinstance(ms_data, MutationSigmas):
                mutation_sigmas = ms_data
            else:
                mutation_sigmas = MutationSigmas()
        else:
            mutation_sigmas = MutationSigmas()

        # Metrics
        metrics_data = d.pop('metrics', {})

        # --- Extract simple fields ---
        genome_id = d.pop('id', 'unknown')
        mutation_rate = d.pop('mutation_rate', 0.1)
        params = d.pop('params', DEFAULT_PARAMS.copy())
        group_ratios = d.pop('group_ratios', {})
        latency_val = d.pop('latency', 0.0)
        evaluated = d.pop('evaluated', False)

        return cls(
            id=genome_id,
            mutation_rate=mutation_rate,
            mode=mode,
            params=params,
            group_ratios=group_ratios,
            strategies=strategies,
            weight_tensors=weight_tensors,
            mutation_sigmas=mutation_sigmas,
            fitness=fitness,
            metrics=metrics_data,
            latency=latency_val,
            evaluated=evaluated,
        )

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
        if self.mode == GenomeMode.EXPRESSION_TREE and self.strategies:
            printer(separator)
            printer("Evolved Strategies (Expression Trees):")
            for name, tree in sorted(self.strategies.items()):
                # Clean up the name for display
                display_name = name.replace("evolved_", "").replace("_", " ").title()
                printer(f"  [ {display_name} ]")
                printer(f"    {tree.to_string()}")
                printer("")  # Empty line between strategies for readability

        # 4b. Weight Tensors (Weighted Sum Mode)
        if self.mode == GenomeMode.WEIGHTED_SUM and self.weight_tensors is not None:
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

def _expression_node_from_dict(data: Dict[str, Any]) -> ExpressionNode:
    """
    Recursively reconstruct an ExpressionNode from a dictionary.

    This is the deserialization counterpart of ExpressionNode.to_dict().
    Defined here to avoid modifying the expressions module.
    """
    children = [
        _expression_node_from_dict(child_data)
        for child_data in data.get('children', [])
    ]
    return ExpressionNode(
        type=data['type'],
        value=data['value'],
        children=children,
    )


class GenomeCompiler:
    """
    Dynamically compiles a Genome's symbolic trees into executable logic.
    Decoupled from specific strategy names (works for 'movement', 'ranking', or anything new).
    """

    def compile(self, genome: Genome) -> CompiledGenome:
        """
        Converts a Genome into the specific kwargs required by SwarmRetriever.

        Returns:
            CompiledGenome with typed fields (also supports dict-like access
            for backward compatibility).
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

        # Return typed CompiledGenome (backward compatible with dict access)
        return CompiledGenome(
            agent_groups=agent_groups,
            ranking_strategies=ranking_strategies,
            params=dict(genome.params),
        )



    def _compile_tree(self, expr_tree: ExpressionNode) -> Callable[[HeuristicContext], float]:
        """
        Compiles a single expression tree into a lambda.
        """
        # _extract_features returns Set[str], sorted() gives us a deduplicated list
        required_features = sorted(self._extract_features(expr_tree))

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
            val = node.value
            # Handle HeuristicKey enum or string values
            if hasattr(val, 'value'):
                features.add(str(val.value))  # Enum's string value
            else:
                features.add(str(val))
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