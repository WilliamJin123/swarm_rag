import dataclasses
from typing import Any, List, Callable
import random

from swarm_rag.evolution.evolution_context import EvolutionContext
from swarm_rag.evolution.expressions import ExpressionEvolution, ExpressionNode
from .genome import Genome

class GeneticRegistry:
    _selection_registry = {}
    _crossover_registry = {}
    _mutation_registry = {}

    @classmethod
    def register_selection(cls, name: str):
        def decorator(fn):
            cls._selection_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_crossover(cls, name: str):
        def decorator(fn):
            cls._crossover_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def register_mutation(cls, name: str):
        def decorator(fn):
            cls._mutation_registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_selection(cls, name: str) -> Callable:
        return cls._selection_registry[name]

    @classmethod
    def get_crossover(cls, name: str) -> Callable:
        return cls._crossover_registry[name]

    @classmethod
    def get_mutation(cls, name: str) -> Callable:
        return cls._mutation_registry[name]
    
    @classmethod
    def all_selection(cls) -> dict[str, Callable]:
        """
        Return the complete selection registry.
        """
        return cls._selection_registry

    @classmethod
    def all_crossover(cls) -> dict[str, Callable]:
        """
        Return the complete crossover registry.
        """
        return cls._crossover_registry

    @classmethod
    def all_mutation(cls) -> dict[str, Callable]:
        """
        Return the complete mutation registry.
        """
        return cls._mutation_registry
    
    @classmethod
    def all(cls) -> dict[str, Callable]:
        return {
            **cls._selection_registry,
            **cls._crossover_registry,
            **cls._mutation_registry,
        }

class GeneticStrategies:
    """
    Standard library of genetic operators.
    """

    # --- HELPER: Introspection ---
    @staticmethod
    def _get_mutable_fields(obj: Any):
        """
        Dynamically categorize fields of a dataclass for generic mutation/crossover.
        Returns: (numeric_fields, tree_fields)
        """
        numeric_fields = []
        tree_fields = []
        
        # 'fields' gives us metadata about every field in the dataclass
        for f in dataclasses.fields(obj):
            # 1. Numerics (int/float) - excluding system fields like 'fitness'
            if f.type in [int, float] and f.name not in ['fitness', 'latency_ms']:
                numeric_fields.append(f.name)
            
            # 2. Expression Trees - explicit type check preferred, or name convention
            # We check if the type hint is ExpressionNode
            elif f.type == ExpressionNode: 
                tree_fields.append(f.name)
                
        return numeric_fields, tree_fields

    # --- SELECTION ---

    @staticmethod
    @GeneticRegistry.register_selection("tournament")
    def tournament_selection(ctx: EvolutionContext) -> Genome:
        k = ctx.config.selection_k
        candidates = random.sample(ctx.population, k)
        return max(candidates, key=lambda g: g.fitness)

    @staticmethod
    @GeneticRegistry.register_selection("roulette")
    def roulette_selection(ctx: EvolutionContext) -> Genome:
        total_fit = sum(g.fitness for g in ctx.population)
        pick = random.uniform(0, total_fit)
        current = 0
        for g in ctx.population:
            current += g.fitness
            if current > pick:
                return g
        return ctx.population[-1]

    # --- CROSSOVER ---

    @staticmethod
    @GeneticRegistry.register_crossover("uniform_parameter_mix")
    def uniform_parameter_mix(parent1: Genome, parent2: Genome, ctx: EvolutionContext) -> Genome:
        """Mixes traits 50/50. """
        # 1. Create a shallow copy of Parent 1 using its exact class
        child = dataclasses.replace(parent1)

        # 2. Introspect fields
        numerics, trees = GeneticStrategies._get_mutable_fields(parent1)
        
        # 3. Mix Numerics
        for field in numerics:
            if random.random() > 0.5:
                # Take value from Parent 2
                val = getattr(parent2, field)
                setattr(child, field, val)
                
        # 4. Mix Trees (With .copy())
        for field in trees:
            if random.random() > 0.5:
                # Take from Parent 2
                original_tree = getattr(parent2, field)
            else:
                # Take from Parent 1 (already in 'child', but we need a fresh copy)
                original_tree = getattr(parent1, field)
            
            # We MUST copy the tree object, otherwise the child shares memory with parent
            setattr(child, field, original_tree.copy())

        return child

    # --- MUTATION ---

    @staticmethod
    @GeneticRegistry.register_mutation("expression_tree_mutation")
    def expression_tree_mutation(genome: Genome, ctx: EvolutionContext) -> Genome:
        rate = ctx.config.mutation_rate
        numerics, trees = GeneticStrategies._get_mutable_fields(genome)

        # 1. Generic Numeric Jitter
        # Picks ANY numeric field (n_agents, decay, or new extension fields)
        if random.random() < rate and numerics:
            target_field = random.choice(numerics)
            current_val = getattr(genome, target_field)
            
            # Apply Jitter
            new_val = current_val * random.uniform(0.8, 1.2)
            
            # Type Safety: If it was int, keep it int
            if isinstance(current_val, int):
                # Ensure generic constraints (e.g., don't go below 1 for counts)
                new_val = max(1, int(new_val))
            else:
                # Clamp floats (example: keeping decay between 0-1 is common)
                if 0.0 <= current_val <= 1.0:
                    new_val = max(0.001, min(0.999, new_val))
            
            setattr(genome, target_field, new_val)

        # 2. Generic Tree Mutation
        # Dynamically finds the matching 'available_X_features' list for 'X_expr'
        for tree_field in trees:
            if random.random() < rate:
                feature_list = ctx.expression_features.get(tree_field, [])
                current_tree = getattr(genome, tree_field)
                mutated_tree = ExpressionEvolution.mutate_tree(
                    current_tree,
                    features=feature_list,
                    mutation_rate=rate
                )
                setattr(genome, tree_field, mutated_tree)

        return genome
    
