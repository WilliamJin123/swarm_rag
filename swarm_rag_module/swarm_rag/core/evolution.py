import numpy as np
from typing import List, Dict, Callable, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import operator
import math
import random

from .heuristics import HeuristicContext, Heuristics
from .swarm_retriever import SwarmRetriever
from ..interfaces.base import GraphStore, VectorStore, EmbeddingProvider

# Time complexity: O(population × generations × evaluation)

@dataclass
class ExprNode:
    """
    A node in the expression tree.
    Can be an operator, function, feature, or constant.
    """
    type: str  # 'op', 'func', 'feature', 'const'
    value: Union[str, float]  # operator name, function name, feature name, or constant
    children: List['ExprNode'] = field(default_factory=list)

    def evaluate(self, features: Dict[str, float]) -> float:
        """Recursively evaluate the expression tree."""
        if self.type == 'const':
            return float(self.value)
        
        elif self.type == 'feature':
            return features.get(self.value, 0.0)
        
        elif self.type == 'func':
            if not self.children:
                return 0.0
            arg = self.children[0].evaluate(features)
            
            if self.value == 'square':
                return arg ** 2
            elif self.value == 'sqrt':
                return math.sqrt(abs(arg))
            elif self.value == 'exp':
                return math.exp(min(arg, 10))  # Prevent overflow
            elif self.value == 'log':
                return math.log(abs(arg) + 1e-8)
            elif self.value == 'abs':
                return abs(arg)
            else:
                return arg
        
        elif self.type == 'op':
            if len(self.children) < 2:
                return 0.0
            left = self.children[0].evaluate(features)
            right = self.children[1].evaluate(features)
            
            if self.value == '+':
                return left + right
            elif self.value == '-':
                return left - right
            elif self.value == '*':
                return left * right
            elif self.value == '/':
                return left / (right + 1e-8)
            elif self.value == 'max':
                return max(left, right)
            elif self.value == 'min':
                return min(left, right)
            else:
                return left
        
        return 0.0
    
    def depth(self) -> int:
        """Calculate tree depth (complexity metric)."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """Count total nodes (complexity metric)."""
        return 1 + sum(child.size() for child in self.children)
    
    def to_string(self) -> str:
        """Convert to readable mathematical expression."""
        if self.type == 'const':
            return f"{self.value:.3f}"
        elif self.type == 'feature':
            return str(self.value)
        elif self.type == 'func':
            if self.children:
                return f"{self.value}({self.children[0].to_string()})"
            return self.value
        elif self.type == 'op':
            if len(self.children) >= 2:
                left = self.children[0].to_string()
                right = self.children[1].to_string()
                return f"({left} {self.value} {right})"
            return "?"
        return "?"
    
    def copy(self) -> 'ExprNode':
        """Deep copy of the tree."""
        return ExprNode(
            type=self.type,
            value=self.value,
            children=[child.copy() for child in self.children]
        )
    
@dataclass 
class Genome:
    """
    A complete retrieval strategy with BOTH hyperparameters
    and expression trees in one genome.
    """

    # Hyperparameters
    n_agents: int
    steps: int
    decay: float
    initial_pool_size: int
    start_subset: int
    
    # Expression trees (the evolved heuristics)
    movement_expr: ExprNode
    ranking_expr: ExprNode
    deposit_expr: ExprNode

    # Available features for each strategy (TO CHANGE)
    available_movement_features: List[str] = field(default_factory=lambda: [
        'semantic', 'centrality', 'diversity', 'jitter'
    ])
    available_ranking_features: List[str] = field(default_factory=lambda: [
        'semantic', 'votes', 'centrality'
    ])
    available_deposit_features: List[str] = field(default_factory=lambda: [
        'flat', 'semantic', 'hub', 'explorer'
    ])

    # Performance metrics
    fitness: float = 0.0
    recall_at_k: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    latency_ms: float = 0.0

    def complexity(self) -> int:
        """Total complexity across all expressions."""
        return (
            self.movement_expr.size() +
            self.ranking_expr.size() +
            self.deposit_expr.size()
        )
    
class ExpressionEvolution:
    """Genetic operations on expression trees."""
    
    BINARY_OPS = ['+', '-', '*', '/', 'max', 'min']
    UNARY_FUNCS = ['square', 'sqrt', 'exp', 'log', 'abs']
    
    @staticmethod
    def random_tree(
        features: List[str],
        max_depth: int = 4,
        leaf_prob: float = 0.3
    ) -> ExprNode:
        """Generate a random expression tree."""

        # Base case: create leaf
        if max_depth <= 1 or random.random() < leaf_prob:
            if random.random() < 0.7:  # Feature
                return ExprNode('feature', random.choice(features))
            else:  # Constant
                return ExprNode('const', random.uniform(0, 1))
        
        # Recursive case: create operator or function
        if random.random() < 0.7:  # Binary operator
            op = random.choice(ExpressionEvolution.BINARY_OPS)
            left = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            right = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            return ExprNode('op', op, [left, right])
        else:  # Unary function
            func = random.choice(ExpressionEvolution.UNARY_FUNCS)
            child = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            return ExprNode('func', func, [child])
    
    @staticmethod
    def mutate_tree(tree: ExprNode, features: List[str], mutation_rate: float = 0.2) -> ExprNode:
        """
        Mutate an expression tree.
        Can change operators, constants, features, or replace subtrees.
        """
        tree = tree.copy()
        
        if random.random() < mutation_rate:
            # Mutation types
            mut_type = random.choice(['node', 'subtree', 'constant'])
            
            if mut_type == 'node':
                # Change the node value
                if tree.type == 'op':
                    tree.value = random.choice(ExpressionEvolution.BINARY_OPS)
                elif tree.type == 'func':
                    tree.value = random.choice(ExpressionEvolution.UNARY_FUNCS)
                elif tree.type == 'feature':
                    tree.value = random.choice(features)
                elif tree.type == 'const':
                    tree.value += random.gauss(0, 0.1)
                    tree.value = max(0, min(2, tree.value))  # Clamp
            
            elif mut_type == 'subtree':
                # Replace with random subtree
                return ExpressionEvolution.random_tree(features, max_depth=3)
            
            elif mut_type == 'constant':
                # Add/multiply by constant
                const_node = ExprNode('const', random.uniform(0.5, 1.5))
                op = random.choice(['+', '*'])
                return ExprNode('op', op, [tree.copy(), const_node])
        
        # Recursively mutate children
        tree.children = [
            ExpressionEvolution.mutate_tree(child, features, mutation_rate)
            for child in tree.children
        ]
        
        return tree
    
    @staticmethod
    def crossover_trees(tree1: ExprNode, tree2: ExprNode) -> Tuple[ExprNode, ExprNode]:
        """
        Subtree crossover between two expression trees.
        Swaps random subtrees between parents.
        """
        t1 = tree1.copy()
        t2 = tree2.copy()
        
        # Get all subtrees
        def get_subtrees(node: ExprNode, depth: int = 0) -> List[Tuple[ExprNode, int]]:
            result = [(node, depth)]
            for child in node.children:
                result.extend(get_subtrees(child, depth + 1))
            return result
        
        subtrees1 = get_subtrees(t1)
        subtrees2 = get_subtrees(t2)
        
        if len(subtrees1) > 1 and len(subtrees2) > 1:
            # Pick random subtrees (avoid root for diversity)
            st1, _ = random.choice(subtrees1[1:] if len(subtrees1) > 1 else subtrees1)
            st2, _ = random.choice(subtrees2[1:] if len(subtrees2) > 1 else subtrees2)
            
            # Swap them
            st1.type, st2.type = st2.type, st1.type
            st1.value, st2.value = st2.value, st1.value
            st1.children, st2.children = st2.children, st1.children
        
        return t1, t2
    
    @staticmethod
    def simplify_tree(tree: ExprNode, max_size: int = 30) -> ExprNode:
        """
        Simplify tree if too complex.
        Prunes to stay under size limit.
        """
        if tree.size() <= max_size:
            return tree
        
        # If too complex, replace with simpler version
        # Strategy: keep the main structure but simplify deepest subtrees
        def prune(node: ExprNode, depth: int = 0) -> ExprNode:
            if depth > 5:  # Too deep, replace with leaf
                if node.type == 'feature':
                    return node
                return ExprNode('const', 0.5)
            
            node.children = [prune(child, depth + 1) for child in node.children]
            return node
        
        return prune(tree.copy())
    
class EvolutionaryOptimizer:
    """
    Evolves hyperparameters and expression trees together.
    Single population / evolution loop for both (might change in the future)
    """

    def __init__(
        self,
        retriever,
        queries: List[str],
        ground_truth: List[List[int]],
        population_size: int = 30,
        n_generations: int = 20,
        top_k: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.6,
        n_workers: int = 4,
        max_expr_size: int = 25
    ):
        self.retriever = retriever
        self.queries = queries
        self.ground_truth = ground_truth
        self.population_size = population_size
        self.n_generations = n_generations
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_workers = n_workers
        self.max_expr_size = max_expr_size
        
        # Feature to function mapping for evaluation
        # TO CHANGE: add more customizability
        self.feature_funcs = {
            'semantic': Heuristics.semantic_similarity,
            'centrality': Heuristics.node_centrality,
            'diversity': Heuristics.pheromone_repulsion,
            'jitter': Heuristics.random_jitter,
            'votes': Heuristics.percentage_visited,
            'hub': Heuristics.node_centrality,
            'explorer': Heuristics.pheromone_repulsion,
            'flat': Heuristics.deposit_flat,
        }
        
        self.population_history = []
        self.best_genome = None


