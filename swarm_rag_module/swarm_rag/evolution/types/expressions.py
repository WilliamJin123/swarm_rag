from typing import Any, Callable, Union, List, Dict, Tuple
from dataclasses import dataclass, field
import math
import random
import torch

@dataclass
class ExpressionNode:
    """
    A node in the expression tree.
    Can be an operator, function, feature, or constant.
    """
    type: str  # 'op', 'func', 'feature', 'const'
    value: Union[str, float]  # operator name, function name, feature name, or constant
    children: List['ExpressionNode'] = field(default_factory=list)

    def evaluate(self, features: Dict[str, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Recursively evaluate the expression tree."""
        if self.type == 'const':
            return float(self.value)

        elif self.type == 'feature':
            # Returns a Tensor (N,) or scalar depending on input
            val = features.get(self.value, 0.0)
            return val

        elif self.type == 'func':
            if not self.children:
                return 0.0
            arg = self.children[0].evaluate(features)
            arg_tensor = torch.as_tensor(arg) if not isinstance(arg, torch.Tensor) else arg

            if self.value == 'square':
                return torch.square(arg_tensor)
            elif self.value == 'sqrt':
                return torch.sqrt(torch.abs(arg_tensor))
            elif self.value == 'exp':
                return torch.exp(torch.minimum(arg_tensor, torch.tensor(10.0)))  # Prevent overflow
            elif self.value == 'log':
                return torch.log(torch.abs(arg_tensor) + 1e-8)
            elif self.value == 'abs':
                return torch.abs(arg_tensor)
            elif self.value == 'sin':
                return torch.sin(arg_tensor)
            elif self.value == 'tanh':
                return torch.tanh(arg_tensor)
            elif self.value == 'sigmoid':
                # Stable sigmoid
                return 1.0 / (1.0 + torch.exp(-torch.clamp(arg_tensor, -10, 10)))

            return 0.0

        elif self.type == 'op':
            if not self.children:
                return 0.0
            left = self.children[0].evaluate(features)
            if len(self.children) == 1:
                if self.value == '-':
                    return -left
                return left

            right = self.children[1].evaluate(features)

            if self.value == '+':
                return left + right
            elif self.value == '-':
                return left - right
            elif self.value == '*':
                return left * right
            elif self.value == '/':
                right_tensor = torch.as_tensor(right) if not isinstance(right, torch.Tensor) else right
                denom = torch.where(right_tensor == 0, torch.tensor(1e-8), right_tensor) if isinstance(right_tensor, torch.Tensor) else (right if right != 0 else 1e-8)
                return left / denom
            elif self.value == 'max':
                left_tensor = left if isinstance(left, torch.Tensor) else torch.tensor(float(left))
                right_tensor = right if isinstance(right, torch.Tensor) else torch.tensor(float(right))
                return torch.maximum(left_tensor, right_tensor)
            elif self.value == 'min':
                left_tensor = left if isinstance(left, torch.Tensor) else torch.tensor(float(left))
                right_tensor = right if isinstance(right, torch.Tensor) else torch.tensor(float(right))
                return torch.minimum(left_tensor, right_tensor)

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
        """
        Convert to readable mathematical expression.
        Uses operator precedence to minimize unnecessary parentheses.
        """
        PRECEDENCE = {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            'max': 0, 'min': 0
        }
        
        def _get_precedence(node: 'ExpressionNode') -> int:
            if node.type == 'op':
                return PRECEDENCE.get(node.value, 99)
            elif node.type == 'func':
                return 100 # Function calls bind tightly
            return 100 # Consts/Features bind tightest

        if self.type == 'const':
            return f"{self.value:.3f}"
        
        elif self.type == 'feature':
            val = self.value
            if hasattr(val, 'value'): 
                return str(val.value)
            return str(val)
        
        elif self.type == 'func':
            if self.children:
                child_str = self.children[0].to_string()
                return f"{self.value}({child_str})"
            return self.value
        
        elif self.type == 'op':
            if len(self.children) >= 2:
                left = self.children[0]
                right = self.children[1]
                
                left_str = left.to_string()
                right_str = right.to_string()
                
                my_prec = _get_precedence(self)
                left_prec = _get_precedence(left)
                right_prec = _get_precedence(right)
                
                # Wrap left if it has lower precedence
                if left_prec < my_prec:
                    left_str = f"({left_str})"
                
                # Wrap right if it has lower precedence OR equal precedence (for left-associativity)
                # e.g. a - (b - c) needs parens, but (a - b) - c doesn't
                # Division and Subtraction are non-associative in a way that requires care.
                # simpler rule: if right child has same precedence, wrap it (safe for - and /)
                if right_prec < my_prec or (right_prec == my_prec and self.value in ['-', '/']):
                     right_str = f"({right_str})"

                # Special formatting for min/max
                if self.value in ['max', 'min']:
                     return f"{self.value}({left_str}, {right_str})"

                return f"{left_str} {self.value} {right_str}"
            return "?"
        return "?"
    
    def copy(self) -> 'ExpressionNode':
        """Deep copy of the tree."""
        return ExpressionNode(
            type=self.type,
            value=self.value,
            children=[child.copy() for child in self.children]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the ExpressionNode into a plain Python dictionary.
        
        Returns:
            A dictionary representation of the node and its children.
        """
        val = self.value
        if hasattr(val, 'value'): # Handle Enums
            val = val.value

        return {
            "type": self.type,
            "value": val,
            "children": [child.to_dict() for child in self.children]
        }

    def to_code_string(self) -> str:
        """
        Converts the expression tree into a valid Python code string.
        This is used for fast compilation into a lambda function.
        """
        if self.type == 'const':
            return f"torch.tensor({self.value})"

        elif self.type == 'feature':
            # e.g., "node.degree" -> "node_degree"
            val = self.value.value if hasattr(self.value, 'value') else str(self.value)
            return val.replace(".", "_").replace(" ", "_").replace("-", "_")

        elif self.type == 'func':
            if not self.children:
                return "torch.tensor(0.0)"
            child_code = self.children[0].to_code_string()

            if self.value == 'square':
                return f"(torch.square({child_code}))"
            elif self.value == 'sqrt':
                return f"torch.sqrt(torch.abs({child_code}))"
            elif self.value == 'exp':
                return f"torch.exp(torch.minimum({child_code}, torch.tensor(10.0)))"
            elif self.value == 'log':
                return f"torch.log(torch.abs({child_code}) + 1e-8)"
            elif self.value == 'abs':
                return f"torch.abs({child_code})"
            elif self.value == 'sin':
                return f"torch.sin({child_code})"
            elif self.value == 'tanh':
                return f"torch.tanh({child_code})"
            elif self.value == 'sigmoid':
                return f"(1.0 / (1.0 + torch.exp(-torch.clamp({child_code}, -10, 10))))"
            else:
                return child_code

        elif self.type == 'op':
            if len(self.children) < 2:
                if len(self.children) == 1 and self.value == '-':
                    return f"(-{self.children[0].to_code_string()})"
                return "torch.tensor(0.0)"

            left_code = self.children[0].to_code_string()
            right_code = self.children[1].to_code_string()

            if self.value == '+':
                return f"({left_code} + {right_code})"
            elif self.value == '-':
                return f"({left_code} - {right_code})"
            elif self.value == '*':
                return f"({left_code} * {right_code})"
            elif self.value == '/':
                return f"({left_code} / ({right_code} + 1e-8))"
            elif self.value == 'max':
                return f"torch.maximum(torch.as_tensor({left_code}), torch.as_tensor({right_code}))"
            elif self.value == 'min':
                return f"torch.minimum(torch.as_tensor({left_code}), torch.as_tensor({right_code}))"
            else:
                return left_code

        return "torch.tensor(0.0)"
    
    def compile(self, arg_names: List[str]) -> Callable:
        """
        Compiles the tree into a function that accepts specific positional arguments.
        Args:
            arg_names: A list of strings defining the function signature.
                       e.g., ['degree', 'pheromone']
        """
        code_string = self.to_code_string()
        args_str = ", ".join(arg_names)
        lambda_string = f"lambda {args_str}: {code_string}"

        try:
            # Inject torch
            return eval(lambda_string, {"torch": torch}, {})
        except Exception as e:
            print(f"Compilation Failed: {lambda_string}")
            raise e
class ExpressionEvolution:
    """Genetic operations on expression trees."""
    
    BINARY_OPS = ['+', '-', '*', '/', 'max', 'min']
    UNARY_FUNCS = ['square', 'sqrt', 'exp', 'log', 'abs', 'sin', 'tanh', 'sigmoid']
    
    @staticmethod
    def random_tree(
        features: List[str],
        max_depth: int = 4,
        method: str = 'grow'
    ) -> ExpressionNode:
        """
        Generate a random expression tree.
        
        Args:
            features: List of available feature names for leaf nodes.
            max_depth: The maximum depth of the tree.
            method: 'grow' or 'full'.
                    - 'grow': Randomly chooses between a leaf and a function at each level.
                    - 'full': Ensures every path to a leaf is at max_depth.
        """

        # Base case: create leaf
        if max_depth == 0:
            if random.random() < 0.7:  # Feature
                return ExpressionNode('feature', random.choice(features))
            else:
                return ExpressionNode('const', random.uniform(0, 1))
        
        # Recursive case: create operator or function
        if method == 'full' or (method == 'grow' and random.random() < 0.9):
            # Choose a binary operator
            op = random.choice(ExpressionEvolution.BINARY_OPS)
            left = ExpressionEvolution.random_tree(features, max_depth - 1, method)
            right = ExpressionEvolution.random_tree(features, max_depth - 1, method)
            return ExpressionNode('op', op, [left, right])
        else: # 'grow' method and we decided to create a leaf
            if random.random() < 0.7:
                return ExpressionNode('feature', random.choice(features))
            else:
                return ExpressionNode('const', random.uniform(0, 1))
    
    @staticmethod
    def generate_ramped_half_and_half(
        features: List[str],
        population_size: int,
        max_depth: int = 6
    ) -> List[ExpressionNode]:
        """
        Generates an initial population using the ramped half-and-half method.
        """
        if population_size <= 0:
            return []
            
        population = []
        # Depths from 2 to max_depth
        depths = list(range(2, max_depth + 1))

        # We need to distribute population across these depths
        # and for each depth, split 50/50 between 'grow' and 'full'
        
        # Calculate how many trees per (depth, method) pair
        # There are len(depths) * 2 groups
        n_groups = len(depths) * 2
        per_group = population_size // n_groups
        
        # 1. Systematic Generation
        for depth in depths:
            # Half Grow
            for _ in range(per_group):
                population.append(ExpressionEvolution.random_tree(features, depth, 'grow'))
            # Half Full
            for _ in range(per_group):
                population.append(ExpressionEvolution.random_tree(features, depth, 'full'))
        
        # 2. Fill Remainder (if population_size didn't divide evenly)
        while len(population) < population_size:
            depth = random.choice(depths)
            method = random.choice(['grow', 'full'])
            population.append(ExpressionEvolution.random_tree(features, depth, method))
            
        random.shuffle(population)
        return population

    @staticmethod
    def mutate_tree(
        tree: ExpressionNode, 
        features: List[str], 
        mutation_rate: float = 0.2,
        inplace: bool = False
    ) -> ExpressionNode:
        """
        Mutate an expression tree.
        Can change operators, constants, features, or replace subtrees.
        """
        if not inplace:
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
                const_node = ExpressionNode('const', random.uniform(0.5, 1.5))
                op = random.choice(['+', '*'])
                return ExpressionNode('op', op, [tree.copy(), const_node])
        
        # Recursively mutate children
        tree.children = [
            ExpressionEvolution.mutate_tree(child, features, mutation_rate)
            for child in tree.children
        ]
        
        return tree
    
    @staticmethod
    def get_all_nodes(node: ExpressionNode) -> List[ExpressionNode]:
        """Returns a list of all nodes in the tree."""
        return [node] + sum([ExpressionEvolution.get_all_nodes(c) for c in node.children], [])

    @staticmethod
    def crossover_trees(tree1: ExpressionNode, tree2: ExpressionNode) -> Tuple[ExpressionNode, ExpressionNode]:
        """
        Subtree crossover between two expression trees.
        """
        t1 = tree1.copy()
        t2 = tree2.copy()
        
        nodes1 = ExpressionEvolution.get_all_nodes(t1)
        nodes2 = ExpressionEvolution.get_all_nodes(t2)
        
        if len(nodes1) > 1 and len(nodes2) > 1:
            # Pick random nodes to swap (avoid root to maintain structure)
            node1_to_swap = random.choice(nodes1[1:])
            node2_to_swap = random.choice(nodes2[1:])

            # Find the parents of the selected nodes
            def find_parent(root, target_node):
                if target_node in root.children:
                    return root
                for child in root.children:
                    parent = find_parent(child, target_node)
                    if parent:
                        return parent
                return None

            parent1 = find_parent(t1, node1_to_swap)
            parent2 = find_parent(t2, node2_to_swap)

            if parent1 and parent2:
                # Find the index of the child in each parent
                idx1 = parent1.children.index(node1_to_swap)
                idx2 = parent2.children.index(node2_to_swap)

                # Perform the swap
                parent1.children[idx1], parent2.children[idx2] = node2_to_swap, node1_to_swap

        return t1, t2
    
    @staticmethod
    def simplify_tree(tree: ExpressionNode, max_size: int = 30) -> ExpressionNode:
        """
        Simplify tree if too complex.
        Prunes to stay under size limit.
        """
        if tree.size() <= max_size:
            return tree
        
        # If too complex, replace with simpler version
        # Strategy: keep the main structure but simplify deepest subtrees
        def prune(node: ExpressionNode, depth: int = 0) -> ExpressionNode:
            if depth > 5:  # Too deep, replace with leaf
                if node.type == 'feature':
                    return node
                return ExpressionNode('const', 0.5)
            
            node.children = [prune(child, depth + 1) for child in node.children]
            return node
        
        return prune(tree.copy())
  