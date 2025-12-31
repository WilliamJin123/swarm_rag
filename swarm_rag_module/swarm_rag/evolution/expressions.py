from typing import Callable, Union, List, Dict, Tuple
from dataclasses import dataclass, field
import math
import random


@dataclass
class ExpressionNode:
    """
    A node in the expression tree.
    Can be an operator, function, feature, or constant.
    """
    type: str  # 'op', 'func', 'feature', 'const'
    value: Union[str, float]  # operator name, function name, feature name, or constant
    children: List['ExpressionNode'] = field(default_factory=list)

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
    
    def copy(self) -> 'ExpressionNode':
        """Deep copy of the tree."""
        return ExpressionNode(
            type=self.type,
            value=self.value,
            children=[child.copy() for child in self.children]
        )

    def to_code_string(self) -> str:
        """
        Converts the expression tree into a valid Python code string.
        This is used for fast compilation into a lambda function.
        """
        if self.type == 'const':
            return str(self.value)
        
        elif self.type == 'feature':
            return f"features['{self.value}']"
        
        elif self.type == 'func':
            if not self.children:
                return "0.0"
            child_code = self.children[0].to_code_string()
            
            if self.value == 'square':
                return f"({child_code} ** 2)"
            elif self.value == 'sqrt':
                return f"math.sqrt(abs({child_code}))"
            elif self.value == 'exp':
                return f"math.exp(min({child_code}, 10))"
            elif self.value == 'log':
                return f"math.log(abs({child_code}) + 1e-8)"
            elif self.value == 'abs':
                return f"abs({child_code})"
            else:
                return child_code
        
        elif self.type == 'op':
            if len(self.children) < 2:
                return "0.0"
            left_code = self.children[0].to_code_string()
            right_code = self.children[1].to_code_string()
            
            if self.value == '+':
                return f"({left_code} + {right_code})"
            elif self.value == '-':
                return f"({left_code} - {right_code})"
            elif self.value == '*':
                return f"({left_code} * {right_code})"
            elif self.value == '/':
                # Protected division to prevent ZeroDivisionError
                return f"({left_code} / ({right_code} + 1e-8))"
            elif self.value == 'max':
                return f"max({left_code}, {right_code})"
            elif self.value == 'min':
                return f"min({left_code}, {right_code})"
            else:
                return left_code
        
        return "0.0"
    
    def compile(self) -> Callable[[Dict[str, float]], float]:
        """
        Compiles the expression tree into a fast-executing lambda function.
        This is a one-time operation that creates a reusable Python function.
        """
        code_string = self.to_code_string()
        lambda_string = f"lambda features: {code_string}"
        
        try:
            # Provide the 'math' module to the eval context so functions like sqrt can be used.
            compiled_func = eval(lambda_string, {"math": math}, {})
            return compiled_func
        except Exception as e:
            # Fallback to the slow recursive evaluation if compilation fails for any reason.
            print(f"Warning: Could not compile expression '{self.to_string()}'. Error: {e}. Using slow evaluation.")
            return lambda features: self.evaluate(features)

class ExpressionEvolution:
    """Genetic operations on expression trees."""
    
    BINARY_OPS = ['+', '-', '*', '/', 'max', 'min']
    UNARY_FUNCS = ['square', 'sqrt', 'exp', 'log', 'abs']
    
    @staticmethod
    def random_tree(
        features: List[str],
        max_depth: int = 4,
        leaf_prob: float = 0.3
    ) -> ExpressionNode:
        """Generate a random expression tree."""

        # Base case: create leaf
        if max_depth <= 1 or random.random() < leaf_prob:
            if random.random() < 0.7:  # Feature
                return ExpressionNode('feature', random.choice(features))
            else:  # Constant
                return ExpressionNode('const', random.uniform(0, 1))
        
        # Recursive case: create operator or function
        if random.random() < 0.7:  # Binary operator
            op = random.choice(ExpressionEvolution.BINARY_OPS)
            left = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            right = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            return ExpressionNode('op', op, [left, right])
        else:  # Unary function
            func = random.choice(ExpressionEvolution.UNARY_FUNCS)
            child = ExpressionEvolution.random_tree(features, max_depth - 1, leaf_prob)
            return ExpressionNode('func', func, [child])
    
    @staticmethod
    def mutate_tree(tree: ExpressionNode, features: List[str], mutation_rate: float = 0.2) -> ExpressionNode:
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
    def crossover_trees(tree1: ExpressionNode, tree2: ExpressionNode) -> Tuple[ExpressionNode, ExpressionNode]:
        """
        Subtree crossover between two expression trees.
        """
        t1 = tree1.copy()
        t2 = tree2.copy()
        
        def get_all_nodes(node: ExpressionNode):
            """Returns a list of all nodes in the tree."""
            return [node] + sum([get_all_nodes(c) for c in node.children], [])
        
        nodes1 = get_all_nodes(t1)
        nodes2 = get_all_nodes(t2)
        
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
  