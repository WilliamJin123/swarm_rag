import ast
from typing import Union, List, Dict, Any
from ..types.expressions import ExpressionNode

class ExpressionParser:
    """
    Parses string representations of mathematical expressions into ExpressionNode trees.
    Uses Python's AST to ensure safe and correct parsing.
    """
    
    # Allowed functions
    ALLOWED_FUNCS = {
        'square', 'sqrt', 'exp', 'log', 'abs', 'sin', 'tanh', 'sigmoid',
        'max', 'min' # Treated as ops in ExpressionNode but parsed as calls
    }
    
    # Allowed operators map (AST -> String)
    OP_MAP = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
    }

    @staticmethod
    def parse(expression: str) -> ExpressionNode:
        """
        Parses a string expression into an ExpressionNode tree.
        Example: "semantic_similarity * 0.8 + pagerank * 0.2"
        """
        if not expression or not isinstance(expression, str):
            raise ValueError(f"Invalid expression: {expression}")
            
        try:
            tree = ast.parse(expression, mode='eval')
            return ExpressionParser._convert(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {expression}") from e

    @staticmethod
    def _convert(node: ast.AST) -> ExpressionNode:
        """
        Recursive converter from AST nodes to ExpressionNodes.
        """
        
        # 1. Binary Operators (A + B)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ExpressionParser.OP_MAP:
                raise ValueError(f"Unsupported operator: {op_type}")
                
            op_str = ExpressionParser.OP_MAP[op_type]
            left = ExpressionParser._convert(node.left)
            right = ExpressionParser._convert(node.right)
            
            return ExpressionNode(type='op', value=op_str, children=[left, right])
            
        # 2. Function Calls (sqrt(A), max(A, B))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls allowed (e.g. sin(x))")
                
            func_name = node.func.id
            if func_name not in ExpressionParser.ALLOWED_FUNCS:
                raise ValueError(f"Unknown function: {func_name}")
            
            args = [ExpressionParser._convert(arg) for arg in node.args]
            
            # Special handling for max/min which are 'op' in ExpressionNode but calls in AST
            if func_name in ['max', 'min']:
                if len(args) != 2:
                    raise ValueError(f"'{func_name}' requires exactly 2 arguments")
                return ExpressionNode(type='op', value=func_name, children=args)
            
            # Standard unary functions
            if len(args) != 1:
                raise ValueError(f"Function '{func_name}' requires exactly 1 argument")
                
            return ExpressionNode(type='func', value=func_name, children=args)
            
        # 3. Names/Variables (Features)
        elif isinstance(node, ast.Name):
            return ExpressionNode(type='feature', value=node.id)
            
        # 4. Constants/Numbers
        elif isinstance(node, ast.Constant): # Python 3.8+
            if isinstance(node.value, (int, float)):
                return ExpressionNode(type='const', value=float(node.value))
            else:
                raise ValueError(f"Unsupported constant type: {type(node.value)}")
                
        elif isinstance(node, ast.Num): # Python < 3.8 compatibility
             return ExpressionNode(type='const', value=float(node.n))

        # 5. Unary Operators (Only -A supported usually)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = ExpressionParser._convert(node.operand)
                # Represent -A as 0 - A or specific handling?
                # ExpressionNode logic supports unary op '-' if 1 child
                return ExpressionNode(type='op', value='-', children=[operand])
            else:
                 raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")
