import importlib.util

def fail_on_missing_imports(modules: list[str], extra_name: str = None):
    """
    Checks if a list of modules can be imported. 
    If not, raises an ImportError with the specific pip command to fix it.
    
    Args:
        modules: List of python import names (e.g. ['torch', 'stark_qa'])
        extra_name: The name of the extra in pyproject.toml (e.g. 'stark')
    """
    missing = [
        m for m in modules
        if importlib.util.find_spec(m) is None
    ]
    if not missing:
        return

    if extra_name is not None:
        msg = (
            f"Missing required dependencies: {', '.join(missing)}.\n"
            f"Please install them by running:\n\n"
            f"    pip install \"swarm_rag[{extra_name}]\""
        )
    else:
        msg = (
            f"Missing required dependencies: {', '.join(missing)}.\n"
            f"Please install them by running:\n\n"
            f"    pip install {' '.join(missing)}"
        )

    raise ImportError(msg) from None