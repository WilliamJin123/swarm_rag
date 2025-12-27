import importlib.util
from threading import Lock
from collections import OrderedDict

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

class LRUCache:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data = OrderedDict()
        self.lock = Lock()

    def get(self, key):
         with self.lock:
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]
    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            if len(self.data) > self.maxsize:
                self.data.popitem(last=False)
