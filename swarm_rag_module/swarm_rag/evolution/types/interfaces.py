
from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class RetrievalBackend(Protocol):
    """
    Protocol defining the contract for any system that can be optimized 
    by this Evolution Engine.
    """
    def retrieve_batch(self, queries: List[str], **kwargs) -> List[List[Any]]:
        """
        Must accept queries and arbitrary keyword arguments (the genome),
        and return a list of results (IDs, Nodes, or Strings) matching the ground truth.
        """
        ...
