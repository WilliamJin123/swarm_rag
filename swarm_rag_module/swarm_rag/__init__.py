from .core.swarm_retriever import SwarmRetriever
from .interfaces.abstract_classes import VectorStore, GraphStore, EmbeddingProvider

from .integrations.stark import (
    StarkGraphAdapter,
    StarkVectorStore,
    StarkPreComputedEmbeddingHandler,
)
from .integrations.torch_vector_store import TorchVectorStore
from .integrations.torch_graph_store import TorchGraphStore

# GPU utilities
from .utils.device import get_device

__all__ = [
    'SwarmRetriever',
    'VectorStore',
    'GraphStore',
    'EmbeddingProvider',
    # STaRK-specific
    'StarkGraphAdapter',
    'StarkVectorStore',
    'StarkPreComputedEmbeddingHandler',
    # Generic torch stores
    'TorchVectorStore',
    'TorchGraphStore',
    'get_device',
]
