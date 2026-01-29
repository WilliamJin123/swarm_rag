
"""
PyTorch-Native Graph Store

Stores graph adjacency in CSR (Compressed Sparse Row) format using torch tensors.
Works on both CPU and GPU - device is configurable.

Memory footprint: ~65MB for prime dataset (129K nodes, 16M edges)
vs ~17GB for dense format (99.6% reduction)
"""

from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import scipy.sparse as sp

# Assuming these exist in your project structure
from ..interfaces.abstract_classes import GraphStore
from ..interfaces.types import TorchDeviceStr
from ..utils.device import get_device

import logging

logger = logging.getLogger(__name__)


class TorchGraphStore(GraphStore):
    """
    PyTorch-native graph store using CSR (Compressed Sparse Row) format.

    Stores adjacency in sparse format for massive memory savings while
    maintaining fast batch neighbor lookups via vectorized operations.

    Key features:
    - CSR format: O(nodes + edges) memory vs O(nodes * max_degree) for dense
    - Batch neighbor lookup via vectorized gather operations
    - Dynamic max_degree per batch (not fixed)
    - Works on both CPU and GPU

    Usage:
        store = TorchGraphStore.from_adjacency_dict(adj_dict, device="cuda")
        store = TorchGraphStore.from_adjacency_dict(adj_dict, device="cpu")
        store = TorchGraphStore.from_csr(csr_matrix)
        neighbors, mask = store.get_neighbors_batch(torch.tensor([1, 2, 3]))
    """

    def __init__(
        self,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        degree_tensor: torch.Tensor,
        n_nodes: int,
        avg_degree: float,
        device: Optional[TorchDeviceStr] = None
    ):
        """
        Initialize graph store with CSR components.

        Args:
            crow_indices: Row pointers tensor of shape (n_nodes + 1,)
            col_indices: Neighbor IDs tensor of shape (total_edges,)
            degree_tensor: Tensor of shape (n_nodes,) containing node degrees
            n_nodes: Total number of nodes in graph
            avg_degree: Average degree of the graph
            device: Target device ("cuda" or "cpu")
        """
        self._device = device or get_device()
        self._n_nodes = n_nodes
        self._avg_degree = avg_degree

        # Store CSR components on device
        self._crow_indices = crow_indices.to(device=self._device, dtype=torch.long)
        self._col_indices = col_indices.to(device=self._device, dtype=torch.long)
        self._degrees = degree_tensor.to(device=self._device, dtype=torch.int32)

        self._max_degree = int(self._degrees.max().item()) if n_nodes > 0 else 0

        # Compute memory usage
        crow_mem = self._crow_indices.numel() * self._crow_indices.element_size()
        col_mem = self._col_indices.numel() * self._col_indices.element_size()
        deg_mem = self._degrees.numel() * self._degrees.element_size()
        total_mem = crow_mem + col_mem + deg_mem
        dense_mem = n_nodes * self._max_degree * 8 if self._max_degree > 0 else 0

        savings = 100 * (1 - total_mem / dense_mem) if dense_mem > 0 else 0.0

        logger.info(
            f"TorchGraphStore (CSR) initialized: {n_nodes} nodes, {self._col_indices.numel()} edges, "
            f"max_degree={self._max_degree}, avg_degree={avg_degree:.1f}, device={self._device}, "
            f"memory={total_mem / 1024 / 1024:.1f}MB (vs {dense_mem / 1024 / 1024:.0f}MB dense, "
            f"{savings:.1f}% savings)"
        )

    @classmethod
    def from_adjacency_dict(
        cls,
        adj_dict: Dict[int, List[int]],
        device: Optional[TorchDeviceStr] = None,
        avg_degree: float = None
    ) -> "TorchGraphStore":
        """
        Create TorchGraphStore from adjacency dictionary.

        Args:
            adj_dict: Dictionary mapping node_id -> list of neighbor IDs
            device: Target device (auto-detected if None)
            avg_degree: Average degree (computed if None)

        Returns:
            TorchGraphStore instance
        """
        if not adj_dict:
            raise ValueError("Cannot create store from empty adjacency dict")

        target_device = device or get_device()

        # Ensure nodes are handled even if not contiguous in keys (0 to max_id)
        nodes = sorted(adj_dict.keys())
        n_nodes = max(nodes) + 1 if nodes else 0

        indptr = [0] * (n_nodes + 1)
        total_edges = 0

        # Build indptr (crow_indices)
        for node_id in range(n_nodes):
            neighbors = adj_dict.get(node_id, [])
            deg = len(neighbors)
            indptr[node_id + 1] = indptr[node_id] + deg
            total_edges += deg

        # Build indices (col_indices)
        indices = [0] * total_edges
        for node_id in range(n_nodes):
            neighbors = adj_dict.get(node_id, [])
            if neighbors:
                start = indptr[node_id]
                indices[start:start + len(neighbors)] = neighbors

        degrees = [indptr[i + 1] - indptr[i] for i in range(n_nodes)]

        if avg_degree is None:
            non_zero_nodes = sum(1 for d in degrees if d > 0)
            avg_degree = total_edges / non_zero_nodes if non_zero_nodes > 0 else 0.0

        return cls(
            crow_indices=torch.as_tensor(indptr, dtype=torch.long, device=target_device),
            col_indices=torch.as_tensor(indices, dtype=torch.long, device=target_device),
            degree_tensor=torch.as_tensor(degrees, dtype=torch.int32, device=target_device),
            n_nodes=n_nodes,
            avg_degree=avg_degree,
            device=target_device
        )

    @classmethod
    def from_csr(
        cls,
        csr_matrix: sp.csr_matrix,
        device: Optional[TorchDeviceStr] = None,
        avg_degree: float = None
    ) -> "TorchGraphStore":
        """
        Create TorchGraphStore from scipy CSR sparse matrix.

        Args:
            csr_matrix: Scipy CSR sparse adjacency matrix
            device: Target device (auto-detected if None)
            avg_degree: Average degree (computed if None)

        Returns:
            TorchGraphStore instance
        """
        n_nodes = csr_matrix.shape[0]
        target_device = device or get_device()

        indptr = torch.as_tensor(csr_matrix.indptr, dtype=torch.long, device=target_device)
        indices = torch.as_tensor(csr_matrix.indices, dtype=torch.long, device=target_device)
        degrees = indptr[1:] - indptr[:-1]

        if avg_degree is None:
            non_zero = (degrees > 0).sum().item()
            avg_degree = float(degrees.sum().item()) / non_zero if non_zero > 0 else 0.0

        return cls(
            crow_indices=indptr,
            col_indices=indices,
            degree_tensor=degrees.to(torch.int32),
            n_nodes=n_nodes,
            avg_degree=avg_degree,
            device=device
        )

    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """Get neighbors for a single node. Returns tensor on device."""
        if node_id < 0 or node_id >= self._n_nodes:
            return torch.tensor([], dtype=torch.long, device=self._device)

        start = self._crow_indices[node_id].item()
        end = self._crow_indices[node_id + 1].item()

        if start == end:
            return torch.tensor([], dtype=torch.long, device=self._device)

        return self._col_indices[start:end]

    def get_neighbors_batch(
        self,
        node_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch neighbor lookup for multiple nodes.

        Args:
            node_ids: Tensor/list of node IDs

        Returns:
            Tuple of:
                - neighbors: Tensor of shape (batch_size, batch_max_degree)
                  Padded with -1 for missing neighbors
                - mask: Boolean tensor of shape (batch_size, batch_max_degree)
                  True where valid neighbors exist
        """
        ids = node_ids.to(device=self._device, dtype=torch.long)
        batch_size = ids.shape[0]

        if batch_size == 0:
            return (
                torch.empty((0, 0), device=self._device, dtype=torch.long),
                torch.empty((0, 0), device=self._device, dtype=torch.bool)
            )

        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        # Clamp IDs to prevent index errors during lookup, zeroed out later via mask
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        # Get start/end indices and lengths for all nodes in batch
        starts = self._crow_indices[clamped_ids]
        ends = self._crow_indices[clamped_ids + 1]
        lengths = ends - starts
        
        # Apply valid mask to lengths (invalid nodes have length 0)
        lengths = torch.where(valid_mask, lengths, torch.zeros_like(lengths))

        total_neighbors = lengths.sum().item()
        batch_max_degree = int(lengths.max().item()) if batch_size > 0 else 0

        if batch_max_degree == 0:
            return (
                torch.full((batch_size, 1), -1, device=self._device, dtype=torch.long),
                torch.zeros((batch_size, 1), device=self._device, dtype=torch.bool)
            )

        if total_neighbors == 0:
             # Edge case: batch_size > 0 but no valid neighbors found
            return (
                torch.full((batch_size, 1), -1, device=self._device, dtype=torch.long),
                torch.zeros((batch_size, 1), device=self._device, dtype=torch.bool)
            )

        # Initialize output tensors
        neighbors = torch.full(
            (batch_size, batch_max_degree), -1,
            device=self._device, dtype=torch.long
        )
        mask = torch.zeros(
            (batch_size, batch_max_degree),
            device=self._device, dtype=torch.bool
        )

        # Vectorized neighbor gathering
        lengths_int = lengths.int()

        # 1. Determine which row (node ID) each neighbor belongs to in the flat list
        #    e.g. lengths=[2, 1] -> row_indices=[0, 0, 1]
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=self._device),
            lengths_int
        )

        # 2. Calculate the start offset for each node in the flat index space
        #    e.g. lengths=[2, 1] -> cumsum=[2, 3] -> group_starts=[0, 2]
        group_starts = torch.cumsum(lengths, dim=0) - lengths
        
        # 3. Repeat these start offsets to match the size of the flat neighbor list
        #    e.g. group_starts=[0, 2] -> repeated=[0, 0, 2]
        start_offsets_flat = torch.repeat_interleave(group_starts, lengths_int)

        # 4. Generate the column positions (0 to degree-1) for each node
        #    flat_idx = [0, 1, 2]
        #    col_pos = flat_idx - start_offsets = [0-0, 1-0, 2-2] = [0, 1, 0]
        flat_idx = torch.arange(total_neighbors, device=self._device)
        col_positions = flat_idx - start_offsets_flat

        # 5. Gather actual neighbor indices from CSR col_indices
        #    Calculate flat indices into self._col_indices
        #    flat_starts = [start_0, start_0, start_1]
        flat_starts = torch.repeat_interleave(starts, lengths_int)
        flat_indices = flat_starts + col_positions

        # Safety clamp for indices (shouldn't be necessary if logic is correct, but defensive)
        max_col_idx = self._col_indices.shape[0] - 1
        flat_indices = torch.clamp(flat_indices, 0, max_col_idx)

        flat_neighbors = self._col_indices[flat_indices]

        # 6. Scatter back into the dense batch tensor
        neighbors[row_indices, col_positions] = flat_neighbors
        mask[row_indices, col_positions] = True

        return neighbors, mask

    def get_degrees_batch(
        self,
        node_ids: torch.Tensor
    ) -> torch.Tensor:
        """Batch degree lookup for multiple nodes."""
        ids = node_ids.to(device=self._device, dtype=torch.long)

        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        degrees = self._degrees[clamped_ids]
        degrees = torch.where(valid_mask, degrees, torch.zeros_like(degrees))

        return degrees

    def get_degree(self, node_id: int) -> int:
        """Get degree for a single node."""
        if node_id < 0 or node_id >= self._n_nodes:
            return 0
        return int(self._degrees[node_id].item())

    def contains(self, node_id: int) -> bool:
        """Check if node exists in graph."""
        return 0 <= node_id < self._n_nodes

    def get_avg_degree(self) -> float:
        """Return average graph degree."""
        return self._avg_degree

    @property
    def max_degree(self) -> int:
        """Maximum node degree."""
        return self._max_degree

    @property
    def device(self) -> str:
        """Current device."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU (cuda or mps)."""
        return self._device != "cpu"

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def to(self, device: Optional[TorchDeviceStr]) -> "TorchGraphStore":
        """Move store to different device. Deprecated: set device at construction."""
        logger.warning("TorchGraphStore.to() is deprecated - set device at construction time")
        self._crow_indices = self._crow_indices.to(device)
        self._col_indices = self._col_indices.to(device)
        self._degrees = self._degrees.to(device)
        self._device = device
        return self

    def close(self):
        """Release memory held by this store."""
        if hasattr(self, '_crow_indices') and self._crow_indices is not None:
            del self._crow_indices
            self._crow_indices = None

        if hasattr(self, '_col_indices') and self._col_indices is not None:
            del self._col_indices
            self._col_indices = None

        if hasattr(self, '_degrees') and self._degrees is not None:
            del self._degrees
            self._degrees = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("TorchGraphStore resources released")

    def __del__(self):
        """Destructor to ensure memory is released."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release resources."""
        self.close()
        return False


__all__ = ['TorchGraphStore']