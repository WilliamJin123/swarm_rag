"""
GPU-Accelerated Graph Store for SwarmRAG

Stores graph adjacency in CSR (Compressed Sparse Row) format on GPU.
Provides batch neighbor lookups to eliminate CPU bottleneck in swarm traversal.

Memory footprint: ~65MB for prime dataset (129K nodes, 16M edges)
vs ~17GB for dense format (99.6% reduction)
"""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp

from ..interfaces.abstract_classes import GraphStore
from ..utils.device import get_device

if TYPE_CHECKING:
    import torch

import logging
logger = logging.getLogger(__name__)


class GPUGraphStore(GraphStore):
    """
    GPU-accelerated graph store using CSR (Compressed Sparse Row) format.

    Stores adjacency in sparse format for massive memory savings while
    maintaining fast batch neighbor lookups via vectorized operations.

    Key features:
    - CSR format: O(nodes + edges) memory vs O(nodes * max_degree) for dense
    - Batch neighbor lookup via vectorized gather operations
    - Dynamic max_degree per batch (not fixed)
    - Compatible with existing GraphStore interface

    Usage:
        # From adjacency dict
        store = GPUGraphStore.from_adjacency_dict(adj_dict)

        # From CSR matrix
        store = GPUGraphStore.from_csr(csr_matrix)

        # Batch operations
        neighbors, mask = store.get_neighbors_batch([1, 2, 3])
        degrees = store.get_degrees_batch([1, 2, 3])
    """

    def __init__(
        self,
        crow_indices: "torch.Tensor",
        col_indices: "torch.Tensor",
        degree_tensor: "torch.Tensor",
        n_nodes: int,
        avg_degree: float,
        device: str = None
    ):
        """
        Initialize GPU graph store with CSR components.

        Args:
            crow_indices: Row pointers tensor of shape (n_nodes + 1,)
                         crow_indices[i] is the start index in col_indices for node i
            col_indices: Neighbor IDs tensor of shape (total_edges,)
            degree_tensor: Tensor of shape (n_nodes,) containing node degrees
            n_nodes: Total number of nodes in graph
            avg_degree: Average degree of the graph
            device: Target device ("cuda" or "cpu")
        """
        import torch

        self._device = device or get_device()
        self._n_nodes = n_nodes
        self._avg_degree = avg_degree

        # Store CSR components on device
        if isinstance(crow_indices, torch.Tensor):
            self._crow_indices = crow_indices.to(device=self._device, dtype=torch.long)
        else:
            self._crow_indices = torch.tensor(crow_indices, device=self._device, dtype=torch.long)

        if isinstance(col_indices, torch.Tensor):
            self._col_indices = col_indices.to(device=self._device, dtype=torch.long)
        else:
            self._col_indices = torch.tensor(col_indices, device=self._device, dtype=torch.long)

        if isinstance(degree_tensor, torch.Tensor):
            self._degrees = degree_tensor.to(device=self._device, dtype=torch.int32)
        else:
            self._degrees = torch.tensor(degree_tensor, device=self._device, dtype=torch.int32)

        # Cache max_degree (computed once)
        self._max_degree = int(self._degrees.max().item()) if n_nodes > 0 else 0

        # Compute memory usage
        crow_mem = self._crow_indices.numel() * self._crow_indices.element_size()
        col_mem = self._col_indices.numel() * self._col_indices.element_size()
        deg_mem = self._degrees.numel() * self._degrees.element_size()
        total_mem = crow_mem + col_mem + deg_mem

        # Estimate dense memory for comparison
        dense_mem = n_nodes * self._max_degree * 8  # int64 = 8 bytes

        logger.info(
            f"GPUGraphStore (CSR) initialized: {n_nodes} nodes, {self._col_indices.numel()} edges, "
            f"max_degree={self._max_degree}, avg_degree={avg_degree:.1f}, device={self._device}, "
            f"memory={total_mem / 1024 / 1024:.1f}MB (vs {dense_mem / 1024 / 1024:.0f}MB dense, "
            f"{100 * (1 - total_mem / dense_mem):.1f}% savings)"
        )

    @classmethod
    def from_adjacency_dict(
        cls,
        adj_dict: Dict[int, List[int]],
        device: str = None,
        avg_degree: float = None
    ) -> "GPUGraphStore":
        """
        Create GPUGraphStore from adjacency dictionary.

        Args:
            adj_dict: Dictionary mapping node_id -> list of neighbor IDs
            device: Target device (auto-detected if None)
            avg_degree: Average degree (computed if None)

        Returns:
            GPUGraphStore instance
        """
        import torch

        if not adj_dict:
            raise ValueError("Cannot create store from empty adjacency dict")

        # Determine dimensions
        nodes = sorted(adj_dict.keys())
        n_nodes = max(nodes) + 1 if nodes else 0

        # Build CSR format directly
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        total_edges = 0

        # First pass: compute degrees and indptr
        for node_id in range(n_nodes):
            neighbors = adj_dict.get(node_id, [])
            deg = len(neighbors)
            indptr[node_id + 1] = indptr[node_id] + deg
            total_edges += deg

        # Allocate indices array
        indices = np.zeros(total_edges, dtype=np.int64)

        # Second pass: fill indices
        for node_id in range(n_nodes):
            neighbors = adj_dict.get(node_id, [])
            if neighbors:
                start = indptr[node_id]
                indices[start:start + len(neighbors)] = neighbors

        # Compute degrees
        degrees = np.diff(indptr).astype(np.int32)

        if avg_degree is None:
            non_zero_nodes = np.sum(degrees > 0)
            avg_degree = total_edges / non_zero_nodes if non_zero_nodes > 0 else 0.0

        return cls(
            crow_indices=torch.from_numpy(indptr),
            col_indices=torch.from_numpy(indices),
            degree_tensor=torch.from_numpy(degrees),
            n_nodes=n_nodes,
            avg_degree=avg_degree,
            device=device
        )

    @classmethod
    def from_csr(
        cls,
        csr_matrix: sp.csr_matrix,
        device: str = None,
        avg_degree: float = None
    ) -> "GPUGraphStore":
        """
        Create GPUGraphStore from scipy CSR sparse matrix.

        Args:
            csr_matrix: Scipy CSR sparse adjacency matrix
            device: Target device (auto-detected if None)
            avg_degree: Average degree (computed if None)

        Returns:
            GPUGraphStore instance
        """
        import torch

        n_nodes = csr_matrix.shape[0]

        # Directly use CSR components - no dense conversion needed!
        indptr = csr_matrix.indptr.astype(np.int64)
        indices = csr_matrix.indices.astype(np.int64)

        # Compute degrees from CSR structure
        degrees = np.diff(indptr).astype(np.int32)

        if avg_degree is None:
            non_zero = np.sum(degrees > 0)
            avg_degree = float(np.sum(degrees)) / non_zero if non_zero > 0 else 0.0

        return cls(
            crow_indices=torch.from_numpy(indptr),
            col_indices=torch.from_numpy(indices),
            degree_tensor=torch.from_numpy(degrees),
            n_nodes=n_nodes,
            avg_degree=avg_degree,
            device=device
        )

    def get_neighbors(self, node_id: int) -> np.ndarray:
        """
        Get neighbors for a single node (GraphStore interface).

        Args:
            node_id: Node ID to get neighbors for

        Returns:
            Array of neighbor node IDs
        """
        if node_id < 0 or node_id >= self._n_nodes:
            return np.array([], dtype=np.int64)

        start = self._crow_indices[node_id].item()
        end = self._crow_indices[node_id + 1].item()

        if start == end:
            return np.array([], dtype=np.int64)

        return self._col_indices[start:end].cpu().numpy()

    def get_neighbors_batch(
        self,
        node_ids: Union[List[int], np.ndarray, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Batch neighbor lookup for multiple nodes.

        This is the key method for GPU acceleration - fetches all neighbors
        for multiple nodes using vectorized CSR extraction.

        Args:
            node_ids: Tensor/array/list of node IDs

        Returns:
            Tuple of:
                - neighbors: Tensor of shape (batch_size, batch_max_degree) with neighbor IDs
                  Padded with -1 for missing neighbors
                - mask: Boolean tensor of shape (batch_size, batch_max_degree)
                  True where valid neighbors exist
        """
        import torch

        # Convert to tensor on device
        if isinstance(node_ids, torch.Tensor):
            ids = node_ids.to(device=self._device, dtype=torch.long)
        elif isinstance(node_ids, np.ndarray):
            ids = torch.from_numpy(node_ids).to(device=self._device, dtype=torch.long)
        else:
            ids = torch.tensor(node_ids, device=self._device, dtype=torch.long)

        batch_size = ids.shape[0]

        if batch_size == 0:
            return (
                torch.empty((0, 0), device=self._device, dtype=torch.long),
                torch.empty((0, 0), device=self._device, dtype=torch.bool)
            )

        # Handle out-of-bounds indices
        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        # Get row boundaries from CSR
        starts = self._crow_indices[clamped_ids]
        ends = self._crow_indices[clamped_ids + 1]
        lengths = ends - starts

        # Zero out lengths for invalid nodes
        lengths = torch.where(valid_mask, lengths, torch.zeros_like(lengths))

        # Compute batch_max_degree dynamically for this batch only
        batch_max_degree = int(lengths.max().item()) if batch_size > 0 else 0

        if batch_max_degree == 0:
            # All nodes have zero degree or are out of bounds
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

        # Build indices for parallel extraction
        # row_indices: which row each neighbor belongs to
        # col_positions: position within that row
        total_neighbors = lengths.sum().item()

        if total_neighbors > 0:
            # Fix: Convert lengths to int32 for repeat_interleave compatibility
            # torch.repeat_interleave requires int32/int64, but int64 can cause issues on some platforms
            lengths_int = lengths.int()

            # Create row indices via repeat_interleave
            row_indices = torch.repeat_interleave(
                torch.arange(batch_size, device=self._device),
                lengths_int
            )

            # Create column positions for each row
            # This creates [0,1,2,...,deg[0]-1, 0,1,2,...,deg[1]-1, ...]
            offsets = torch.zeros(total_neighbors, device=self._device, dtype=torch.long)
            cumsum = torch.cumsum(lengths, dim=0)
            # Mark start of each new row - guard against out-of-bounds indexing
            # This happens when trailing rows have zero length (invalid nodes)
            if batch_size > 1:
                # Only set offsets where cumsum index is within bounds
                valid_offset_mask = cumsum[:-1] < total_neighbors
                if valid_offset_mask.any():
                    valid_cumsum_indices = cumsum[:-1][valid_offset_mask]
                    valid_lengths = lengths[:-1][valid_offset_mask]
                    offsets[valid_cumsum_indices] = valid_lengths
            col_positions = torch.arange(total_neighbors, device=self._device) - torch.cumsum(offsets, dim=0)

            # Gather all neighbor indices at once
            # Build flat indices into col_indices
            flat_starts = torch.repeat_interleave(starts, lengths_int)
            flat_indices = flat_starts + col_positions

            # Bounds check: ensure flat_indices don't exceed col_indices size
            max_col_idx = self._col_indices.shape[0] - 1
            flat_indices = torch.clamp(flat_indices, 0, max_col_idx)

            flat_neighbors = self._col_indices[flat_indices]

            # Bounds check: ensure row/col positions are valid for output tensor
            col_positions = torch.clamp(col_positions, 0, batch_max_degree - 1)

            # Scatter into output
            neighbors[row_indices, col_positions] = flat_neighbors
            mask[row_indices, col_positions] = True

        return neighbors, mask

    def get_degrees_batch(
        self,
        node_ids: Union[List[int], np.ndarray, "torch.Tensor"]
    ) -> "torch.Tensor":
        """
        Batch degree lookup for multiple nodes.

        Args:
            node_ids: Tensor/array/list of node IDs

        Returns:
            Tensor of degrees for each node
        """
        import torch

        # Convert to tensor on device
        if isinstance(node_ids, torch.Tensor):
            ids = node_ids.to(device=self._device, dtype=torch.long)
        elif isinstance(node_ids, np.ndarray):
            ids = torch.from_numpy(node_ids).to(device=self._device, dtype=torch.long)
        else:
            ids = torch.tensor(node_ids, device=self._device, dtype=torch.long)

        # Handle out-of-bounds
        valid_mask = (ids >= 0) & (ids < self._n_nodes)
        clamped_ids = torch.clamp(ids, 0, self._n_nodes - 1)

        degrees = self._degrees[clamped_ids]

        # Set out-of-bounds to 0
        degrees = torch.where(valid_mask, degrees, torch.zeros_like(degrees))

        return degrees

    def get_degree(self, node_id: int) -> int:
        """
        Get degree for a single node.

        Args:
            node_id: Node ID

        Returns:
            Node degree (0 if out of bounds)
        """
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
    def n_nodes(self) -> int:
        """Total number of nodes."""
        return self._n_nodes

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
        """Check if using GPU."""
        return self._device == "cuda"

    def to(self, device: str) -> "GPUGraphStore":
        """
        Move store to different device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self._crow_indices = self._crow_indices.to(device)
        self._col_indices = self._col_indices.to(device)
        self._degrees = self._degrees.to(device)
        self._device = device
        return self

    def close(self):
        """
        Release GPU memory held by this store.

        Deletes CSR tensors (crow_indices, col_indices, degrees) and clears CUDA cache.
        Safe to call multiple times.
        """
        import torch

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

        logger.debug("GPUGraphStore resources released")

    def __del__(self):
        """Destructor to ensure GPU memory is released."""
        try:
            self.close()
        except Exception:
            # Ignore errors during interpreter shutdown
            pass


__all__ = ['GPUGraphStore']
