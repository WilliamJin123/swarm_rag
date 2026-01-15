"""
GPU Device Utilities with Auto-Fallback

Provides transparent GPU acceleration support with automatic fallback to CPU.
Control via environment variable SWARM_RAG_DEVICE or force_cpu parameter.
"""

import os
from functools import lru_cache
from typing import Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import torch

import logging
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device(force_cpu: bool = False) -> str:
    """
    Auto-detect best available device with environment variable override.

    Environment variable: SWARM_RAG_DEVICE
        - "auto" (default): Auto-detect GPU availability
        - "cpu": Force CPU mode
        - "cuda": Force CUDA (will fail if unavailable)

    Args:
        force_cpu: If True, always return "cpu" regardless of env/GPU availability

    Returns:
        Device string: "cuda" or "cpu"
    """
    if force_cpu:
        logger.debug("Device: CPU (forced)")
        return "cpu"

    env_device = os.environ.get("SWARM_RAG_DEVICE", "auto").lower()

    if env_device == "cpu":
        logger.debug("Device: CPU (env override)")
        return "cpu"

    if env_device == "cuda":
        # User explicitly requested CUDA - let it fail if unavailable
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Device: CUDA ({device_name})")
                return "cuda"
            else:
                raise RuntimeError("CUDA requested but not available")
        except ImportError:
            raise RuntimeError("CUDA requested but PyTorch not installed")

    # Auto-detect mode
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: CUDA auto-detected ({device_name})")
            return "cuda"
    except ImportError:
        logger.debug("Device: CPU (PyTorch not installed)")
        return "cpu"

    logger.debug("Device: CPU (CUDA not available)")
    return "cpu"


def get_array_module():
    """
    Return CuPy if GPU available, else NumPy.

    This allows writing device-agnostic array code:
        xp = get_array_module()
        arr = xp.array([1, 2, 3])
        result = xp.sum(arr)

    Returns:
        Module: cupy or numpy
    """
    if get_device() == "cuda":
        try:
            import cupy as cp
            return cp
        except ImportError:
            logger.warning("CuPy not installed, falling back to NumPy despite CUDA availability")

    return np


def ensure_tensor(
    data: Union[np.ndarray, "torch.Tensor", list],
    device: str = None,
    dtype: "torch.dtype" = None
) -> "torch.Tensor":
    """
    Convert input to PyTorch tensor on the specified device.

    Args:
        data: Input data (numpy array, torch tensor, or list)
        device: Target device (defaults to auto-detected device)
        dtype: Optional torch dtype (e.g., torch.float32)

    Returns:
        torch.Tensor on the specified device
    """
    import torch

    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    return tensor.to(device)


def to_numpy(data: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """
    Convert tensor to numpy array (handling GPU tensors).

    Args:
        data: Input tensor or array

    Returns:
        numpy.ndarray
    """
    if isinstance(data, np.ndarray):
        return data

    # Handle torch tensors
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass

    return np.asarray(data)


def clear_device_cache():
    """
    Clear the cached device detection result.
    Useful for testing or when GPU availability changes.
    """
    get_device.cache_clear()


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information if available.

    Returns:
        Dictionary with 'allocated', 'cached', 'total' in bytes,
        or empty dict if GPU not available.
    """
    if get_device() != "cuda":
        return {}

    try:
        import torch
        return {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'total': torch.cuda.get_device_properties(0).total_memory,
        }
    except Exception:
        return {}


# === CuPy Integration ===

def is_cupy_available() -> bool:
    """
    Check if CuPy is available and functional.

    Returns:
        True if CuPy is available and a GPU is detected
    """
    if get_device() != "cuda":
        return False

    try:
        import cupy as cp
        # Try a simple operation to verify CuPy is functional
        _ = cp.array([1, 2, 3])
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"CuPy available but not functional: {e}")
        return False


def to_cupy(data: Union[np.ndarray, "torch.Tensor", list]) -> "np.ndarray":
    """
    Convert data to CuPy array on GPU, with fallback to NumPy.

    Args:
        data: Input data (numpy array, torch tensor, or list)

    Returns:
        CuPy array if GPU available, otherwise NumPy array
    """
    if get_device() == "cuda":
        try:
            import cupy as cp

            if hasattr(data, '__cuda_array_interface__'):
                # Data is already a CUDA array (torch tensor, cupy array)
                return cp.asarray(data)

            # Handle torch tensors
            try:
                import torch
                if isinstance(data, torch.Tensor):
                    if data.is_cuda:
                        # Direct conversion via DLPack for zero-copy
                        return cp.from_dlpack(data)
                    else:
                        # Move to GPU first
                        return cp.asarray(data.cpu().numpy())
            except ImportError:
                pass

            # Handle numpy arrays and lists
            return cp.asarray(data)
        except ImportError:
            logger.debug("CuPy not available, returning NumPy array")
        except Exception as e:
            logger.debug(f"CuPy conversion failed: {e}")

    # Fallback to NumPy
    return np.asarray(data)


def cupy_to_numpy(data) -> np.ndarray:
    """
    Convert CuPy array to NumPy array.

    Args:
        data: CuPy array or any array-like

    Returns:
        NumPy array
    """
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
    except ImportError:
        pass

    return np.asarray(data)


def cupy_matmul(a, b):
    """
    Matrix multiplication using CuPy when available.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Result matrix (CuPy if inputs were CuPy, NumPy otherwise)
    """
    xp = get_array_module()
    return xp.matmul(xp.asarray(a), xp.asarray(b))


def cupy_dot(a, b):
    """
    Dot product using CuPy when available.

    Args:
        a: First array
        b: Second array

    Returns:
        Dot product result
    """
    xp = get_array_module()
    return xp.dot(xp.asarray(a), xp.asarray(b))


def cupy_norm(a, axis=None, keepdims=False):
    """
    Compute L2 norm using CuPy when available.

    Args:
        a: Input array
        axis: Axis along which to compute norm
        keepdims: Keep dimensions

    Returns:
        L2 norm
    """
    xp = get_array_module()
    arr = xp.asarray(a)
    return xp.linalg.norm(arr, axis=axis, keepdims=keepdims)


def cupy_normalize(a, axis=-1, eps=1e-8):
    """
    L2 normalize array using CuPy when available.

    Args:
        a: Input array
        axis: Axis along which to normalize
        eps: Small epsilon for numerical stability

    Returns:
        Normalized array
    """
    xp = get_array_module()
    arr = xp.asarray(a)
    norm = xp.linalg.norm(arr, axis=axis, keepdims=True) + eps
    return arr / norm


def cupy_cosine_similarity(query, candidates):
    """
    Compute cosine similarity between query and candidates using CuPy.

    Optimized for GPU when available.

    Args:
        query: Query vector of shape (dim,) or (1, dim)
        candidates: Candidate matrix of shape (n, dim)

    Returns:
        Similarity scores of shape (n,)
    """
    xp = get_array_module()

    query = xp.asarray(query).flatten()
    candidates = xp.asarray(candidates)

    # Normalize query
    query_norm = query / (xp.linalg.norm(query) + 1e-8)

    # Normalize candidates
    candidate_norms = xp.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8
    candidates_normalized = candidates / candidate_norms

    # Dot product for cosine similarity
    similarities = xp.dot(candidates_normalized, query_norm)

    return similarities


def cupy_topk(scores, k):
    """
    Get top-k indices and values using CuPy when available.

    Args:
        scores: Score array
        k: Number of top elements

    Returns:
        Tuple of (top_scores, top_indices)
    """
    xp = get_array_module()
    scores = xp.asarray(scores)

    k = min(k, len(scores))

    # Use argpartition for efficiency
    indices = xp.argpartition(scores, -k)[-k:]
    top_scores = scores[indices]

    # Sort the top-k
    sorted_order = xp.argsort(top_scores)[::-1]
    indices = indices[sorted_order]
    top_scores = top_scores[sorted_order]

    return top_scores, indices


def sync_device():
    """
    Synchronize CUDA device if using GPU.

    Call this before timing operations for accurate measurements.
    """
    if get_device() == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        try:
            import cupy as cp
            cp.cuda.Device().synchronize()
        except ImportError:
            pass


__all__ = [
    'get_device',
    'get_array_module',
    'ensure_tensor',
    'to_numpy',
    'clear_device_cache',
    'get_gpu_memory_info',
    # CuPy integration
    'is_cupy_available',
    'to_cupy',
    'cupy_to_numpy',
    'cupy_matmul',
    'cupy_dot',
    'cupy_norm',
    'cupy_normalize',
    'cupy_cosine_similarity',
    'cupy_topk',
    'sync_device',
]
