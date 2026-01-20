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


__all__ = [
    'get_device',
    'ensure_tensor',
    'to_numpy',
    'clear_device_cache',
    'get_gpu_memory_info',
]
