"""
GPU Device Utilities with Auto-Fallback

Provides transparent GPU acceleration support with automatic fallback to CPU.
Control via environment variable SWARM_RAG_DEVICE or force_cpu parameter.
"""

import os
from functools import lru_cache
from typing import Union, Any
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
        - "mps": Force MPS (will fail if unavailable)

    Args:
        force_cpu: If True, always return "cpu" regardless of env/GPU availability

    Returns:
        Device string: "cuda", "mps", or "cpu"
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
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: CUDA ({device_name})")
            return "cuda"
        else:
            raise RuntimeError("CUDA requested but not available")

    if env_device == "mps":
        # User explicitly requested MPS - let it fail if unavailable
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Device: MPS (Apple Silicon)")
            return "mps"
        else:
            raise RuntimeError("MPS requested but not available")

    # Auto-detect mode: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: CUDA auto-detected ({device_name})")
        return "cuda"

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Device: MPS auto-detected (Apple Silicon)")
        return "mps"

    logger.debug("Device: CPU (no GPU available)")
    return "cpu"


def ensure_tensor(
    data: Union[torch.Tensor, list],
    device: str = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Convert input to PyTorch tensor on the specified device.

    Args:
        data: Input data (torch tensor or list)
        device: Target device (defaults to auto-detected device)
        dtype: Optional torch dtype (e.g., torch.float32)

    Returns:
        torch.Tensor on the specified device
    """
    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    return tensor.to(device)


def clear_device_cache():
    """
    Clear the cached device detection result.
    Useful for testing or when GPU availability changes.
    """
    get_device.cache_clear()


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information if available.

    Note: Only works for CUDA devices. MPS has a different memory API.

    Returns:
        Dictionary with 'allocated', 'cached', 'total' in bytes,
        or empty dict if CUDA not available.
    """
    if get_device() != "cuda":
        return {}

    try:
        return {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'total': torch.cuda.get_device_properties(0).total_memory,
        }
    except Exception:
        return {}


def resolve_device(device: str = "auto") -> str:
    """
    Resolve device string to actual device.

    Args:
        device: Device specification - "auto", "cuda", "mps", or "cpu"

    Returns:
        Resolved device string: "cuda", "mps", or "cpu"

    Raises:
        RuntimeError: If requested device is not available
    """
    if device == "cpu":
        return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    elif device == "mps":
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return "mps"
    else:  # auto
        return get_device()


def is_accelerated_device(device: str) -> bool:
    """Check if device is GPU-accelerated (cuda or mps)."""
    return device in ("cuda", "mps")


def to_device(data: Any, device: torch.device) -> Any:
    """
    Move tensor or nested structure to device.

    Recursively handles dicts, lists, and individual tensors.

    Args:
        data: Input data (tensor, dict, list, or other)
        device: Target torch.device

    Returns:
        Data with all tensors moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data


def smart_convert(
    data: Union[torch.Tensor, list],
    device: str = None
) -> torch.Tensor:
    """
    Smart conversion that minimizes data movement.

    Intelligently converts data to tensor, avoiding unnecessary
    copies when data is already in the right format.

    Args:
        data: Input data (torch tensor or list)
        device: Target device for tensor output (defaults to auto-detected)

    Returns:
        Tensor on target device with minimal copies

    Examples:
        # GPU tensor stays on GPU
        >>> smart_convert(gpu_tensor, "cuda")  # no copy

        # List to GPU tensor
        >>> smart_convert([1.0, 2.0], "cuda")  # single transfer
    """
    target_device = device or get_device()

    if isinstance(data, torch.Tensor):
        # Already a tensor - just ensure correct device
        if str(data.device).startswith(target_device):
            return data  # No move needed
        return data.to(target_device)

    return torch.as_tensor(data, device=target_device)


def move_to_device(
    data: Union[torch.Tensor, list],
    device: str = None
) -> torch.Tensor:
    """
    Move data to device.

    This is a convenience function that ensures data ends up as a
    PyTorch tensor on the specified device with minimal data movement.

    Args:
        data: Input data (torch tensor or list)
        device: Target device (defaults to auto-detected)

    Returns:
        torch.Tensor on the specified device
    """
    target_device = device or get_device()

    if isinstance(data, torch.Tensor):
        if str(data.device).startswith(target_device):
            return data
        return data.to(target_device)

    return torch.as_tensor(data, device=target_device)


def tensor_like(
    data: Union[torch.Tensor, list],
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Create tensor matching the device and dtype of a reference tensor.

    Useful for creating new tensors that should match existing computation
    without explicitly passing device/dtype parameters.

    Args:
        data: Input data to convert
        reference: Reference tensor whose device and dtype to match

    Returns:
        Tensor with same device and dtype as reference

    Examples:
        >>> embeddings = torch.randn(100, 768, device='cuda')
        >>> query = tensor_like([1.0, 2.0, 3.0], embeddings)
        >>> query.device  # cuda:0
        >>> query.dtype   # torch.float32
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=reference.device, dtype=reference.dtype)

    return torch.as_tensor(data, device=reference.device, dtype=reference.dtype)


def is_tensor(data: Any) -> bool:
    """Check if data is a PyTorch tensor without importing torch."""
    return type(data).__module__ == 'torch' and type(data).__name__ == 'Tensor'


def supports_gpu(min_docs: int = 1000) -> bool:
    """
    Check if GPU operations are supported and worthwhile.

    Args:
        min_docs: Minimum dataset size to justify GPU overhead

    Returns:
        True if GPU is available (cuda or mps)
    """
    return get_device() in ("cuda", "mps")


__all__ = [
    'get_device',
    'ensure_tensor',
    'clear_device_cache',
    'get_gpu_memory_info',
    'resolve_device',
    'is_accelerated_device',
    'to_device',
    'smart_convert',
    'move_to_device',
    'tensor_like',
    'is_tensor',
    'supports_gpu',
]
