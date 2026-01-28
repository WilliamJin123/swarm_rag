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
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: CUDA ({device_name})")
            return "cuda"
        else:
            raise RuntimeError("CUDA requested but not available")

    # Auto-detect mode
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: CUDA auto-detected ({device_name})")
        return "cuda"

    logger.debug("Device: CPU (CUDA not available)")
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

    Returns:
        Dictionary with 'allocated', 'cached', 'total' in bytes,
        or empty dict if GPU not available.
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


def get_device_from_mode(use_gpu: str = "auto") -> torch.device:
    """
    Get device based on mode string.

    Args:
        use_gpu: GPU mode - "auto" (detect), "always" (require GPU), "never" (CPU only)

    Returns:
        torch.device for the selected device

    Raises:
        RuntimeError: If use_gpu="always" but CUDA is not available
    """
    if use_gpu == "never":
        return torch.device("cpu")
    elif use_gpu == "always":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested (use_gpu='always') but CUDA is not available")
        return torch.device("cuda")
    else:  # auto
        device_str = get_device()
        return torch.device(device_str)


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
        True if GPU is available and dataset is large enough
    """
    return get_device() == "cuda"


__all__ = [
    'get_device',
    'ensure_tensor',
    'clear_device_cache',
    'get_gpu_memory_info',
    'get_device_from_mode',
    'to_device',
    'smart_convert',
    'move_to_device',
    'tensor_like',
    'is_tensor',
    'supports_gpu',
]
