import numpy as np
import pytest
from swarm_rag.ops import init_backend, xp, to_device, to_cpu

def test_backend_initialization_cpu():
    """Verify that the CPU backend initializes correctly and uses NumPy."""
    init_backend(use_gpu=False)
    assert xp is np, "xp should be numpy when GPU is disabled"
    
    data = [1.0, 2.0, 3.0]
    arr = to_device(data)
    assert isinstance(arr, np.ndarray), "to_device should return numpy array in CPU mode"
    assert np.allclose(arr, data)

def test_to_cpu_consistency():
    """Verify that to_cpu always returns a NumPy array."""
    init_backend(use_gpu=False)
    data = np.array([1, 2, 3])
    arr = to_cpu(data)
    assert isinstance(arr, np.ndarray)
    assert not hasattr(arr, 'get'), "Result should be a raw numpy array, not a cupy-like object"

def test_gpu_fallback_behavior():
    """
    Test the behavior when GPU is requested. 
    If cupy is missing, it should fallback gracefully to numpy.
    """
    # This just checks that the call doesn't crash and handles the state
    init_backend(use_gpu=True)
    
    try:
        import cupy as cp
        assert xp is cp, "xp should be cupy if cupy is installed and use_gpu=True"
        assert hasattr(xp, 'ndarray'), "xp should have ndarray attribute"
    except ImportError:
        assert xp is np, "xp should fallback to numpy if cupy is not installed"

if __name__ == "__main__":
    test_backend_initialization_cpu()
    test_to_cpu_consistency()
    test_gpu_fallback_behavior()
    print("GPU Ops Unit Tests Passed (Logic Check)")
