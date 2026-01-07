import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Default to numpy
xp = np
_use_gpu = False

def init_backend(use_gpu: bool):
    """
    Initialize the computation backend.
    
    Args:
        use_gpu (bool): If True, attempts to import cupy and use it as the backend.
                        Falls back to numpy if cupy is not installed.
    """
    global xp, _use_gpu
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            _use_gpu = True
            logger.info("Swarm RAG: GPU backend enabled (CuPy).")
        except ImportError:
            logger.warning("Swarm RAG: GPU backend requested but CuPy not found. Falling back to NumPy.")
            xp = np
            _use_gpu = False
    else:
        xp = np
        _use_gpu = False
        logger.info("Swarm RAG: CPU backend enabled (NumPy).")

def to_device(array):
    """
    Moves an array to the current backend device.
    If backend is GPU, converts to cupy.ndarray.
    If backend is CPU, converts to numpy.ndarray.
    """
    if array is None:
        return None
        
    if _use_gpu:
        # If it's already a cupy array (or similar), asarray is cheap/no-op
        return xp.asarray(array)
    
    # If it's a cupy array but we are in CPU mode, we need to bring it back
    if hasattr(array, 'get'): # Check for cupy-like object
        return array.get()
        
    return np.asarray(array)

def to_cpu(array):
    """
    Moves an array to the CPU (NumPy).
    """
    if array is None:
        return None

    if hasattr(array, 'get'):
        return array.get()
    
    if hasattr(array, 'cpu'): # PyTorch compat just in case
        return array.cpu().numpy()
        
    return np.asarray(array)
