"""
Unified storage module for evolution runs.

Provides:
- RunManager: Device-aware checkpoint/log/result management
- Directory structure: {base_dir}/{dataset}/{run_id}/...
"""
from .run_manager import RunManager

__all__ = ["RunManager"]
