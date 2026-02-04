"""
STaRK Dataset Loading Utilities

This module provides functions to load and precompute data for STaRK benchmarks.
"""

from .load_stark import (
    load_and_download_embeddings,
    load_and_download_skb,
    load_and_download_qa,
    precompute_stark_adjacency,
)

__all__ = [
    "load_and_download_embeddings",
    "load_and_download_skb",
    "load_and_download_qa",
    "precompute_stark_adjacency",
]
