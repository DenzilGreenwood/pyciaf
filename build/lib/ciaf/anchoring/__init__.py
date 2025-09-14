"""
Dataset and model anchoring system for lazy capsule materialization.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .dataset_anchor import DatasetAnchor
from .lazy_manager import LazyProvenanceManager
from .simple_lazy_manager import LazyManager
from .true_lazy_manager import LazyReference, TrueLazyManager

__all__ = [
    "DatasetAnchor",
    "LazyProvenanceManager",
    "LazyManager",
    "TrueLazyManager",
    "LazyReference",
]
