"""
CIAF Wrappers Package

Drop-in wrappers for integrating existing ML models with CIAF.

Created: 2025-09-09
Last Modified: 2025-09-25
Author: Denzil James Greenwood
Version: 1.1.0
"""

from .model_wrapper import CIAFModelWrapper

# Import enhanced wrapper with error handling
try:
    from .enhanced_model_wrapper import EnhancedCIAFModelWrapper
    ENHANCED_WRAPPER_AVAILABLE = True
except ImportError:
    ENHANCED_WRAPPER_AVAILABLE = False
    EnhancedCIAFModelWrapper = None

__all__ = ["CIAFModelWrapper", "EnhancedCIAFModelWrapper", "ENHANCED_WRAPPER_AVAILABLE"]