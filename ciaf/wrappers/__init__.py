"""
CIAF Wrappers Package

Protocol-based drop-in wrappers for integrating existing ML models with CIAF.
Now includes policy-driven configuration and protocol interfaces for enhanced
architecture consistency with other CIAF modules.

Created: 2025-09-09
Last Modified: 2025-09-27
Author: Denzil James Greenwood
Version: 2.0.0
"""

from typing import Any

# Core interfaces and protocols
from .interfaces import (
    ModelWrapper,
    ModelAdapter,
    ModelMetadataProvider,
    ModelValidator,
    ModelTrainingHandler,
    ModelInferenceHandler,
    LCMMetadataHandler,
    ModelEnhancementProvider,
    ComplianceIntegrator,
    PerformanceOptimizer,
)

# Policy-driven configuration
from .policy import (
    WrapperPolicy,
    WrapperMode,
    ModelType,
    ComplianceMode,
    PerformanceLevel,
    ModelCompatibilityPolicy,
    EnhancementPolicy,
    TrainingPolicy,
    InferencePolicy,
    LCMIntegrationPolicy,
    CompliancePolicy,
    PerformancePolicy,
    get_default_wrapper_policy,
    set_default_wrapper_policy,
    create_wrapper_policy,
)

# Protocol implementations (only if available)
try:
    from .protocol_implementations import (
        DefaultModelAdapter,
        DefaultModelMetadataProvider,
        DefaultModelValidator,
        DefaultModelTrainingHandler,
        DefaultModelInferenceHandler,
        DefaultLCMMetadataHandler,
        DefaultModelEnhancementProvider,
        DefaultComplianceIntegrator,
        DefaultPerformanceOptimizer,
        create_default_wrapper_protocols,
    )
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = False
    # Create None placeholders
    DefaultModelAdapter = None
    DefaultModelMetadataProvider = None
    DefaultModelValidator = None
    DefaultModelTrainingHandler = None
    DefaultModelInferenceHandler = None
    DefaultLCMMetadataHandler = None
    DefaultModelEnhancementProvider = None
    DefaultComplianceIntegrator = None
    DefaultPerformanceOptimizer = None
    create_default_wrapper_protocols = None

# Legacy wrapper implementations (with deprecation warnings)
try:
    from .model_wrapper import CIAFModelWrapper
    LEGACY_WRAPPER_AVAILABLE = True
except ImportError:
    LEGACY_WRAPPER_AVAILABLE = False
    CIAFModelWrapper = None

try:
    from .enhanced_model_wrapper import EnhancedCIAFModelWrapper
    ENHANCED_WRAPPER_AVAILABLE = True
except ImportError:
    ENHANCED_WRAPPER_AVAILABLE = False
    EnhancedCIAFModelWrapper = None

# Modern protocol-based wrapper implementation
try:
    from .modern_wrapper import ModernCIAFModelWrapper
    MODERN_WRAPPER_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Modern wrapper not available: {e}")
    MODERN_WRAPPER_AVAILABLE = False
    ModernCIAFModelWrapper = None


# Factory functions for creating wrappers
def create_model_wrapper(model: Any, 
                        model_name: str,
                        policy: WrapperPolicy = None,
                        wrapper_type: str = "auto") -> Any:
    """
    Create a model wrapper using the best available implementation.
    
    Args:
        model: The ML model to wrap
        model_name: Unique name for the model
        policy: WrapperPolicy to use (creates default if None)
        wrapper_type: Type of wrapper ("modern", "enhanced", "legacy", "auto")
    
    Returns:
        Best available wrapper implementation
    """
    import warnings
    
    policy = policy or get_default_wrapper_policy()
    
    # Try to use modern wrapper first
    if wrapper_type in ["modern", "auto"] and MODERN_WRAPPER_AVAILABLE:
        return ModernCIAFModelWrapper(model=model, model_name=model_name, policy=policy)
    
    # Fall back to enhanced wrapper
    elif wrapper_type in ["enhanced", "auto"] and ENHANCED_WRAPPER_AVAILABLE:
        warnings.warn(
            "Using enhanced wrapper. Consider upgrading to modern protocol-based wrapper.",
            DeprecationWarning,
            stacklevel=2
        )
        return EnhancedCIAFModelWrapper(model=model, model_name=model_name)
    
    # Fall back to legacy wrapper
    elif wrapper_type in ["legacy", "auto"] and LEGACY_WRAPPER_AVAILABLE:
        warnings.warn(
            "Using legacy wrapper. Consider upgrading to modern protocol-based wrapper.",
            DeprecationWarning,
            stacklevel=2
        )
        return CIAFModelWrapper(model=model, model_name=model_name)
    
    else:
        raise ImportError(f"No wrapper implementation available for type: {wrapper_type}")


def create_auto_wrapper(model: Any, model_name: str, **kwargs) -> Any:
    """
    Automatically create the best wrapper for the given model.
    
    Args:
        model: ML model to wrap
        model_name: Unique name for the model
        **kwargs: Additional arguments for wrapper configuration
    
    Returns:
        Configured wrapper instance
    """
    # Extract policy-related kwargs
    policy_kwargs = {}
    wrapper_kwargs = {}
    
    for key, value in kwargs.items():
        if key.endswith('_policy') or key in ['wrapper_mode', 'compliance_mode', 'performance_level']:
            policy_kwargs[key] = value
        else:
            wrapper_kwargs[key] = value
    
    # Create policy if policy kwargs provided
    policy = None
    if policy_kwargs:
        policy = create_wrapper_policy(**policy_kwargs)
    
    return create_model_wrapper(
        model=model,
        model_name=model_name,
        policy=policy,
        **wrapper_kwargs
    )


__all__ = [
    # Core interfaces
    "ModelWrapper",
    "ModelAdapter", 
    "ModelMetadataProvider",
    "ModelValidator",
    "ModelTrainingHandler",
    "ModelInferenceHandler", 
    "LCMMetadataHandler",
    "ModelEnhancementProvider",
    "ComplianceIntegrator",
    "PerformanceOptimizer",
    
    # Policy framework
    "WrapperPolicy",
    "WrapperMode",
    "ModelType", 
    "ComplianceMode",
    "PerformanceLevel",
    "ModelCompatibilityPolicy",
    "EnhancementPolicy",
    "TrainingPolicy",
    "InferencePolicy",
    "LCMIntegrationPolicy", 
    "CompliancePolicy",
    "PerformancePolicy",
    "get_default_wrapper_policy",
    "set_default_wrapper_policy",
    "create_wrapper_policy",
    
    # Factory functions
    "create_model_wrapper",
    "create_auto_wrapper",
    
    # Availability flags
    "PROTOCOL_IMPLEMENTATIONS_AVAILABLE",
    "MODERN_WRAPPER_AVAILABLE", 
    "ENHANCED_WRAPPER_AVAILABLE",
    "LEGACY_WRAPPER_AVAILABLE",
] + (
    # Protocol implementations (only if available)
    [
        "DefaultModelAdapter",
        "DefaultModelMetadataProvider", 
        "DefaultModelValidator",
        "DefaultModelTrainingHandler",
        "DefaultModelInferenceHandler",
        "DefaultLCMMetadataHandler",
        "DefaultModelEnhancementProvider",
        "DefaultComplianceIntegrator",
        "DefaultPerformanceOptimizer",
        "create_default_wrapper_protocols",
    ] if PROTOCOL_IMPLEMENTATIONS_AVAILABLE else []
) + (
    # Legacy wrappers (with deprecation)
    [
        "CIAFModelWrapper",
        "EnhancedCIAFModelWrapper",
    ] if any([LEGACY_WRAPPER_AVAILABLE, ENHANCED_WRAPPER_AVAILABLE]) else []
) + (
    # Modern wrapper
    [
        "ModernCIAFModelWrapper",
    ] if MODERN_WRAPPER_AVAILABLE else []
)