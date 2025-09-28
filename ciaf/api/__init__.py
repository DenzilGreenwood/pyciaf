"""
CIAF API Package
================

Enhanced API components for the Cognitive Insight Audit Framework with
protocol-based architecture, policy-driven configuration, and comprehensive
integration with all CIAF modules.

Created: 2025-09-09
Last Modified: 2025-09-28
Author: Denzil James Greenwood
Version: 2.0.0
"""

from typing import Any

# Core protocol interfaces
from .interfaces import (
    DatasetAPIHandler,
    ModelAPIHandler,
    TrainingAPIHandler,
    InferenceAPIHandler,
    AuditAPIHandler,
    ComplianceAPIHandler,
    SecurityAPIHandler,
    MetricsAPIHandler,
    APIResponseHandler,
    APIMiddleware,
    CIAFAPIFramework,
    APIRequest,
    APIResponse,
    APIError,
    APIFilters,
    APIStatus,
)

# Policy-driven configuration
from .policy import (
    APIPolicy,
    APIMode,
    SecurityLevel,
    ComplianceFramework,
    AuthenticationMethod,
    RateLimitStrategy,
    CachingStrategy,
    SecurityPolicy,
    RateLimitPolicy,
    CachingPolicy,
    CompliancePolicy,
    PerformancePolicy,
    IntegrationPolicy,
    LoggingPolicy,
    get_development_api_policy,
    get_production_api_policy,
    get_testing_api_policy,
    get_default_api_policy,
    set_default_api_policy,
    create_api_policy,
)

# Protocol implementations
try:
    from .protocol_implementations import (
        DefaultDatasetAPIHandler,
        DefaultModelAPIHandler,
        DefaultTrainingAPIHandler,
        DefaultInferenceAPIHandler,
        DefaultAPIResponseHandler,
        DefaultSecurityAPIHandler,
    )
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = False
    DefaultDatasetAPIHandler = None
    DefaultModelAPIHandler = None
    DefaultTrainingAPIHandler = None
    DefaultInferenceAPIHandler = None
    DefaultAPIResponseHandler = None
    DefaultSecurityAPIHandler = None

# Consolidated API framework
try:
    from .consolidated_api import (
        ConsolidatedCIAFAPIFramework,
        ConsolidatedAuditAPIHandler,
        ConsolidatedMetricsAPIHandler,
    )
    CONSOLIDATED_API_AVAILABLE = True
except ImportError:
    CONSOLIDATED_API_AVAILABLE = False
    ConsolidatedCIAFAPIFramework = None
    ConsolidatedAuditAPIHandler = None
    ConsolidatedMetricsAPIHandler = None

# Legacy framework (with deprecation warnings)
try:
    from .framework import CIAFFramework
    LEGACY_FRAMEWORK_AVAILABLE = True
except ImportError:
    LEGACY_FRAMEWORK_AVAILABLE = False
    CIAFFramework = None


# Factory functions for creating API frameworks
def create_api_framework(
    policy: APIPolicy = None,
    framework_type: str = "auto"
) -> Any:
    """
    Create an API framework using the best available implementation.
    
    Args:
        policy: APIPolicy to use (creates default if None)
        framework_type: Type of framework ("consolidated", "legacy", "auto")
    
    Returns:
        Best available API framework implementation
    """
    import warnings
    
    policy = policy or get_default_api_policy()
    
    # Try consolidated framework first (recommended)
    if framework_type in ["consolidated", "auto"] and CONSOLIDATED_API_AVAILABLE:
        return ConsolidatedCIAFAPIFramework(policy=policy)
    
    # Fall back to legacy framework
    elif framework_type in ["legacy", "auto"] and LEGACY_FRAMEWORK_AVAILABLE:
        warnings.warn(
            "Using legacy API framework. Consider upgrading to consolidated framework.",
            DeprecationWarning,
            stacklevel=2
        )
        # Convert policy to legacy format if needed
        return CIAFFramework(
            framework_name="CIAF",
            policy=getattr(policy, 'custom_config', {}).get('legacy_policy'),
            anchor_signer=None
        )
    
    else:
        raise ImportError(f"No API framework implementation available for type: {framework_type}")


def create_development_api() -> Any:
    """Create API framework optimized for development."""
    policy = get_development_api_policy()
    return create_api_framework(policy=policy, framework_type="auto")


def create_production_api() -> Any:
    """Create API framework optimized for production."""
    policy = get_production_api_policy()
    return create_api_framework(policy=policy, framework_type="consolidated")


def create_testing_api() -> Any:
    """Create API framework optimized for testing."""
    policy = get_testing_api_policy()
    return create_api_framework(policy=policy, framework_type="auto")


__all__ = [
    # Core interfaces
    "DatasetAPIHandler",
    "ModelAPIHandler",
    "TrainingAPIHandler",
    "InferenceAPIHandler",
    "AuditAPIHandler",
    "ComplianceAPIHandler",
    "SecurityAPIHandler",
    "MetricsAPIHandler",
    "APIResponseHandler",
    "APIMiddleware",
    "CIAFAPIFramework",
    
    # Type aliases
    "APIRequest",
    "APIResponse", 
    "APIError",
    "APIFilters",
    "APIStatus",
    
    # Policy framework
    "APIPolicy",
    "APIMode",
    "SecurityLevel",
    "ComplianceFramework",
    "AuthenticationMethod",
    "RateLimitStrategy",
    "CachingStrategy",
    "SecurityPolicy",
    "RateLimitPolicy",
    "CachingPolicy",
    "CompliancePolicy",
    "PerformancePolicy",
    "IntegrationPolicy",
    "LoggingPolicy",
    "get_development_api_policy",
    "get_production_api_policy",
    "get_testing_api_policy",
    "get_default_api_policy",
    "set_default_api_policy",
    "create_api_policy",
    
    # Factory functions
    "create_api_framework",
    "create_development_api",
    "create_production_api",
    "create_testing_api",
    
    # Availability flags
    "PROTOCOL_IMPLEMENTATIONS_AVAILABLE",
    "CONSOLIDATED_API_AVAILABLE",
    "LEGACY_FRAMEWORK_AVAILABLE",
] + (
    # Protocol implementations
    [
        "DefaultDatasetAPIHandler",
        "DefaultModelAPIHandler",
        "DefaultTrainingAPIHandler",
        "DefaultInferenceAPIHandler",
        "DefaultAPIResponseHandler",
        "DefaultSecurityAPIHandler",
    ] if PROTOCOL_IMPLEMENTATIONS_AVAILABLE else []
) + (
    # Consolidated framework
    [
        "ConsolidatedCIAFAPIFramework",
        "ConsolidatedAuditAPIHandler",
        "ConsolidatedMetricsAPIHandler",
    ] if CONSOLIDATED_API_AVAILABLE else []
) + (
    # Legacy framework
    [
        "CIAFFramework",
    ] if LEGACY_FRAMEWORK_AVAILABLE else []
)
