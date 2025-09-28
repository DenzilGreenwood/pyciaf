"""
CIAF API Policy Configuration
============================

Policy-driven configuration for the CIAF API system, providing comprehensive
control over API behavior, security, compliance, and operational parameters.

Created: 2025-09-28
Author: Denzil James Greenwood  
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime, timedelta


class APIMode(Enum):
    """API operating modes."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


class SecurityLevel(Enum):
    """API security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    EU_AI_ACT = "eu_ai_act"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CUSTOM = "custom"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    CUSTOM = "custom"
    NONE = "none"  # For development only


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"
    NONE = "none"


class CachingStrategy(Enum):
    """API caching strategies."""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class SecurityPolicy:
    """API security policy configuration."""
    
    # Authentication
    authentication_method: AuthenticationMethod = AuthenticationMethod.JWT
    require_authentication: bool = True
    session_timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    max_login_attempts: int = 3
    lockout_duration: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    
    # Authorization
    enable_rbac: bool = True
    default_permissions: Set[str] = field(default_factory=set)
    admin_permissions: Set[str] = field(default_factory=lambda: {
        "create_model", "delete_model", "manage_users", "view_audit_logs"
    })
    
    # API Security
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_https_only: bool = True
    api_key_length: int = 32
    
    # Audit
    log_all_requests: bool = True
    log_request_bodies: bool = False  # May contain sensitive data
    log_response_bodies: bool = False
    audit_sensitive_operations: bool = True


@dataclass
class RateLimitPolicy:
    """Rate limiting policy configuration."""
    
    enabled: bool = True
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    
    # Global limits
    global_requests_per_minute: int = 1000
    global_requests_per_hour: int = 10000
    
    # Per-user limits
    user_requests_per_minute: int = 100
    user_requests_per_hour: int = 1000
    
    # Endpoint-specific limits
    endpoint_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "/api/v1/inference": {"requests_per_minute": 50, "requests_per_hour": 500},
        "/api/v1/training": {"requests_per_minute": 10, "requests_per_hour": 100},
        "/api/v1/models": {"requests_per_minute": 20, "requests_per_hour": 200}
    })
    
    # Burst handling
    allow_burst: bool = True
    burst_multiplier: float = 2.0
    burst_window: timedelta = field(default_factory=lambda: timedelta(seconds=30))


@dataclass
class CachingPolicy:
    """API caching policy configuration."""
    
    enabled: bool = True
    strategy: CachingStrategy = CachingStrategy.MEMORY
    default_ttl: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    
    # Endpoint-specific caching
    endpoint_cache_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "/api/v1/models": {"ttl": timedelta(hours=1), "enabled": True},
        "/api/v1/datasets": {"ttl": timedelta(hours=2), "enabled": True},
        "/api/v1/metrics": {"ttl": timedelta(minutes=5), "enabled": True},
        "/api/v1/inference": {"ttl": timedelta(minutes=1), "enabled": False}  # Fresh data needed
    })
    
    # Cache invalidation
    auto_invalidate: bool = True
    invalidation_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "model_updated": ["/api/v1/models/*", "/api/v1/metrics/models/*"],
        "dataset_updated": ["/api/v1/datasets/*", "/api/v1/metrics/datasets/*"]
    })


@dataclass
class CompliancePolicy:
    """Compliance policy configuration."""
    
    enabled_frameworks: List[ComplianceFramework] = field(default_factory=lambda: [
        ComplianceFramework.GDPR, ComplianceFramework.EU_AI_ACT
    ])
    
    # GDPR settings
    gdpr_config: Dict[str, Any] = field(default_factory=lambda: {
        "data_retention_days": 2555,  # 7 years
        "require_consent": True,
        "enable_right_to_be_forgotten": True,
        "data_portability": True,
        "privacy_by_design": True
    })
    
    # EU AI Act settings
    ai_act_config: Dict[str, Any] = field(default_factory=lambda: {
        "high_risk_system": False,
        "transparency_requirements": True,
        "human_oversight": True,
        "accuracy_requirements": True,
        "robustness_testing": True
    })
    
    # Audit requirements
    compliance_audit_frequency: timedelta = field(default_factory=lambda: timedelta(days=30))
    automated_compliance_checks: bool = True
    compliance_reporting: bool = True


@dataclass
class PerformancePolicy:
    """API performance policy configuration."""
    
    # Request handling
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    request_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Connection handling
    max_concurrent_connections: int = 1000
    connection_pool_size: int = 100
    keep_alive_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    
    # Background processing
    enable_async_processing: bool = True
    background_task_queue_size: int = 1000
    worker_threads: int = 4
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    performance_alerting: bool = True


@dataclass
class IntegrationPolicy:
    """Integration policy for external systems."""
    
    # LCM Integration
    enable_lcm_integration: bool = True
    lcm_auto_tracking: bool = True
    lcm_audit_level: str = "full"
    
    # Wrapper Integration
    enable_universal_wrappers: bool = True
    auto_detect_model_types: bool = True
    fallback_to_legacy_wrappers: bool = True
    
    # Compliance Integration
    enable_realtime_compliance: bool = True
    compliance_validation_level: str = "strict"
    
    # External APIs
    external_api_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    external_api_retries: int = 3
    external_api_circuit_breaker: bool = True


@dataclass
class LoggingPolicy:
    """Logging policy configuration."""
    
    # Log levels
    default_log_level: str = "INFO"
    security_log_level: str = "WARNING"
    audit_log_level: str = "INFO"
    
    # Log destinations
    log_to_file: bool = True
    log_to_console: bool = True
    log_to_syslog: bool = False
    
    # Log rotation
    log_file_max_size: str = "100MB"
    log_file_backup_count: int = 10
    log_rotation_when: str = "midnight"
    
    # Log content
    include_request_id: bool = True
    include_user_id: bool = True
    include_stack_trace: bool = True
    mask_sensitive_data: bool = True
    
    # Audit logging
    audit_log_file: str = "ciaf_audit.log"
    audit_log_format: str = "JSON"
    audit_log_retention_days: int = 2555  # 7 years


@dataclass
class APIPolicy:
    """Complete API policy configuration."""
    
    # Basic configuration
    policy_id: str = "ciaf_api_default_v1"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Operating mode
    api_mode: APIMode = APIMode.PRODUCTION
    debug_mode: bool = False
    
    # Policy components
    security: SecurityPolicy = field(default_factory=SecurityPolicy)
    rate_limiting: RateLimitPolicy = field(default_factory=RateLimitPolicy)
    caching: CachingPolicy = field(default_factory=CachingPolicy)
    compliance: CompliancePolicy = field(default_factory=CompliancePolicy)
    performance: PerformancePolicy = field(default_factory=PerformancePolicy)
    integration: IntegrationPolicy = field(default_factory=IntegrationPolicy)
    logging: LoggingPolicy = field(default_factory=LoggingPolicy)
    
    # Custom extensions
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def is_development_mode(self) -> bool:
        """Check if API is in development mode."""
        return self.api_mode == APIMode.DEVELOPMENT
    
    def is_production_mode(self) -> bool:
        """Check if API is in production mode."""
        return self.api_mode == APIMode.PRODUCTION
    
    def requires_high_security(self) -> bool:
        """Check if high security is required."""
        return self.api_mode in [APIMode.PRODUCTION, APIMode.STAGING]
    
    def get_compliance_frameworks(self) -> List[ComplianceFramework]:
        """Get enabled compliance frameworks."""
        return self.compliance.enabled_frameworks


# Default policies for different environments
def get_development_api_policy() -> APIPolicy:
    """Get API policy for development environment."""
    policy = APIPolicy()
    policy.api_mode = APIMode.DEVELOPMENT
    policy.debug_mode = True
    
    # Relaxed security for development
    policy.security.require_authentication = False
    policy.security.enable_https_only = False
    policy.security.log_request_bodies = True
    
    # Relaxed rate limiting
    policy.rate_limiting.enabled = False
    
    # Disabled caching for fresh data
    policy.caching.enabled = False
    
    # Minimal compliance for development
    policy.compliance.enabled_frameworks = []
    policy.compliance.automated_compliance_checks = False
    
    return policy


def get_production_api_policy() -> APIPolicy:
    """Get API policy for production environment."""
    policy = APIPolicy()
    policy.api_mode = APIMode.PRODUCTION
    policy.debug_mode = False
    
    # Strict security
    policy.security.require_authentication = True
    policy.security.enable_https_only = True
    policy.security.log_request_bodies = False
    policy.security.audit_sensitive_operations = True
    
    # Conservative rate limiting
    policy.rate_limiting.enabled = True
    policy.rate_limiting.user_requests_per_minute = 50
    
    # Aggressive caching
    policy.caching.enabled = True
    policy.caching.default_ttl = timedelta(hours=1)
    
    # Full compliance
    policy.compliance.enabled_frameworks = [
        ComplianceFramework.GDPR, 
        ComplianceFramework.EU_AI_ACT,
        ComplianceFramework.SOC2
    ]
    
    return policy


def get_testing_api_policy() -> APIPolicy:
    """Get API policy for testing environment."""
    policy = APIPolicy()
    policy.api_mode = APIMode.TESTING
    policy.debug_mode = True
    
    # Moderate security
    policy.security.require_authentication = True
    policy.security.enable_https_only = False
    policy.security.log_request_bodies = True
    
    # Moderate rate limiting
    policy.rate_limiting.enabled = True
    policy.rate_limiting.user_requests_per_minute = 200
    
    # Light caching
    policy.caching.enabled = True
    policy.caching.default_ttl = timedelta(minutes=5)
    
    # Compliance testing
    policy.compliance.enabled_frameworks = [ComplianceFramework.GDPR]
    policy.compliance.automated_compliance_checks = True
    
    return policy


# Global policy management
_default_api_policy: Optional[APIPolicy] = None


def get_default_api_policy() -> APIPolicy:
    """Get the default API policy."""
    global _default_api_policy
    if _default_api_policy is None:
        _default_api_policy = get_production_api_policy()
    return _default_api_policy


def set_default_api_policy(policy: APIPolicy) -> None:
    """Set the default API policy."""
    global _default_api_policy
    _default_api_policy = policy


def create_api_policy(
    api_mode: APIMode = APIMode.PRODUCTION,
    security_level: SecurityLevel = SecurityLevel.HIGH,
    **overrides: Any
) -> APIPolicy:
    """
    Create a customized API policy.
    
    Args:
        api_mode: Operating mode for the API
        security_level: Security level requirements
        **overrides: Override specific policy settings
        
    Returns:
        Customized APIPolicy instance
    """
    # Start with environment-specific base policy
    if api_mode == APIMode.DEVELOPMENT:
        policy = get_development_api_policy()
    elif api_mode == APIMode.TESTING:
        policy = get_testing_api_policy()
    else:
        policy = get_production_api_policy()
    
    # Adjust security based on level
    if security_level == SecurityLevel.CRITICAL:
        policy.security.max_login_attempts = 2
        policy.security.session_timeout = timedelta(hours=2)
        policy.rate_limiting.user_requests_per_minute = 20
        policy.compliance.enabled_frameworks.append(ComplianceFramework.ISO_27001)
    elif security_level == SecurityLevel.LOW:
        policy.security.max_login_attempts = 10
        policy.security.session_timeout = timedelta(days=7)
        policy.rate_limiting.user_requests_per_minute = 500
    
    # Apply custom overrides
    for key, value in overrides.items():
        if hasattr(policy, key):
            setattr(policy, key, value)
    
    return policy


__all__ = [
    # Enums
    "APIMode",
    "SecurityLevel", 
    "ComplianceFramework",
    "AuthenticationMethod",
    "RateLimitStrategy",
    "CachingStrategy",
    
    # Policy classes
    "SecurityPolicy",
    "RateLimitPolicy",
    "CachingPolicy",
    "CompliancePolicy",
    "PerformancePolicy",
    "IntegrationPolicy",
    "LoggingPolicy",
    "APIPolicy",
    
    # Factory functions
    "get_development_api_policy",
    "get_production_api_policy", 
    "get_testing_api_policy",
    "get_default_api_policy",
    "set_default_api_policy",
    "create_api_policy",
]