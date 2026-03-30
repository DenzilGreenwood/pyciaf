"""
CIAF API Policy Configuration
============================

Policy-driven configuration for the CIAF API system, providing comprehensive
control over API behavior, security, compliance, and operational parameters.

Created: 2025-09-28
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, ConfigDict


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



class SecurityPolicy(BaseModel):
    """API security policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Authentication
    authentication_method: AuthenticationMethod = Field(
        default=AuthenticationMethod.JWT,
        description="Authentication method to use"
    )
    require_authentication: bool = Field(default=True, description="Require authentication for API access")
    session_timeout: timedelta = Field(
        default_factory=lambda: timedelta(hours=24),
        description="Session timeout duration"
    )
    max_login_attempts: int = Field(default=3, ge=1, description="Maximum login attempts before lockout")
    lockout_duration: timedelta = Field(
        default_factory=lambda: timedelta(minutes=15),
        description="Lockout duration after max attempts"
    )

    # Authorization
    enable_rbac: bool = Field(default=True, description="Enable role-based access control")
    default_permissions: Set[str] = Field(default_factory=set, description="Default permissions for users")
    admin_permissions: Set[str] = Field(
        default_factory=lambda: {
            "create_model",
            "delete_model",
            "manage_users",
            "view_audit_logs",
        },
        description="Admin permissions"
    )

    # API Security
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")
    enable_https_only: bool = Field(default=True, description="Require HTTPS only")
    api_key_length: int = Field(default=32, ge=16, le=128, description="API key length in characters")

    # Audit
    log_all_requests: bool = Field(default=True, description="Log all API requests")
    log_request_bodies: bool = Field(default=False, description="Log request bodies (may contain sensitive data)")
    log_response_bodies: bool = Field(default=False, description="Log response bodies")
    audit_sensitive_operations: bool = Field(default=True, description="Audit sensitive operations")



class RateLimitPolicy(BaseModel):
    """Rate limiting policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    enabled: bool = Field(default=True, description="Enable rate limiting")
    strategy: RateLimitStrategy = Field(default=RateLimitStrategy.TOKEN_BUCKET, description="Rate limiting strategy")

    # Global limits
    global_requests_per_minute: int = Field(default=1000, ge=0, description="Global requests per minute")
    global_requests_per_hour: int = Field(default=10000, ge=0, description="Global requests per hour")

    # Per-user limits
    user_requests_per_minute: int = Field(default=100, ge=0, description="User requests per minute")
    user_requests_per_hour: int = Field(default=1000, ge=0, description="User requests per hour")

    # Endpoint-specific limits
    endpoint_limits: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "/api/v1/inference": {"requests_per_minute": 50, "requests_per_hour": 500},
            "/api/v1/training": {"requests_per_minute": 10, "requests_per_hour": 100},
            "/api/v1/models": {"requests_per_minute": 20, "requests_per_hour": 200},
        },
        description="Endpoint-specific rate limits"
    )

    # Burst handling
    allow_burst: bool = Field(default=True, description="Allow burst traffic")
    burst_multiplier: float = Field(default=2.0, gt=1.0, description="Burst multiplier")
    burst_window: timedelta = Field(
        default_factory=lambda: timedelta(seconds=30),
        description="Burst window duration"
    )



class CachingPolicy(BaseModel):
    """API caching policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    enabled: bool = Field(default=True, description="Enable caching")
    strategy: CachingStrategy = Field(default=CachingStrategy.MEMORY, description="Caching strategy")
    default_ttl: timedelta = Field(
        default_factory=lambda: timedelta(minutes=15),
        description="Default cache TTL"
    )

    # Endpoint-specific caching
    endpoint_cache_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "/api/v1/models": {"ttl": timedelta(hours=1), "enabled": True},
            "/api/v1/datasets": {"ttl": timedelta(hours=2), "enabled": True},
            "/api/v1/metrics": {"ttl": timedelta(minutes=5), "enabled": True},
            "/api/v1/inference": {
                "ttl": timedelta(minutes=1),
                "enabled": False,
            },  # Fresh data needed
        },
        description="Endpoint-specific cache configuration"
    )

    # Cache invalidation
    auto_invalidate: bool = Field(default=True, description="Enable auto cache invalidation")
    invalidation_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "model_updated": ["/api/v1/models/*", "/api/v1/metrics/models/*"],
            "dataset_updated": ["/api/v1/datasets/*", "/api/v1/metrics/datasets/*"],
        },
        description="Cache invalidation patterns"
    )



class CompliancePolicy(BaseModel):
    """Compliance policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    enabled_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: [
            ComplianceFramework.GDPR,
            ComplianceFramework.EU_AI_ACT,
        ],
        description="Enabled compliance frameworks"
    )

    # GDPR settings
    gdpr_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "data_retention_days": 2555,  # 7 years
            "require_consent": True,
            "enable_right_to_be_forgotten": True,
            "data_portability": True,
            "privacy_by_design": True,
        },
        description="GDPR configuration"
    )

    # EU AI Act settings
    ai_act_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "high_risk_system": False,
            "transparency_requirements": True,
            "human_oversight": True,
            "accuracy_requirements": True,
            "robustness_testing": True,
        },
        description="EU AI Act configuration"
    )

    # Audit requirements
    compliance_audit_frequency: timedelta = Field(
        default_factory=lambda: timedelta(days=30),
        description="Compliance audit frequency"
    )
    automated_compliance_checks: bool = Field(default=True, description="Enable automated compliance checks")
    compliance_reporting: bool = Field(default=True, description="Enable compliance reporting")



class PerformancePolicy(BaseModel):
    """API performance policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Request handling
    max_request_size: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        description="Maximum request size in bytes (10MB)"
    )
    max_response_size: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        description="Maximum response size in bytes (50MB)"
    )
    request_timeout: timedelta = Field(
        default_factory=lambda: timedelta(seconds=30),
        description="Request timeout duration"
    )

    # Connection handling
    max_concurrent_connections: int = Field(default=1000, ge=1, description="Maximum concurrent connections")
    connection_pool_size: int = Field(default=100, ge=1, description="Connection pool size")
    keep_alive_timeout: timedelta = Field(
        default_factory=lambda: timedelta(seconds=60),
        description="Keep-alive timeout"
    )

    # Background processing
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    background_task_queue_size: int = Field(default=1000, ge=1, description="Background task queue size")
    worker_threads: int = Field(default=4, ge=1, description="Number of worker threads")

    # Monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    metrics_collection_interval: timedelta = Field(
        default_factory=lambda: timedelta(seconds=30),
        description="Metrics collection interval"
    )
    performance_alerting: bool = Field(default=True, description="Enable performance alerting")



class IntegrationPolicy(BaseModel):
    """Integration policy for external systems."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # LCM Integration
    enable_lcm_integration: bool = Field(default=True, description="Enable LCM integration")
    lcm_auto_tracking: bool = Field(default=True, description="Enable automatic LCM tracking")
    lcm_audit_level: str = Field(default="full", description="LCM audit level")

    # Wrapper Integration
    enable_universal_wrappers: bool = Field(default=True, description="Enable universal wrappers")
    auto_detect_model_types: bool = Field(default=True, description="Auto-detect model types")
    fallback_to_legacy_wrappers: bool = Field(default=True, description="Fallback to legacy wrappers")

    # Compliance Integration
    enable_realtime_compliance: bool = Field(default=True, description="Enable real-time compliance")
    compliance_validation_level: str = Field(default="strict", description="Compliance validation level")

    # External APIs
    external_api_timeout: timedelta = Field(
        default_factory=lambda: timedelta(seconds=10),
        description="External API timeout"
    )
    external_api_retries: int = Field(default=3, ge=0, description="External API retry count")
    external_api_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker")


class LoggingPolicy(BaseModel):
    """Logging policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Log levels
    default_log_level: str = Field(default="INFO", description="Default log level")
    security_log_level: str = Field(default="WARNING", description="Security log level")
    audit_log_level: str = Field(default="INFO", description="Audit log level")

    # Log destinations
    log_to_file: bool = Field(default=True, description="Log to file")
    log_to_console: bool = Field(default=True, description="Log to console")
    log_to_syslog: bool = Field(default=False, description="Log to syslog")

    # Log rotation
    log_file_max_size: str = Field(default="100MB", description="Maximum log file size")
    log_file_backup_count: int = Field(default=10, ge=0, description="Log file backup count")
    log_rotation_when: str = Field(default="midnight", description="Log rotation schedule")

    # Log content
    include_request_id: bool = Field(default=True, description="Include request ID in logs")
    include_user_id: bool = Field(default=True, description="Include user ID in logs")
    include_stack_trace: bool = Field(default=True, description="Include stack traces")
    mask_sensitive_data: bool = Field(default=True, description="Mask sensitive data in logs")

    # Audit logging
    audit_log_file: str = Field(default="ciaf_audit.log", description="Audit log file name")
    audit_log_format: str = Field(default="JSON", description="Audit log format")
    audit_log_retention_days: int = Field(default=2555, ge=0, description="Audit log retention (7 years)")



class APIPolicy(BaseModel):
    """Complete API policy configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Basic configuration
    policy_id: str = Field(default="ciaf_api_default_v1", description="Policy identifier")
    version: str = Field(default="1.0.0", description="Policy version")
    created_at: datetime = Field(default_factory=datetime.now, description="Policy creation timestamp")

    # Operating mode
    api_mode: APIMode = Field(default=APIMode.PRODUCTION, description="API operating mode")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    # Policy components
    security: SecurityPolicy = Field(default_factory=SecurityPolicy, description="Security policy")
    rate_limiting: RateLimitPolicy = Field(default_factory=RateLimitPolicy, description="Rate limiting policy")
    caching: CachingPolicy = Field(default_factory=CachingPolicy, description="Caching policy")
    compliance: CompliancePolicy = Field(default_factory=CompliancePolicy, description="Compliance policy")
    performance: PerformancePolicy = Field(default_factory=PerformancePolicy, description="Performance policy")
    integration: IntegrationPolicy = Field(default_factory=IntegrationPolicy, description="Integration policy")
    logging: LoggingPolicy = Field(default_factory=LoggingPolicy, description="Logging policy")

    # Custom extensions
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration extensions")

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
        ComplianceFramework.SOC2,
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
    **overrides: Any,
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
