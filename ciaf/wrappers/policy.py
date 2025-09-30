"""
Policy-driven configuration for CIAF wrapper system.

This module provides comprehensive policy management for model wrappers,
following the same patterns as other CIAF modules. Policies control
wrapper behavior, enhancement features, compliance settings, and performance
optimizations.

Created: 2025-09-27
Author: Denzil James Greenwood  
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from datetime import datetime

# Import core utilities for policy integrity
from ..lcm import canonical_json, canonical_hash

if TYPE_CHECKING:
    from .interfaces import (
        ModelAdapter, ModelMetadataProvider, ModelValidator,
        ModelTrainingHandler, ModelInferenceHandler, LCMMetadataHandler,
        ModelEnhancementProvider, ComplianceIntegrator, PerformanceOptimizer
    )


class WrapperMode(Enum):
    """Operating modes for model wrappers."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"
    COMPLIANCE = "compliance"


class ModelType(Enum):
    """Supported model types - expanded for universal coverage."""
    SCIKIT_LEARN = "scikit_learn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    JAX = "jax"
    ONNX = "onnx"
    KERAS = "keras"
    CUSTOM = "custom"
    AUTO_DETECT = "auto_detect"


class ComplianceMode(Enum):
    """Compliance mode settings."""
    GENERAL = "general"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    GOVERNMENT = "government"
    RESEARCH = "research"
    STRICT = "strict"


class PerformanceLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    MAXIMUM = "maximum"


class DataType(Enum):
    """Supported data types for universal processing."""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    MIXED = "mixed"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    MULTIMODAL = "multimodal"


@dataclass
class ModelCompatibilityPolicy:
    """Policy for model compatibility and validation."""
    
    # Model type detection
    auto_detect_model_type: bool = True
    supported_model_types: List[ModelType] = field(default_factory=lambda: [
        ModelType.SCIKIT_LEARN, ModelType.PYTORCH, ModelType.TENSORFLOW, 
        ModelType.CUSTOM, ModelType.AUTO_DETECT
    ])
    strict_type_validation: bool = False
    
    # Model validation
    validate_model_structure: bool = True
    validate_model_readiness: bool = True
    validate_training_compatibility: bool = True
    validate_inference_compatibility: bool = True
    
    # Error handling
    fail_on_incompatible_model: bool = False
    fallback_to_simulation: bool = True
    log_compatibility_warnings: bool = True


@dataclass
class EnhancementPolicy:
    """Policy for model enhancement features."""
    
    # Core enhancements
    enable_preprocessing: bool = True
    enable_explainability: bool = True
    enable_uncertainty: bool = True
    enable_metadata_tags: bool = True
    
    # Auto-configuration
    auto_configure_enhancements: bool = True
    detect_enhancement_compatibility: bool = True
    
    # Enhancement thresholds
    min_samples_for_explainability: int = 10
    max_features_for_lime: int = 1000
    uncertainty_methods: List[str] = field(default_factory=lambda: ["bootstrap", "bayesian"])
    
    # Fallback behavior
    graceful_enhancement_failure: bool = True
    log_enhancement_warnings: bool = True


@dataclass
class TrainingPolicy:
    """Policy for model training operations."""
    
    # Training validation
    validate_training_data: bool = True
    min_training_samples: int = 10
    max_training_time: Optional[int] = None  # seconds
    
    # Training behavior
    fit_model_by_default: bool = True
    create_training_snapshots: bool = True
    preserve_training_metadata: bool = True
    
    # Error handling
    continue_on_training_failure: bool = True
    fallback_to_ciaf_simulation: bool = True
    log_training_details: bool = True
    
    # Performance
    enable_training_optimization: bool = True
    cache_training_data: bool = False
    parallel_training: bool = False


@dataclass
class InferencePolicy:
    """Policy for model inference operations."""
    
    # Inference validation
    validate_inference_input: bool = True
    preprocess_inference_input: bool = True
    postprocess_inference_output: bool = True
    
    # Receipt generation
    create_inference_receipts: bool = True
    enable_receipt_connections: bool = True
    store_detailed_metadata: bool = True
    
    # Enhancement integration
    include_explanations: bool = True
    include_uncertainty: bool = True
    include_metadata_tags: bool = True
    
    # Performance
    enable_inference_caching: bool = False
    cache_explanations: bool = True
    max_inference_time: Optional[int] = None  # seconds
    
    # Error handling
    fallback_on_inference_failure: bool = True
    log_inference_details: bool = True


@dataclass
class LCMIntegrationPolicy:
    """Policy for LCM system integration."""
    
    # LCM features
    enable_lcm_integration: bool = True
    preserve_lcm_metadata: bool = True
    enable_pickle_preservation: bool = True
    
    # Metadata handling
    extract_comprehensive_metadata: bool = True
    include_training_metadata: bool = True
    include_inference_metadata: bool = True
    include_deployment_metadata: bool = True
    
    # Serialization
    serialize_on_pickle: bool = True
    restore_on_unpickle: bool = True
    verify_metadata_integrity: bool = True
    
    # Performance
    defer_lcm_processing: bool = False
    batch_lcm_operations: bool = False
    lcm_cache_size: int = 1000


@dataclass
class CompliancePolicy:
    """Policy for compliance and regulatory requirements."""
    
    # Compliance mode
    compliance_mode: ComplianceMode = ComplianceMode.GENERAL
    regulatory_frameworks: List[str] = field(default_factory=lambda: ["general"])
    
    # Audit requirements
    enable_audit_trails: bool = True
    detailed_audit_logging: bool = True
    retain_audit_data: bool = True
    audit_retention_days: int = 365
    
    # Privacy and security
    anonymize_sensitive_data: bool = True
    encrypt_stored_metadata: bool = False
    secure_model_storage: bool = False
    
    # Compliance validation
    validate_regulatory_compliance: bool = True
    fail_on_compliance_violations: bool = False
    alert_on_compliance_issues: bool = True
    
    # Documentation
    generate_compliance_reports: bool = True
    include_technical_documentation: bool = True


@dataclass
class PerformancePolicy:
    """Policy for performance optimization."""
    
    # Performance level
    performance_level: PerformanceLevel = PerformanceLevel.STANDARD
    
    # Resource limits
    max_memory_usage: Optional[int] = None  # MB
    max_cpu_cores: Optional[int] = None
    max_gpu_memory: Optional[int] = None  # MB
    
    # Caching
    enable_model_caching: bool = False
    enable_preprocessing_caching: bool = True
    enable_result_caching: bool = False
    cache_size_limit: int = 1000  # MB
    
    # Optimization
    optimize_for_training: bool = True
    optimize_for_inference: bool = True
    enable_batch_processing: bool = False
    
    # Monitoring
    monitor_performance: bool = True
    log_performance_metrics: bool = False
    performance_alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "memory_usage": 0.8,  # 80%
        "inference_time": 10.0,  # seconds
        "training_time": 3600.0  # 1 hour
    })


@dataclass
class WrapperPolicy:
    """
    Comprehensive policy for CIAF model wrapper behavior.
    
    This policy controls all aspects of wrapper functionality including
    model compatibility, enhancements, training, inference, LCM integration,
    compliance, and performance optimization.
    """
    
    # Policy metadata
    policy_version: str = "1.0"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    organization: str = "CIAF Implementation"
    
    # Operating mode
    wrapper_mode: WrapperMode = WrapperMode.DEVELOPMENT
    
    # Sub-policies
    compatibility_policy: ModelCompatibilityPolicy = field(default_factory=ModelCompatibilityPolicy)
    enhancement_policy: EnhancementPolicy = field(default_factory=EnhancementPolicy)
    training_policy: TrainingPolicy = field(default_factory=TrainingPolicy)
    inference_policy: InferencePolicy = field(default_factory=InferencePolicy)
    lcm_integration_policy: LCMIntegrationPolicy = field(default_factory=LCMIntegrationPolicy)
    compliance_policy: CompliancePolicy = field(default_factory=CompliancePolicy)
    performance_policy: PerformancePolicy = field(default_factory=PerformancePolicy)
    
    # Protocol implementations (optional, for dependency injection)
    model_adapter: Optional["ModelAdapter"] = None
    metadata_provider: Optional["ModelMetadataProvider"] = None
    model_validator: Optional["ModelValidator"] = None
    training_handler: Optional["ModelTrainingHandler"] = None
    inference_handler: Optional["ModelInferenceHandler"] = None
    lcm_metadata_handler: Optional["LCMMetadataHandler"] = None
    enhancement_provider: Optional["ModelEnhancementProvider"] = None
    compliance_integrator: Optional["ComplianceIntegrator"] = None
    performance_optimizer: Optional["PerformanceOptimizer"] = None
    
    def __post_init__(self):
        """Initialize default protocol implementations if not provided."""
        if any(impl is None for impl in [
            self.model_adapter, self.metadata_provider, self.model_validator,
            self.training_handler, self.inference_handler, self.lcm_metadata_handler,
            self.enhancement_provider, self.compliance_integrator, self.performance_optimizer
        ]):
            self._init_default_protocols()
    
    def _init_default_protocols(self):
        """Initialize default protocol implementations."""
        # Import here to avoid circular imports and check availability
        try:
            from .protocol_implementations import create_default_wrapper_protocols
            defaults = create_default_wrapper_protocols()
            
            if self.model_adapter is None:
                self.model_adapter = defaults['model_adapter']
            if self.metadata_provider is None:
                self.metadata_provider = defaults['metadata_provider']
            if self.model_validator is None:
                self.model_validator = defaults['model_validator']
            if self.training_handler is None:
                self.training_handler = defaults['training_handler']
            if self.inference_handler is None:
                self.inference_handler = defaults['inference_handler']
            if self.lcm_metadata_handler is None:
                self.lcm_metadata_handler = defaults['lcm_metadata_handler']
            if self.enhancement_provider is None:
                self.enhancement_provider = defaults['enhancement_provider']
            if self.compliance_integrator is None:
                self.compliance_integrator = defaults['compliance_integrator']
            if self.performance_optimizer is None:
                self.performance_optimizer = defaults['performance_optimizer']
        except ImportError as e:
            # Create placeholder None values if implementations not available
            import warnings
            warnings.warn(f"Protocol implementations not available: {e}")
            
            # Set all to None if not already set
            if self.model_adapter is None:
                self.model_adapter = None
            if self.metadata_provider is None:
                self.metadata_provider = None
            if self.model_validator is None:
                self.model_validator = None
            if self.training_handler is None:
                self.training_handler = None
            if self.inference_handler is None:
                self.inference_handler = None
            if self.lcm_metadata_handler is None:
                self.lcm_metadata_handler = None
            if self.enhancement_provider is None:
                self.enhancement_provider = None
            if self.compliance_integrator is None:
                self.compliance_integrator = None
            if self.performance_optimizer is None:
                self.performance_optimizer = None
    
    @classmethod
    def development(cls) -> "WrapperPolicy":
        """Create a development-friendly policy."""
        policy = cls()
        policy.wrapper_mode = WrapperMode.DEVELOPMENT
        
        # Relaxed compatibility
        policy.compatibility_policy.strict_type_validation = False
        policy.compatibility_policy.fail_on_incompatible_model = False
        policy.compatibility_policy.fallback_to_simulation = True
        
        # Enable all enhancements with graceful failure
        policy.enhancement_policy.enable_preprocessing = True
        policy.enhancement_policy.enable_explainability = True
        policy.enhancement_policy.enable_uncertainty = True
        policy.enhancement_policy.graceful_enhancement_failure = True
        
        # Detailed logging for development
        policy.training_policy.log_training_details = True
        policy.inference_policy.log_inference_details = True
        policy.performance_policy.log_performance_metrics = True
        
        return policy
    
    @classmethod 
    def production(cls) -> "WrapperPolicy":
        """Create a production-ready policy."""
        policy = cls()
        policy.wrapper_mode = WrapperMode.PRODUCTION
        
        # Strict compatibility validation
        policy.compatibility_policy.strict_type_validation = True
        policy.compatibility_policy.validate_model_structure = True
        policy.compatibility_policy.validate_model_readiness = True
        
        # Production-grade performance
        policy.performance_policy.performance_level = PerformanceLevel.OPTIMIZED
        policy.performance_policy.enable_model_caching = True
        policy.performance_policy.enable_preprocessing_caching = True
        policy.performance_policy.monitor_performance = True
        
        # Comprehensive LCM integration
        policy.lcm_integration_policy.enable_lcm_integration = True
        policy.lcm_integration_policy.preserve_lcm_metadata = True
        policy.lcm_integration_policy.verify_metadata_integrity = True
        
        # Enhanced compliance
        policy.compliance_policy.enable_audit_trails = True
        policy.compliance_policy.detailed_audit_logging = True
        policy.compliance_policy.generate_compliance_reports = True
        
        return policy
    
    @classmethod
    def compliance_strict(cls) -> "WrapperPolicy":
        """Create a strict compliance policy."""
        policy = cls()
        policy.wrapper_mode = WrapperMode.COMPLIANCE
        
        # Strict compliance mode
        policy.compliance_policy.compliance_mode = ComplianceMode.STRICT
        policy.compliance_policy.validate_regulatory_compliance = True
        policy.compliance_policy.fail_on_compliance_violations = True
        policy.compliance_policy.alert_on_compliance_issues = True
        
        # Comprehensive audit trails
        policy.compliance_policy.enable_audit_trails = True
        policy.compliance_policy.detailed_audit_logging = True
        policy.compliance_policy.retain_audit_data = True
        
        # Enhanced security
        policy.compliance_policy.anonymize_sensitive_data = True
        policy.compliance_policy.encrypt_stored_metadata = True
        policy.compliance_policy.secure_model_storage = True
        
        # Strict validation
        policy.compatibility_policy.strict_type_validation = True
        policy.compatibility_policy.fail_on_incompatible_model = True
        policy.training_policy.validate_training_data = True
        policy.inference_policy.validate_inference_input = True
        
        return policy
    
    @classmethod
    def high_performance(cls) -> "WrapperPolicy":
        """Create a high-performance optimized policy."""
        policy = cls()
        
        # Maximum performance optimization
        policy.performance_policy.performance_level = PerformanceLevel.MAXIMUM
        policy.performance_policy.enable_model_caching = True
        policy.performance_policy.enable_preprocessing_caching = True
        policy.performance_policy.enable_result_caching = True
        policy.performance_policy.enable_batch_processing = True
        
        # Deferred LCM processing for speed
        policy.lcm_integration_policy.defer_lcm_processing = True
        policy.lcm_integration_policy.batch_lcm_operations = True
        
        # Optimized inference
        policy.inference_policy.enable_inference_caching = True
        policy.inference_policy.cache_explanations = True
        
        # Reduced validation for speed
        policy.compatibility_policy.strict_type_validation = False
        policy.training_policy.validate_training_data = False
        policy.inference_policy.validate_inference_input = False
        
        return policy
    
    @classmethod
    def create_compliance_policy(cls) -> "WrapperPolicy":
        """Create a compliance-focused policy (alias for compliance_strict)."""
        return cls.compliance_strict()
    
    @classmethod
    def create_production_policy(cls) -> "WrapperPolicy":
        """Create a production-ready policy (alias for production)."""
        return cls.production()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary format."""
        def serialize_value(value):
            """Recursively serialize values, handling enums and complex types."""
            if hasattr(value, 'value'):  # Enum
                return value.value
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value
        
        def serialize_dataclass(obj):
            """Serialize a dataclass object."""
            return {k: serialize_value(v) for k, v in obj.__dict__.items()}
        
        return {
            "policy_version": self.policy_version,
            "created_date": self.created_date,
            "organization": self.organization,
            "wrapper_mode": self.wrapper_mode.value,
            "compatibility_policy": serialize_dataclass(self.compatibility_policy),
            "enhancement_policy": serialize_dataclass(self.enhancement_policy),
            "training_policy": serialize_dataclass(self.training_policy),
            "inference_policy": serialize_dataclass(self.inference_policy),
            "lcm_integration_policy": serialize_dataclass(self.lcm_integration_policy),
            "compliance_policy": serialize_dataclass(self.compliance_policy),
            "performance_policy": serialize_dataclass(self.performance_policy)
        }
    
    def get_policy_hash(self) -> str:
        """Get cryptographic hash of policy for integrity verification."""
        policy_json = canonical_json(self.to_dict())
        return canonical_hash(policy_json)
    
    def format_policy_line(self) -> str:
        """Format policy as a single line for display."""
        return (f"wrapper: mode={self.wrapper_mode.value} | "
                f"compliance={self.compliance_policy.compliance_mode.value} | "
                f"performance={self.performance_policy.performance_level.value} | "
                f"enhancements={len([x for x in [self.enhancement_policy.enable_preprocessing, self.enhancement_policy.enable_explainability, self.enhancement_policy.enable_uncertainty, self.enhancement_policy.enable_metadata_tags] if x])} | "
                f"lcm_integration={self.lcm_integration_policy.enable_lcm_integration}")


# Global default policy
_default_wrapper_policy: Optional[WrapperPolicy] = None


def get_default_wrapper_policy() -> WrapperPolicy:
    """Get the default wrapper policy."""
    global _default_wrapper_policy
    if _default_wrapper_policy is None:
        _default_wrapper_policy = WrapperPolicy.development()
    return _default_wrapper_policy


def set_default_wrapper_policy(policy: WrapperPolicy) -> None:
    """Set the default wrapper policy."""
    global _default_wrapper_policy
    _default_wrapper_policy = policy


def create_wrapper_policy(mode: str = "development", **kwargs) -> WrapperPolicy:
    """Create a wrapper policy with the specified mode and overrides."""
    if mode == "development":
        policy = WrapperPolicy.development()
    elif mode == "production":
        policy = WrapperPolicy.production()
    elif mode == "compliance":
        policy = WrapperPolicy.compliance_strict()
    elif mode == "performance":
        policy = WrapperPolicy.high_performance()
    else:
        policy = WrapperPolicy()
    
    # Apply any keyword overrides
    for key, value in kwargs.items():
        # Handle enum conversions
        if key == "wrapper_mode":
            if isinstance(value, str):
                value = WrapperMode(value)
            setattr(policy, key, value)
        elif key == "compliance_mode":
            if isinstance(value, str):
                value = ComplianceMode(value)
            setattr(policy.compliance_policy, key, value)
        elif key == "performance_level":
            if isinstance(value, str):
                value = PerformanceLevel(value)
            setattr(policy.performance_policy, key, value)
        elif hasattr(policy, key):
            setattr(policy, key, value)
        else:
            # Try to find the attribute in sub-policies
            for sub_policy_name in ['compatibility_policy', 'enhancement_policy', 'training_policy', 
                                   'inference_policy', 'lcm_integration_policy', 'compliance_policy', 
                                   'performance_policy']:
                sub_policy = getattr(policy, sub_policy_name)
                if hasattr(sub_policy, key):
                    setattr(sub_policy, key, value)
                    break
    
    return policy