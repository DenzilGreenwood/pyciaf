"""
Preprocessing Policy Configuration for CIAF

This module provides policy-driven configuration for preprocessing operations,
following the same pattern as other CIAF modules.

Created: 2025-09-27
Author: CIAF Framework
Version: 1.0.0
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ciaf.lcm import canonical_json, canonical_hash


class QualityLevel(Enum):
    """Data quality enforcement levels."""
    PERMISSIVE = "permissive"      # Allow most data through with warnings
    STANDARD = "standard"          # Balanced quality enforcement  
    STRICT = "strict"              # Rigorous quality requirements
    RESEARCH = "research"          # Highest quality for research use


class PreprocessingIntensity(Enum):
    """Preprocessing intensity levels."""
    MINIMAL = "minimal"            # Basic cleaning only
    STANDARD = "standard"          # Standard preprocessing pipeline
    COMPREHENSIVE = "comprehensive" # Full preprocessing with feature engineering
    CUSTOM = "custom"              # Custom preprocessing configuration


@dataclass
class QualityPolicy:
    """Data quality validation policy configuration."""
    
    # Basic validation settings
    min_samples: int = 10
    max_missing_ratio: float = 0.3
    min_unique_ratio: float = 0.01
    
    # Outlier detection
    enable_outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 0.1
    
    # Duplicate handling
    check_duplicates: bool = True
    duplicate_threshold: float = 0.05
    
    # Distribution monitoring
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    
    # Schema validation
    enforce_schema: bool = True
    allow_extra_columns: bool = True
    
    # Quality scoring
    min_quality_score: float = 70.0
    quality_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quality_weights is None:
            self.quality_weights = {
                "completeness": 0.3,
                "consistency": 0.2, 
                "validity": 0.2,
                "uniqueness": 0.15,
                "accuracy": 0.15
            }


@dataclass  
class ProcessingPolicy:
    """Data preprocessing pipeline policy configuration."""
    
    # Text processing
    text_vectorization_method: str = "tfidf"  # "tfidf", "count", "word2vec", "bert"
    text_max_features: int = 10000
    text_ngram_range: tuple = (1, 2)
    text_stop_words: str = "english"
    text_lowercase: bool = True
    
    # Numerical processing
    numerical_scaling: str = "standard"  # "standard", "minmax", "robust", "quantile"
    handle_outliers: str = "clip"        # "clip", "remove", "transform", "keep"
    outlier_percentiles: tuple = (5, 95)
    
    # Categorical processing
    categorical_encoding: str = "onehot"  # "onehot", "label", "target", "binary"
    handle_unknown_categories: str = "error"  # "error", "ignore", "infrequent"
    max_categories: int = 100
    
    # Feature engineering
    enable_feature_selection: bool = True
    feature_selection_method: str = "variance"  # "variance", "correlation", "univariate", "rfe"
    max_features_ratio: float = 0.8
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = False
    reduction_method: str = "pca"  # "pca", "ica", "tsne", "umap"
    n_components: Optional[int] = None
    explained_variance_ratio: float = 0.95
    
    # Processing options
    parallelize: bool = True
    memory_efficient: bool = True
    cache_transformations: bool = True
    
    # Validation during preprocessing
    validate_after_each_step: bool = True
    fail_on_validation_error: bool = False


@dataclass
class PerformancePolicy:
    """Performance and resource management policy."""
    
    # Memory management
    max_memory_usage_mb: int = 2048
    chunk_size: int = 10000
    use_sparse_matrices: bool = True
    
    # Processing limits
    max_processing_time_seconds: int = 300
    max_features: int = 50000
    max_samples_for_fitting: int = 100000
    
    # Optimization settings
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    batch_size: int = 1000
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_persistence: bool = False


@dataclass
class SecurityPolicy:
    """Data security and privacy policy."""
    
    # Data sanitization
    enable_data_sanitization: bool = True
    remove_pii: bool = True
    anonymize_identifiers: bool = True
    
    # Data retention
    max_data_retention_days: int = 365
    auto_cleanup_temp_files: bool = True
    
    # Access control
    require_data_lineage: bool = True
    log_data_access: bool = True
    
    # Privacy protection
    enable_differential_privacy: bool = False
    privacy_budget: float = 1.0


class PreprocessingPolicy:
    """Comprehensive preprocessing policy configuration."""
    
    def __init__(self,
                 quality_policy: Optional[QualityPolicy] = None,
                 processing_policy: Optional[ProcessingPolicy] = None,
                 performance_policy: Optional[PerformancePolicy] = None,
                 security_policy: Optional[SecurityPolicy] = None,
                 custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing policy.
        
        Args:
            quality_policy: Data quality validation configuration
            processing_policy: Preprocessing pipeline configuration
            performance_policy: Performance and resource management
            security_policy: Security and privacy configuration
            custom_config: Additional custom configuration
        """
        self.quality_policy = quality_policy or QualityPolicy()
        self.processing_policy = processing_policy or ProcessingPolicy()
        self.performance_policy = performance_policy or PerformancePolicy()
        self.security_policy = security_policy or SecurityPolicy()
        self.custom_config = custom_config or {}
        
        # Policy metadata
        self._policy_version = "1.0.0"
        self._created_timestamp = None
        self._policy_id = None
    
    @classmethod
    def minimal(cls) -> PreprocessingPolicy:
        """Create minimal preprocessing policy for development/testing."""
        return cls(
            quality_policy=QualityPolicy(
                min_samples=5,
                max_missing_ratio=0.5,
                enable_outlier_detection=False,
                check_duplicates=False,
                min_quality_score=50.0
            ),
            processing_policy=ProcessingPolicy(
                text_max_features=1000,
                enable_feature_selection=False,
                enable_dimensionality_reduction=False,
                validate_after_each_step=False
            ),
            performance_policy=PerformancePolicy(
                max_memory_usage_mb=512,
                max_processing_time_seconds=60,
                enable_parallel_processing=False,
                enable_caching=False
            )
        )
    
    @classmethod
    def standard(cls) -> PreprocessingPolicy:
        """Create standard preprocessing policy for typical use cases."""
        return cls(
            quality_policy=QualityPolicy(
                min_samples=10,
                max_missing_ratio=0.3,
                enable_outlier_detection=True,
                check_duplicates=True,
                min_quality_score=70.0
            ),
            processing_policy=ProcessingPolicy(
                text_max_features=5000,
                enable_feature_selection=True,
                enable_dimensionality_reduction=False,
                validate_after_each_step=True
            ),
            performance_policy=PerformancePolicy(
                max_memory_usage_mb=1024,
                max_processing_time_seconds=180,
                enable_parallel_processing=True,
                enable_caching=True
            )
        )
    
    @classmethod  
    def comprehensive(cls) -> PreprocessingPolicy:
        """Create comprehensive preprocessing policy for production use."""
        return cls(
            quality_policy=QualityPolicy(
                min_samples=50,
                max_missing_ratio=0.1,
                enable_outlier_detection=True,
                check_duplicates=True,
                enable_drift_detection=True,
                min_quality_score=85.0
            ),
            processing_policy=ProcessingPolicy(
                text_max_features=20000,
                enable_feature_selection=True,
                enable_dimensionality_reduction=True,
                categorical_encoding="onehot",
                validate_after_each_step=True
            ),
            performance_policy=PerformancePolicy(
                max_memory_usage_mb=4096,
                max_processing_time_seconds=600,
                enable_parallel_processing=True,
                enable_caching=True,
                cache_persistence=True
            ),
            security_policy=SecurityPolicy(
                enable_data_sanitization=True,
                remove_pii=True,
                require_data_lineage=True,
                log_data_access=True
            )
        )
    
    @classmethod
    def research(cls) -> PreprocessingPolicy:
        """Create research-grade preprocessing policy with highest quality standards."""
        return cls(
            quality_policy=QualityPolicy(
                min_samples=100,
                max_missing_ratio=0.05,
                min_unique_ratio=0.05,
                enable_outlier_detection=True,
                outlier_method="isolation_forest",
                check_duplicates=True,
                enable_drift_detection=True,
                min_quality_score=90.0
            ),
            processing_policy=ProcessingPolicy(
                text_max_features=50000,
                text_vectorization_method="bert",
                enable_feature_selection=True,
                feature_selection_method="rfe",
                enable_dimensionality_reduction=True,
                reduction_method="pca",
                validate_after_each_step=True,
                fail_on_validation_error=True
            ),
            performance_policy=PerformancePolicy(
                max_memory_usage_mb=8192,
                max_processing_time_seconds=1800,
                enable_parallel_processing=True,
                enable_caching=True,
                cache_persistence=True
            ),
            security_policy=SecurityPolicy(
                enable_data_sanitization=True,
                remove_pii=True,
                require_data_lineage=True,
                log_data_access=True,
                enable_differential_privacy=True
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            "policy_version": self._policy_version,
            "quality_policy": self.quality_policy.__dict__,
            "processing_policy": self.processing_policy.__dict__,
            "performance_policy": self.performance_policy.__dict__,
            "security_policy": self.security_policy.__dict__,
            "custom_config": self.custom_config
        }
    
    def policy_digest(self) -> str:
        """Generate cryptographic hash of policy for integrity verification."""
        policy_dict = self.to_dict()
        canonical = canonical_json(policy_dict)
        return canonical_hash(canonical)
    
    def format_policy_line(self) -> str:
        """Format policy as a single descriptive line."""
        quality_level = "strict" if self.quality_policy.min_quality_score >= 85 else \
                      "standard" if self.quality_policy.min_quality_score >= 70 else "permissive"
        
        processing_features = []
        if self.processing_policy.enable_feature_selection:
            processing_features.append("feature_selection")
        if self.processing_policy.enable_dimensionality_reduction:
            processing_features.append("dim_reduction")
        if self.security_policy.enable_data_sanitization:
            processing_features.append("data_sanitization")
        
        return (f"preprocessing: quality={quality_level} | "
                f"text_method={self.processing_policy.text_vectorization_method} | "
                f"features={len(processing_features)} | "
                f"max_memory={self.performance_policy.max_memory_usage_mb}MB | "
                f"parallel={self.performance_policy.enable_parallel_processing}")


def get_default_preprocessing_policy() -> PreprocessingPolicy:
    """Get the default preprocessing policy."""
    return PreprocessingPolicy.standard()


def create_custom_policy(
    quality_level: QualityLevel = QualityLevel.STANDARD,
    processing_intensity: PreprocessingIntensity = PreprocessingIntensity.STANDARD,
    max_memory_mb: int = 1024,
    **kwargs
) -> PreprocessingPolicy:
    """
    Create a custom preprocessing policy with specific parameters.
    
    Args:
        quality_level: Overall quality enforcement level
        processing_intensity: Preprocessing complexity level
        max_memory_mb: Maximum memory usage in MB
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PreprocessingPolicy instance
    """
    # Base configurations for each level
    base_policies = {
        QualityLevel.PERMISSIVE: PreprocessingPolicy.minimal(),
        QualityLevel.STANDARD: PreprocessingPolicy.standard(),
        QualityLevel.STRICT: PreprocessingPolicy.comprehensive(),
        QualityLevel.RESEARCH: PreprocessingPolicy.research()
    }
    
    base_policy = base_policies[quality_level]
    
    # Adjust for processing intensity
    if processing_intensity == PreprocessingIntensity.MINIMAL:
        base_policy.processing_policy.enable_feature_selection = False
        base_policy.processing_policy.enable_dimensionality_reduction = False
        base_policy.processing_policy.text_max_features = min(1000, base_policy.processing_policy.text_max_features)
    elif processing_intensity == PreprocessingIntensity.COMPREHENSIVE:
        base_policy.processing_policy.enable_feature_selection = True
        base_policy.processing_policy.enable_dimensionality_reduction = True
        base_policy.processing_policy.text_max_features = max(20000, base_policy.processing_policy.text_max_features)
    
    # Apply memory constraints
    base_policy.performance_policy.max_memory_usage_mb = max_memory_mb
    
    # Apply any custom overrides
    for key, value in kwargs.items():
        if hasattr(base_policy, key):
            setattr(base_policy, key, value)
        else:
            base_policy.custom_config[key] = value
    
    return base_policy