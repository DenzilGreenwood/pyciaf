"""
Policy-driven configuration for explainability system.

This module provides configurable policies for explainable AI, similar to 
the LCM and compliance policy systems. It enables fine-tuned control over
explanation methods, quality thresholds, and regulatory compliance.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..lcm.policy import canonical_json
from ..core.crypto import sha256_hash


class ExplanationMethod(Enum):
    """Available explanation methods."""
    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear" 
    SHAP_KERNEL = "shap_kernel"
    LIME_TEXT = "lime_text"
    LIME_TABULAR = "lime_tabular"
    FEATURE_IMPORTANCE = "feature_importance"
    GRADIENT_BASED = "gradient_based"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    PERMUTATION_IMPORTANCE = "permutation_importance"


class ExplanationLevel(Enum):
    """Explanation detail levels."""
    MINIMAL = "minimal"          # Basic feature attributions only
    STANDARD = "standard"        # Standard explanations with confidence
    DETAILED = "detailed"        # Detailed with interactions and metadata
    COMPREHENSIVE = "comprehensive"  # Full explanations with all available info


class ComplianceFramework(Enum):
    """Supported compliance frameworks for explainability."""
    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    GDPR = "gdpr"
    ISO_23053 = "iso_23053"
    FAIR_ML = "fair_ml"


@dataclass
class ExplanationQualityPolicy:
    """Policy for explanation quality requirements."""
    
    # Confidence thresholds
    min_explanation_confidence: float = 0.7
    min_feature_coverage: int = 5
    max_features_in_explanation: int = 20
    
    # Feature attribution requirements
    require_feature_names: bool = True
    require_attribution_values: bool = True
    require_importance_ranking: bool = True
    
    # Quality validation
    validate_explanation_consistency: bool = True
    explanation_stability_threshold: float = 0.8
    
    # Audit requirements
    store_explanations: bool = True
    explanation_retention_days: int = 365
    require_explanation_audit_trail: bool = True


@dataclass
class ComplianceRequirements:
    """Compliance requirements for explainability."""
    
    enabled_frameworks: Set[ComplianceFramework] = field(
        default_factory=lambda: {ComplianceFramework.EU_AI_ACT, ComplianceFramework.NIST_AI_RMF}
    )
    
    # EU AI Act requirements
    eu_ai_act_transparency_level: str = "high"  # low, medium, high
    eu_ai_act_require_human_oversight: bool = True
    
    # NIST AI RMF requirements  
    nist_explain_function_level: str = "detailed"  # basic, detailed, comprehensive
    nist_require_bias_assessment: bool = True
    
    # GDPR requirements
    gdpr_right_to_explanation: bool = True
    gdpr_automated_decision_threshold: float = 0.5
    
    # General compliance
    require_regulatory_metadata: bool = True
    compliance_documentation_required: bool = True


@dataclass
class MethodPreferences:
    """Preferences for explanation method selection."""
    
    # Preferred methods in order of preference
    preferred_methods: List[ExplanationMethod] = field(
        default_factory=lambda: [
            ExplanationMethod.SHAP_TREE,
            ExplanationMethod.SHAP_LINEAR, 
            ExplanationMethod.LIME_TABULAR,
            ExplanationMethod.FEATURE_IMPORTANCE
        ]
    )
    
    # Method-specific configurations
    shap_background_sample_size: int = 100
    shap_max_evaluations: int = 500
    lime_num_perturbations: int = 1000
    lime_kernel_width: Optional[float] = None
    
    # Fallback behavior
    enable_fallback_methods: bool = True
    fallback_to_feature_importance: bool = True
    

@dataclass 
class IntegrationPolicy:
    """Policy for integration with other CIAF components."""
    
    # LCM integration
    lcm_integration: bool = True
    store_explanations_in_lcm: bool = True
    lcm_explanation_commitment_type: str = "salted"
    
    # Compliance integration
    compliance_integration: bool = True
    auto_generate_compliance_reports: bool = True
    
    # Audit integration
    audit_integration: bool = True
    explanation_audit_sampling_rate: float = 0.1  # 10% of explanations audited


@dataclass
class PerformancePolicy:
    """Policy for explainability performance management."""
    
    # Resource limits
    max_explanation_time_seconds: float = 30.0
    max_memory_usage_mb: int = 1024
    enable_explanation_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Batch processing
    enable_batch_explanations: bool = True
    batch_size: int = 100
    max_concurrent_explanations: int = 5
    
    # Background processing
    enable_async_explanations: bool = False
    async_explanation_timeout_seconds: float = 300.0


@dataclass
class ExplainabilityPolicy:
    """Comprehensive explainability policy configuration."""
    
    # Core policy components
    explanation_level: ExplanationLevel = ExplanationLevel.STANDARD
    quality_policy: ExplanationQualityPolicy = field(default_factory=ExplanationQualityPolicy)
    compliance_requirements: ComplianceRequirements = field(default_factory=ComplianceRequirements)
    method_preferences: MethodPreferences = field(default_factory=MethodPreferences)
    integration_policy: IntegrationPolicy = field(default_factory=IntegrationPolicy)
    performance_policy: PerformancePolicy = field(default_factory=PerformancePolicy)
    
    # Metadata
    policy_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = "CIAF Explainability Policy"
    
    @classmethod
    def minimal(cls) -> "ExplainabilityPolicy":
        """Create minimal explainability policy for development."""
        return cls(
            explanation_level=ExplanationLevel.MINIMAL,
            quality_policy=ExplanationQualityPolicy(
                min_explanation_confidence=0.5,
                min_feature_coverage=3,
                store_explanations=False,
                require_explanation_audit_trail=False
            ),
            compliance_requirements=ComplianceRequirements(
                enabled_frameworks=set(),
                require_regulatory_metadata=False,
                compliance_documentation_required=False
            ),
            method_preferences=MethodPreferences(
                preferred_methods=[ExplanationMethod.FEATURE_IMPORTANCE],
                enable_fallback_methods=True
            ),
            integration_policy=IntegrationPolicy(
                lcm_integration=False,
                compliance_integration=False,
                audit_integration=False
            ),
            description="Minimal explainability for development"
        )
    
    @classmethod
    def standard(cls) -> "ExplainabilityPolicy":
        """Create standard explainability policy for production."""
        return cls(
            explanation_level=ExplanationLevel.STANDARD,
            description="Standard explainability for production use"
        )
    
    @classmethod
    def comprehensive(cls) -> "ExplainabilityPolicy":
        """Create comprehensive explainability policy for high-risk AI."""
        return cls(
            explanation_level=ExplanationLevel.COMPREHENSIVE,
            quality_policy=ExplanationQualityPolicy(
                min_explanation_confidence=0.9,
                min_feature_coverage=10,
                max_features_in_explanation=50,
                validate_explanation_consistency=True,
                explanation_stability_threshold=0.9,
                explanation_retention_days=2555  # 7 years
            ),
            compliance_requirements=ComplianceRequirements(
                enabled_frameworks={
                    ComplianceFramework.EU_AI_ACT,
                    ComplianceFramework.NIST_AI_RMF,
                    ComplianceFramework.GDPR,
                    ComplianceFramework.ISO_23053
                },
                eu_ai_act_transparency_level="high",
                nist_explain_function_level="comprehensive",
                gdpr_right_to_explanation=True,
                compliance_documentation_required=True
            ),
            method_preferences=MethodPreferences(
                preferred_methods=[
                    ExplanationMethod.SHAP_TREE,
                    ExplanationMethod.SHAP_LINEAR,
                    ExplanationMethod.SHAP_KERNEL,
                    ExplanationMethod.LIME_TABULAR,
                    ExplanationMethod.INTEGRATED_GRADIENTS
                ],
                shap_background_sample_size=500,
                lime_num_perturbations=5000
            ),
            performance_policy=PerformancePolicy(
                max_explanation_time_seconds=120.0,
                max_memory_usage_mb=4096,
            ),
            integration_policy=IntegrationPolicy(
                explanation_audit_sampling_rate=0.25  # 25% audit rate
            ),
            description="Comprehensive explainability for high-risk AI systems"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "explanation_level": self.explanation_level.value,
            "quality_policy": {
                "min_explanation_confidence": self.quality_policy.min_explanation_confidence,
                "min_feature_coverage": self.quality_policy.min_feature_coverage,
                "max_features_in_explanation": self.quality_policy.max_features_in_explanation,
                "require_feature_names": self.quality_policy.require_feature_names,
                "require_attribution_values": self.quality_policy.require_attribution_values,
                "require_importance_ranking": self.quality_policy.require_importance_ranking,
                "validate_explanation_consistency": self.quality_policy.validate_explanation_consistency,
                "explanation_stability_threshold": self.quality_policy.explanation_stability_threshold,
                "store_explanations": self.quality_policy.store_explanations,
                "explanation_retention_days": self.quality_policy.explanation_retention_days,
                "require_explanation_audit_trail": self.quality_policy.require_explanation_audit_trail,
            },
            "compliance_requirements": {
                "enabled_frameworks": [f.value for f in self.compliance_requirements.enabled_frameworks],
                "eu_ai_act_transparency_level": self.compliance_requirements.eu_ai_act_transparency_level,
                "eu_ai_act_require_human_oversight": self.compliance_requirements.eu_ai_act_require_human_oversight,
                "nist_explain_function_level": self.compliance_requirements.nist_explain_function_level,
                "nist_require_bias_assessment": self.compliance_requirements.nist_require_bias_assessment,
                "gdpr_right_to_explanation": self.compliance_requirements.gdpr_right_to_explanation,
                "gdpr_automated_decision_threshold": self.compliance_requirements.gdpr_automated_decision_threshold,
                "require_regulatory_metadata": self.compliance_requirements.require_regulatory_metadata,
                "compliance_documentation_required": self.compliance_requirements.compliance_documentation_required,
            },
            "method_preferences": {
                "preferred_methods": [m.value for m in self.method_preferences.preferred_methods],
                "shap_background_sample_size": self.method_preferences.shap_background_sample_size,
                "shap_max_evaluations": self.method_preferences.shap_max_evaluations,
                "lime_num_perturbations": self.method_preferences.lime_num_perturbations,
                "lime_kernel_width": self.method_preferences.lime_kernel_width,
                "enable_fallback_methods": self.method_preferences.enable_fallback_methods,
                "fallback_to_feature_importance": self.method_preferences.fallback_to_feature_importance,
            },
            "integration_policy": {
                "lcm_integration": self.integration_policy.lcm_integration,
                "store_explanations_in_lcm": self.integration_policy.store_explanations_in_lcm,
                "lcm_explanation_commitment_type": self.integration_policy.lcm_explanation_commitment_type,
                "compliance_integration": self.integration_policy.compliance_integration,
                "auto_generate_compliance_reports": self.integration_policy.auto_generate_compliance_reports,
                "audit_integration": self.integration_policy.audit_integration,
                "explanation_audit_sampling_rate": self.integration_policy.explanation_audit_sampling_rate,
            },
            "performance_policy": {
                "max_explanation_time_seconds": self.performance_policy.max_explanation_time_seconds,
                "max_memory_usage_mb": self.performance_policy.max_memory_usage_mb,
                "enable_explanation_caching": self.performance_policy.enable_explanation_caching,
                "cache_ttl_hours": self.performance_policy.cache_ttl_hours,
                "enable_batch_explanations": self.performance_policy.enable_batch_explanations,
                "batch_size": self.performance_policy.batch_size,
                "max_concurrent_explanations": self.performance_policy.max_concurrent_explanations,
                "enable_async_explanations": self.performance_policy.enable_async_explanations,
                "async_explanation_timeout_seconds": self.performance_policy.async_explanation_timeout_seconds,
            },
            "policy_version": self.policy_version,
            "created_at": self.created_at,
            "description": self.description,
        }
    
    def canonical_json(self) -> str:
        """Get canonical JSON representation for hashing."""
        return canonical_json(self.to_dict())
    
    def policy_digest(self) -> str:
        """Get SHA-256 hash digest of policy."""
        return sha256_hash(self.canonical_json().encode('utf-8'))
    
    def format_policy_line(self) -> str:
        """Get human-readable policy summary line."""
        framework_count = len(self.compliance_requirements.enabled_frameworks)
        method_count = len(self.method_preferences.preferred_methods)
        return (
            f"explainability: level={self.explanation_level.value} | "
            f"min_confidence={self.quality_policy.min_explanation_confidence} | "
            f"frameworks={framework_count} | methods={method_count} | "
            f"lcm_integration={self.integration_policy.lcm_integration}"
        )


# Global policy management
_default_explainability_policy: Optional[ExplainabilityPolicy] = None


def get_default_explainability_policy() -> ExplainabilityPolicy:
    """Get the default explainability policy."""
    global _default_explainability_policy
    if _default_explainability_policy is None:
        _default_explainability_policy = ExplainabilityPolicy.standard()
    return _default_explainability_policy


def set_default_explainability_policy(policy: ExplainabilityPolicy) -> None:
    """Set the default explainability policy."""
    global _default_explainability_policy
    _default_explainability_policy = policy


def create_explainability_policy(
    level: ExplanationLevel = ExplanationLevel.STANDARD,
    compliance_frameworks: Optional[Set[ComplianceFramework]] = None,
    lcm_integration: bool = True
) -> ExplainabilityPolicy:
    """Create a custom explainability policy.
    
    Args:
        level: Explanation detail level
        compliance_frameworks: Set of compliance frameworks to enable
        lcm_integration: Whether to enable LCM integration
        
    Returns:
        Configured explainability policy
    """
    policy = ExplainabilityPolicy.standard()
    policy.explanation_level = level
    
    if compliance_frameworks is not None:
        policy.compliance_requirements.enabled_frameworks = compliance_frameworks
    
    policy.integration_policy.lcm_integration = lcm_integration
    
    return policy