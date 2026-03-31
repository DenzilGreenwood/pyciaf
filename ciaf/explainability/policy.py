"""
Policy-driven configuration for explainability system.

This module provides configurable policies for explainable AI, similar to
the LCM and compliance policy systems. It enables fine-tuned control over
explanation methods, quality thresholds, and regulatory compliance.

Created: 2025-09-26
Last Modified: 2026-03-30
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

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

    MINIMAL = "minimal"  # Basic feature attributions only
    STANDARD = "standard"  # Standard explanations with confidence
    DETAILED = "detailed"  # Detailed with interactions and metadata
    COMPREHENSIVE = "comprehensive"  # Full explanations with all available info


class ComplianceFramework(Enum):
    """Supported compliance frameworks for explainability."""

    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    GDPR = "gdpr"
    ISO_23053 = "iso_23053"
    FAIR_ML = "fair_ml"


class ExplanationQualityPolicy(BaseModel):
    """Policy for explanation quality requirements."""

    # Confidence thresholds
    min_explanation_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    min_feature_coverage: int = Field(5, ge=1, description="Minimum feature coverage")
    max_features_in_explanation: int = Field(
        20, ge=1, description="Maximum features in explanation"
    )

    # Feature attribution requirements
    require_feature_names: bool = Field(True, description="Require feature names")
    require_attribution_values: bool = Field(
        True, description="Require attribution values"
    )
    require_importance_ranking: bool = Field(
        True, description="Require importance ranking"
    )

    # Quality validation
    validate_explanation_consistency: bool = Field(
        True, description="Validate consistency"
    )
    explanation_stability_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Stability threshold"
    )

    # Audit requirements
    store_explanations: bool = Field(True, description="Store explanations")
    explanation_retention_days: int = Field(365, ge=1, description="Retention days")
    require_explanation_audit_trail: bool = Field(
        True, description="Require audit trail"
    )


class ComplianceRequirements(BaseModel):
    """Compliance requirements for explainability."""

    enabled_frameworks: Set[ComplianceFramework] = Field(
        default_factory=lambda: {
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.NIST_AI_RMF,
        },
        description="Enabled compliance frameworks",
    )

    # EU AI Act requirements
    eu_ai_act_transparency_level: str = Field(
        "high", description="EU AI Act transparency level"
    )  # low, medium, high
    eu_ai_act_require_human_oversight: bool = Field(
        True, description="Require human oversight"
    )

    # NIST AI RMF requirements
    nist_explain_function_level: str = Field(
        "detailed", description="NIST function level"
    )  # basic, detailed, comprehensive
    nist_require_bias_assessment: bool = Field(
        True, description="Require bias assessment"
    )

    # GDPR requirements
    gdpr_right_to_explanation: bool = Field(
        True, description="GDPR right to explanation"
    )
    gdpr_automated_decision_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Automated decision threshold"
    )

    # General compliance
    require_regulatory_metadata: bool = Field(
        True, description="Require regulatory metadata"
    )
    compliance_documentation_required: bool = Field(
        True, description="Compliance documentation required"
    )


class MethodPreferences(BaseModel):
    """Preferences for explanation method selection."""

    # Preferred methods in order of preference
    preferred_methods: List[ExplanationMethod] = Field(
        default_factory=lambda: [
            ExplanationMethod.SHAP_TREE,
            ExplanationMethod.SHAP_LINEAR,
            ExplanationMethod.LIME_TABULAR,
            ExplanationMethod.FEATURE_IMPORTANCE,
        ],
        description="Preferred explanation methods",
    )

    # Method-specific configurations
    shap_background_sample_size: int = Field(
        100, ge=1, description="SHAP background sample size"
    )
    shap_max_evaluations: int = Field(500, ge=1, description="SHAP max evaluations")
    lime_num_perturbations: int = Field(
        1000, ge=1, description="LIME perturbation count"
    )
    lime_kernel_width: Optional[float] = Field(None, description="LIME kernel width")

    # Fallback behavior
    enable_fallback_methods: bool = Field(True, description="Enable fallback methods")
    fallback_to_feature_importance: bool = Field(
        True, description="Fallback to feature importance"
    )


class IntegrationPolicy(BaseModel):
    """Policy for integration with other CIAF components."""

    # LCM integration
    lcm_integration: bool = Field(True, description="Enable LCM integration")
    store_explanations_in_lcm: bool = Field(
        True, description="Store explanations in LCM"
    )
    lcm_explanation_commitment_type: str = Field(
        "salted", description="LCM commitment type"
    )

    # Compliance integration
    compliance_integration: bool = Field(
        True, description="Enable compliance integration"
    )
    auto_generate_compliance_reports: bool = Field(
        True, description="Auto-generate reports"
    )

    # Audit integration
    audit_integration: bool = Field(True, description="Enable audit integration")
    explanation_audit_sampling_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Audit sampling rate"
    )


class PerformancePolicy(BaseModel):
    """Policy for explainability performance management."""

    # Resource limits
    max_explanation_time_seconds: float = Field(
        30.0, gt=0.0, description="Max explanation time"
    )
    max_memory_usage_mb: int = Field(1024, ge=1, description="Max memory usage")
    enable_explanation_caching: bool = Field(True, description="Enable caching")
    cache_ttl_hours: int = Field(24, ge=1, description="Cache TTL hours")

    # Batch processing
    enable_batch_explanations: bool = Field(True, description="Enable batch processing")
    batch_size: int = Field(100, ge=1, description="Batch size")
    max_concurrent_explanations: int = Field(
        5, ge=1, description="Max concurrent explanations"
    )

    # Background processing
    enable_async_explanations: bool = Field(
        False, description="Enable async explanations"
    )
    async_explanation_timeout_seconds: float = Field(
        300.0, gt=0.0, description="Async timeout"
    )


class ExplainabilityPolicy(BaseModel):
    """Comprehensive explainability policy configuration."""

    # Core policy components
    explanation_level: ExplanationLevel = Field(
        ExplanationLevel.STANDARD, description="Explanation detail level"
    )
    quality_policy: ExplanationQualityPolicy = Field(
        default_factory=ExplanationQualityPolicy, description="Quality policy"
    )
    compliance_requirements: ComplianceRequirements = Field(
        default_factory=ComplianceRequirements, description="Compliance requirements"
    )
    method_preferences: MethodPreferences = Field(
        default_factory=MethodPreferences, description="Method preferences"
    )
    integration_policy: IntegrationPolicy = Field(
        default_factory=IntegrationPolicy, description="Integration policy"
    )
    performance_policy: PerformancePolicy = Field(
        default_factory=PerformancePolicy, description="Performance policy"
    )

    # Metadata
    policy_version: str = Field("1.0.0", description="Policy version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )
    description: str = Field(
        "CIAF Explainability Policy", description="Policy description"
    )

    @classmethod
    def minimal(cls) -> "ExplainabilityPolicy":
        """Create minimal explainability policy for development."""
        return cls(
            explanation_level=ExplanationLevel.MINIMAL,
            quality_policy=ExplanationQualityPolicy(
                min_explanation_confidence=0.5,
                min_feature_coverage=3,
                store_explanations=False,
                require_explanation_audit_trail=False,
            ),
            compliance_requirements=ComplianceRequirements(
                enabled_frameworks=set(),
                require_regulatory_metadata=False,
                compliance_documentation_required=False,
            ),
            method_preferences=MethodPreferences(
                preferred_methods=[ExplanationMethod.FEATURE_IMPORTANCE],
                enable_fallback_methods=True,
            ),
            integration_policy=IntegrationPolicy(
                lcm_integration=False,
                compliance_integration=False,
                audit_integration=False,
            ),
            description="Minimal explainability for development",
        )

    @classmethod
    def standard(cls) -> "ExplainabilityPolicy":
        """Create standard explainability policy for production."""
        return cls(
            explanation_level=ExplanationLevel.STANDARD,
            description="Standard explainability for production use",
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
                explanation_retention_days=2555,  # 7 years
            ),
            compliance_requirements=ComplianceRequirements(
                enabled_frameworks={
                    ComplianceFramework.EU_AI_ACT,
                    ComplianceFramework.NIST_AI_RMF,
                    ComplianceFramework.GDPR,
                    ComplianceFramework.ISO_23053,
                },
                eu_ai_act_transparency_level="high",
                nist_explain_function_level="comprehensive",
                gdpr_right_to_explanation=True,
                compliance_documentation_required=True,
            ),
            method_preferences=MethodPreferences(
                preferred_methods=[
                    ExplanationMethod.SHAP_TREE,
                    ExplanationMethod.SHAP_LINEAR,
                    ExplanationMethod.SHAP_KERNEL,
                    ExplanationMethod.LIME_TABULAR,
                    ExplanationMethod.INTEGRATED_GRADIENTS,
                ],
                shap_background_sample_size=500,
                lime_num_perturbations=5000,
            ),
            performance_policy=PerformancePolicy(
                max_explanation_time_seconds=120.0,
                max_memory_usage_mb=4096,
            ),
            integration_policy=IntegrationPolicy(
                explanation_audit_sampling_rate=0.25  # 25% audit rate
            ),
            description="Comprehensive explainability for high-risk AI systems",
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
                "enabled_frameworks": [
                    f.value for f in self.compliance_requirements.enabled_frameworks
                ],
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
                "preferred_methods": [
                    m.value for m in self.method_preferences.preferred_methods
                ],
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
        return sha256_hash(self.canonical_json().encode("utf-8"))

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
    lcm_integration: bool = True,
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
