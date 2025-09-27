"""
CIAF Compliance Policy Framework

Defines the canonical policies for compliance validation, audit trail generation,
and regulatory framework mapping used throughout the CIAF compliance system.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..core import sha256_hash
from ..lcm.policy import canonical_json
from .interfaces import (
    ComplianceFramework,
    ValidationSeverity,
    AuditEventType
)

if TYPE_CHECKING:
    from .interfaces import (
        ComplianceValidator,
        AuditTrailProvider,
        RiskAssessor,
        BiasDetector,
        DocumentationGenerator,
        ComplianceStore,
        AlertSystem
    )


class ComplianceLevel(Enum):
    """Compliance enforcement levels."""
    STRICT = "strict"          # All validations must pass
    STANDARD = "standard"      # Critical and high severity must pass
    ADVISORY = "advisory"      # Only track and report, no enforcement


class RetentionPeriod(Enum):
    """Data retention periods for compliance."""
    SHORT = "1_year"
    MEDIUM = "3_years"
    LONG = "7_years"
    PERMANENT = "permanent"


@dataclass
class AuditPolicy:
    """Audit trail generation and retention policy."""
    enabled: bool = True
    retention_period: RetentionPeriod = RetentionPeriod.LONG
    integrity_verification: bool = True
    encryption_required: bool = True
    real_time_alerts: bool = True
    event_types: List[AuditEventType] = None
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = [
                AuditEventType.MODEL_TRAINING,
                AuditEventType.INFERENCE_REQUEST,
                AuditEventType.DATA_ACCESS,
                AuditEventType.COMPLIANCE_CHECK,
                AuditEventType.RISK_ASSESSMENT
            ]


@dataclass
class ValidationPolicy:
    """Validation policy configuration."""
    enabled_frameworks: List[ComplianceFramework] = None
    validation_frequency: str = "daily"  # daily, weekly, monthly, on_demand
    auto_remediation: bool = False
    compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
    failure_threshold: Dict[ValidationSeverity, int] = None
    
    def __post_init__(self):
        if self.enabled_frameworks is None:
            self.enabled_frameworks = [
                ComplianceFramework.GENERAL,
                ComplianceFramework.NIST_AI_RMF,
                ComplianceFramework.ISO_27001
            ]
        if self.failure_threshold is None:
            self.failure_threshold = {
                ValidationSeverity.CRITICAL: 0,  # No critical failures allowed
                ValidationSeverity.HIGH: 2,     # Max 2 high severity failures
                ValidationSeverity.MEDIUM: 10,  # Max 10 medium severity failures
                ValidationSeverity.LOW: 50,     # Max 50 low severity failures
                ValidationSeverity.INFO: -1     # No limit on info messages
            }


@dataclass
class PrivacyPolicy:
    """Privacy and data protection policy."""
    pii_detection_enabled: bool = True
    anonymization_required: bool = True
    consent_tracking: bool = True
    data_minimization: bool = True
    right_to_erasure: bool = True
    cross_border_restrictions: List[str] = None
    
    def __post_init__(self):
        if self.cross_border_restrictions is None:
            self.cross_border_restrictions = []


@dataclass
class CompliancePolicy:
    """
    Comprehensive CIAF compliance policy defining validation rules, audit requirements,
    and protocol implementations for dependency injection.
    """
    
    # Core policy configuration
    policy_version: str = "1.0"
    effective_date: str = None
    organization_name: str = "CIAF Implementation"
    jurisdiction: List[str] = None
    
    # Sub-policies
    audit_policy: AuditPolicy = None
    validation_policy: ValidationPolicy = None
    privacy_policy: PrivacyPolicy = None
    
    # Integration settings
    lcm_integration: bool = True
    anchor_compliance_records: bool = True
    merkle_audit_integrity: bool = True
    
    # Protocol implementations (optional, for dependency injection)
    validator: Optional["ComplianceValidator"] = None
    audit_provider: Optional["AuditTrailProvider"] = None
    risk_assessor: Optional["RiskAssessor"] = None
    bias_detector: Optional["BiasDetector"] = None
    doc_generator: Optional["DocumentationGenerator"] = None
    compliance_store: Optional["ComplianceStore"] = None
    alert_system: Optional["AlertSystem"] = None
    
    def __post_init__(self):
        """Initialize default values and protocols."""
        if self.effective_date is None:
            self.effective_date = datetime.now().isoformat()
        if self.jurisdiction is None:
            self.jurisdiction = ["global"]
        if self.audit_policy is None:
            self.audit_policy = AuditPolicy()
        if self.validation_policy is None:
            self.validation_policy = ValidationPolicy()
        if self.privacy_policy is None:
            self.privacy_policy = PrivacyPolicy()
        
        # Initialize default protocol implementations if needed
        if not any([
            self.validator, self.audit_provider, self.risk_assessor,
            self.bias_detector, self.doc_generator, self.compliance_store,
            self.alert_system
        ]):
            self._init_default_protocols()
    
    def _init_default_protocols(self):
        """Initialize default protocol implementations."""
        # Import here to avoid circular imports
        try:
            from .protocol_implementations import create_default_compliance_protocols
            defaults = create_default_compliance_protocols()
            
            if self.validator is None:
                self.validator = defaults.get('validator')
            if self.audit_provider is None:
                self.audit_provider = defaults.get('audit_provider')
            if self.risk_assessor is None:
                self.risk_assessor = defaults.get('risk_assessor')
            if self.bias_detector is None:
                self.bias_detector = defaults.get('bias_detector')
            if self.doc_generator is None:
                self.doc_generator = defaults.get('doc_generator')
            if self.compliance_store is None:
                self.compliance_store = defaults.get('compliance_store')
            if self.alert_system is None:
                self.alert_system = defaults.get('alert_system')
        except ImportError:
            # If protocol implementations are not available, continue without them
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "policy_version": self.policy_version,
            "effective_date": self.effective_date,
            "organization_name": self.organization_name,
            "jurisdiction": self.jurisdiction,
            "audit_policy": {
                "enabled": self.audit_policy.enabled,
                "retention_period": self.audit_policy.retention_period.value,
                "integrity_verification": self.audit_policy.integrity_verification,
                "encryption_required": self.audit_policy.encryption_required,
                "real_time_alerts": self.audit_policy.real_time_alerts,
                "event_types": [et.value for et in self.audit_policy.event_types]
            },
            "validation_policy": {
                "enabled_frameworks": [f.value for f in self.validation_policy.enabled_frameworks],
                "validation_frequency": self.validation_policy.validation_frequency,
                "auto_remediation": self.validation_policy.auto_remediation,
                "compliance_level": self.validation_policy.compliance_level.value,
                "failure_threshold": {
                    sev.value: threshold 
                    for sev, threshold in self.validation_policy.failure_threshold.items()
                }
            },
            "privacy_policy": {
                "pii_detection_enabled": self.privacy_policy.pii_detection_enabled,
                "anonymization_required": self.privacy_policy.anonymization_required,
                "consent_tracking": self.privacy_policy.consent_tracking,
                "data_minimization": self.privacy_policy.data_minimization,
                "right_to_erasure": self.privacy_policy.right_to_erasure,
                "cross_border_restrictions": self.privacy_policy.cross_border_restrictions
            },
            "integration": {
                "lcm_integration": self.lcm_integration,
                "anchor_compliance_records": self.anchor_compliance_records,
                "merkle_audit_integrity": self.merkle_audit_integrity
            }
        }
    
    def canonical_json(self) -> str:
        """Get canonical JSON representation."""
        return canonical_json(self.to_dict())
    
    def policy_digest(self) -> str:
        """Get digest of the policy itself."""
        return sha256_hash(self.canonical_json().encode('utf-8'))
    
    def format_policy_summary(self) -> str:
        """Format policy summary for pretty printing."""
        return (
            f"CIAF Compliance Policy v{self.policy_version}\n"
            f"Organization: {self.organization_name}\n"
            f"Jurisdiction: {', '.join(self.jurisdiction)}\n"
            f"Frameworks: {[f.value for f in self.validation_policy.enabled_frameworks]}\n"
            f"Compliance Level: {self.validation_policy.compliance_level.value}\n"
            f"Audit Retention: {self.audit_policy.retention_period.value}\n"
            f"LCM Integration: {'Enabled' if self.lcm_integration else 'Disabled'}"
        )
    
    def is_framework_enabled(self, framework: ComplianceFramework) -> bool:
        """Check if a compliance framework is enabled."""
        return framework in self.validation_policy.enabled_frameworks
    
    def get_failure_threshold(self, severity: ValidationSeverity) -> int:
        """Get failure threshold for a given severity level."""
        return self.validation_policy.failure_threshold.get(severity, 0)
    
    def should_track_event(self, event_type: AuditEventType) -> bool:
        """Check if an event type should be tracked in audit trail."""
        return (self.audit_policy.enabled and 
                event_type in self.audit_policy.event_types)
    
    @classmethod
    def default(cls) -> "CompliancePolicy":
        """Get default CIAF compliance policy."""
        return cls()
    
    @classmethod
    def strict(cls) -> "CompliancePolicy":
        """Get strict compliance policy for high-risk environments."""
        policy = cls()
        policy.validation_policy.compliance_level = ComplianceLevel.STRICT
        policy.validation_policy.enabled_frameworks = [
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.NIST_AI_RMF,
            ComplianceFramework.ISO_27001,
            ComplianceFramework.GDPR
        ]
        policy.validation_policy.failure_threshold = {
            ValidationSeverity.CRITICAL: 0,
            ValidationSeverity.HIGH: 0,
            ValidationSeverity.MEDIUM: 5,
            ValidationSeverity.LOW: 20,
            ValidationSeverity.INFO: -1
        }
        policy.audit_policy.retention_period = RetentionPeriod.PERMANENT
        policy.privacy_policy.anonymization_required = True
        policy.privacy_policy.consent_tracking = True
        return policy
    
    @classmethod
    def development(cls) -> "CompliancePolicy":
        """Get development-friendly compliance policy."""
        policy = cls()
        policy.validation_policy.compliance_level = ComplianceLevel.ADVISORY
        policy.validation_policy.enabled_frameworks = [ComplianceFramework.GENERAL]
        policy.audit_policy.retention_period = RetentionPeriod.SHORT
        policy.audit_policy.real_time_alerts = False
        return policy


# Global default policy instance
DEFAULT_COMPLIANCE_POLICY = CompliancePolicy.default()


def get_default_compliance_policy() -> CompliancePolicy:
    """Get the default compliance policy."""
    return DEFAULT_COMPLIANCE_POLICY


def set_default_compliance_policy(policy: CompliancePolicy) -> None:
    """Set the global default compliance policy."""
    global DEFAULT_COMPLIANCE_POLICY
    DEFAULT_COMPLIANCE_POLICY = policy