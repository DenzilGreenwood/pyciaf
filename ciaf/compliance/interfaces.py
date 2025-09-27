"""
Compliance Interfaces for CIAF

This module defines Protocol interfaces for compliance components, following
the same pattern as the core module to enable clean dependency injection
and testing.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ComplianceFramework(Enum):
    """Supported regulatory and compliance frameworks."""
    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    FDA_AI_ML = "fda_ai_ml"
    FAIR_LENDING = "fair_lending"
    MODEL_RISK_MANAGEMENT = "model_risk_management"
    GENERAL = "general"


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditEventType(Enum):
    """Types of audit events tracked by CIAF."""
    MODEL_TRAINING = "model_training"
    DATA_INGESTION = "data_ingestion"
    INFERENCE_REQUEST = "inference_request"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_UPDATE = "model_update"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    USER_ACCESS = "user_access"


@runtime_checkable
class ComplianceValidator(Protocol):
    """Protocol for compliance validation implementations."""
    
    def validate_framework_compliance(
        self,
        framework: ComplianceFramework,
        audit_data: Dict[str, Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Validate compliance with a specific regulatory framework."""
        ...
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        ...


@runtime_checkable
class AuditTrailProvider(Protocol):
    """Protocol for audit trail generation and management."""
    
    def record_event(
        self,
        event_type: AuditEventType,
        event_data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Record an audit event and return event ID."""
        ...
    
    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered audit trail."""
        ...
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the cryptographic integrity of the audit trail."""
        ...


@runtime_checkable
class RiskAssessor(Protocol):
    """Protocol for risk assessment implementations."""
    
    def assess_model_risk(
        self,
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with model deployment."""
        ...
    
    def assess_data_risk(
        self,
        data_metadata: Dict[str, Any],
        usage_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with data usage."""
        ...


@runtime_checkable
class BiasDetector(Protocol):
    """Protocol for bias detection in AI models."""
    
    def detect_bias(
        self,
        predictions: Any,
        protected_attributes: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Detect bias in model predictions."""
        ...
    
    def calculate_fairness_metrics(
        self,
        predictions: Any,
        protected_attributes: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Calculate fairness metrics."""
        ...


@runtime_checkable
class DocumentationGenerator(Protocol):
    """Protocol for compliance documentation generation."""
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        model_metadata: Dict[str, Any],
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate compliance documentation."""
        ...
    
    def export_documentation(self, format: str = "pdf") -> bytes:
        """Export documentation in specified format."""
        ...


@runtime_checkable
class ComplianceStore(Protocol):
    """Protocol for storing compliance data and reports."""
    
    def store_validation_results(
        self,
        model_id: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Store validation results."""
        ...
    
    def get_compliance_history(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get compliance validation history."""
        ...


@runtime_checkable
class AlertSystem(Protocol):
    """Protocol for compliance alert and notification systems."""
    
    def send_compliance_alert(
        self,
        severity: ValidationSeverity,
        message: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Send compliance alert."""
        ...
    
    def configure_alert_rules(
        self,
        rules: List[Dict[str, Any]]
    ) -> None:
        """Configure alerting rules."""
        ...