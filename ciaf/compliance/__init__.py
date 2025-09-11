"""
Compliance Module for CIAF

This module provides comprehensive compliance capabilities for AI models,
including audit trails, regulatory mapping, validation, documentation,
risk assessment, transparency reporting, uncertainty quantification,
corrective action logging, stakeholder impact assessment, visualization,
and cybersecurity compliance.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .audit_trails import AuditEventType, AuditTrailGenerator, ComplianceAuditRecord
from .corrective_action_log import (
    ActionStatus,
    ActionType,
    CorrectiveAction,
    CorrectiveActionLogger,
    CorrectiveActionSummary,
    TriggerType,
)
from .cybersecurity import (
    ComplianceStatus,
    CybersecurityAssessment,
    CybersecurityComplianceEngine,
    SecurityControl,
    SecurityControlImplementation,
    SecurityFramework,
    SecurityLevel,
)
from .documentation import (
    ComplianceDocument,
    ComplianceDocumentationGenerator,
    DocumentationType,
    DocumentSection,
)
from .pre_ingestion_validator import (
    BiasDetectionResult,
    PreIngestionValidator,
    ValidationIssue,
)
from .regulatory_mapping import (
    ComplianceFramework,
    ComplianceRequirement,
    RegulatoryMapper,
)
from .reports import ComplianceReport, ComplianceReportGenerator, ReportType
from .risk_assessment import (
    BiasAssessment,
    ComprehensiveRiskAssessment,
    PerformanceAssessment,
    RiskAssessmentEngine,
    RiskCategory,
    RiskFactor,
    RiskLevel,
    RiskLikelihood,
    SecurityAssessment,
)
from .stakeholder_impact import (
    ComprehensiveStakeholderImpactAssessment,
    ImpactAssessment,
    ImpactCategory,
    ImpactSeverity,
    ImpactTimeline,
    StakeholderGroup,
    StakeholderImpactAssessmentEngine,
    StakeholderType,
)
from .transparency_reports import (
    AlgorithmicTransparencyMetrics,
    DecisionExplanation,
    ReportAudience,
    TransparencyLevel,
    TransparencyReport,
    TransparencyReportGenerator,
)
from .uncertainty_quantification import (
    ConfidenceInterval,
    UncertaintyMethod,
    UncertaintyMetrics,
    UncertaintyQuantifier,
)
from .validators import ComplianceValidator, ValidationResult, ValidationSeverity
from .visualization import (
    CIAFVisualizationEngine,
    ExportFormat,
    NodeType,
    VisualizationConfig,
    VisualizationEdge,
    VisualizationNode,
    VisualizationType,
)

# New Enhanced Modules for 360° AI Governance Compliance


__all__ = [
    # Audit Trails
    "AuditEventType",
    "ComplianceAuditRecord",
    "AuditTrailGenerator",
    # Regulatory Mapping
    "ComplianceFramework",
    "ComplianceRequirement",
    "RegulatoryMapper",
    # Reports
    "ReportType",
    "ComplianceReport",
    "ComplianceReportGenerator",
    # Validators
    "ValidationSeverity",
    "ValidationResult",
    "ComplianceValidator",
    # Documentation
    "DocumentationType",
    "DocumentSection",
    "ComplianceDocument",
    "ComplianceDocumentationGenerator",
    # Risk Assessment
    "RiskCategory",
    "RiskLevel",
    "RiskLikelihood",
    "RiskFactor",
    "BiasAssessment",
    "PerformanceAssessment",
    "SecurityAssessment",
    "ComprehensiveRiskAssessment",
    "RiskAssessmentEngine",
    # Transparency Reports
    "TransparencyLevel",
    "ReportAudience",
    "AlgorithmicTransparencyMetrics",
    "DecisionExplanation",
    "TransparencyReport",
    "TransparencyReportGenerator",
    # Pre-Ingestion Validation
    "ValidationIssue",
    "BiasDetectionResult",
    "PreIngestionValidator",
]
