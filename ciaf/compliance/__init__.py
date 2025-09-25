"""
Compliance Module for CIAF

This module provides comprehensive compliance capabilities for AI models,
including audit trails, regulatory mapping, validation, documentation,
risk assessment, transparency reporting, uncertainty quantification,
corrective action logging, stakeholder impact assessment, visualization,
and cybersecurity compliance.

Created: 2025-09-09
Last Modified: 2025-09-25
Author: Denzil James Greenwood
Version: 1.1.0
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
from .bias_validator import (
    BiasValidator,
    BiasMetric,
    BiasResult,
    BiasAssessment,
    generate_bias_report,
)

# New Enhanced Modules for 360° AI Governance Compliance

# Enterprise-Grade Advanced Features
try:
    from .human_oversight import (
        HumanOversightEngine,
        OversightAlert,
        OversightReview,
        AlertType,
        ReviewStatus
    )
    HUMAN_OVERSIGHT_AVAILABLE = True
except ImportError:
    HUMAN_OVERSIGHT_AVAILABLE = False

try:
    from .web_dashboard import (
        CIAFDashboard,
        DashboardData,
        create_dashboard
    )
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

try:
    from .robustness_testing import (
        RobustnessTestSuite,
        TestResult,
        RobustnessReport,
        TestType,
        TestSeverity,
        AdversarialTester,
        DistributionShiftTester,
        StressTester
    )
    ROBUSTNESS_TESTING_AVAILABLE = True
except ImportError:
    ROBUSTNESS_TESTING_AVAILABLE = False


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
    # Bias Validation
    "BiasValidator",
    "BiasMetric",
    "BiasResult",
    "BiasAssessment",
    "generate_bias_report",
    # Enterprise Advanced Features
    "HumanOversightEngine",
    "OversightAlert", 
    "OversightReview",
    "AlertType",
    "ReviewStatus",
    "CIAFDashboard",
    "DashboardData",
    "create_dashboard", 
    "RobustnessTestSuite",
    "TestResult",
    "RobustnessReport",
    "TestType",
    "TestSeverity",
    "AdversarialTester",
    "DistributionShiftTester",
    "StressTester",
    # Feature availability flags
    "HUMAN_OVERSIGHT_AVAILABLE",
    "WEB_DASHBOARD_AVAILABLE",
    "ROBUSTNESS_TESTING_AVAILABLE",
]
