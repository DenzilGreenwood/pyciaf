"""
Compliance Module for CIAF

This module provides comprehensive compliance capabilities for AI models,
including audit trails, regulatory mapping, validation, documentation,
risk assessment, transparency reporting, uncertainty quantification,
corrective action logging, stakeholder impact assessment, visualization,
and cybersecurity compliance.

Updated with new interfaces, policy framework, and better integration
with the CIAF core and LCM systems.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.2.0
"""

# Core interfaces and policy
from .interfaces import (
    ComplianceFramework,
    ValidationSeverity,
    AuditEventType,
    ComplianceValidator as IComplianceValidator,
    AuditTrailProvider,
    RiskAssessor,
    BiasDetector,
    DocumentationGenerator,
    ComplianceStore,
    AlertSystem,
)

from .policy import (
    CompliancePolicy,
    ComplianceLevel,
    RetentionPeriod,
    AuditPolicy,
    ValidationPolicy,
    PrivacyPolicy,
    get_default_compliance_policy,
    set_default_compliance_policy,
)

# Protocol implementations
try:
    from .protocol_implementations import (
        DefaultComplianceValidator,
        DefaultAuditTrailProvider,
        DefaultRiskAssessor,
        DefaultBiasDetector,
        InMemoryComplianceStore,
        NoOpAlertSystem,
        SimpleDocumentationGenerator,
        create_default_compliance_protocols,
    )
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    PROTOCOL_IMPLEMENTATIONS_AVAILABLE = False

# Core compliance modules
from .audit_trails import AuditTrailGenerator, ComplianceAuditRecord, AuditTrail
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
from .validators import ComplianceValidator, ValidationResult
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
    BiasAssessment as BiasValidatorAssessment,
    generate_bias_report,
)

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
    # Core interfaces and enums
    "ComplianceFramework",
    "ValidationSeverity", 
    "AuditEventType",
    "IComplianceValidator",
    "AuditTrailProvider",
    "RiskAssessor",
    "BiasDetector",
    "DocumentationGenerator",
    "ComplianceStore",
    "AlertSystem",
    
    # Policy framework
    "CompliancePolicy",
    "ComplianceLevel",
    "RetentionPeriod",
    "AuditPolicy",
    "ValidationPolicy", 
    "PrivacyPolicy",
    "get_default_compliance_policy",
    "set_default_compliance_policy",
    
    # Protocol implementations (only if available)
] + (["DefaultComplianceValidator",
    "DefaultAuditTrailProvider",
    "DefaultRiskAssessor",
    "DefaultBiasDetector",
    "InMemoryComplianceStore",
    "NoOpAlertSystem",
    "SimpleDocumentationGenerator",
    "create_default_compliance_protocols"] if PROTOCOL_IMPLEMENTATIONS_AVAILABLE else []) + [
    
    # Audit Trails
    "ComplianceAuditRecord",
    "AuditTrailGenerator",
    "AuditTrail",
    
    # Regulatory Mapping
    "ComplianceRequirement",
    "RegulatoryMapper",
    
    # Reports
    "ReportType",
    "ComplianceReport",
    "ComplianceReportGenerator",
    
    # Validators
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
    "BiasValidatorAssessment",
    "generate_bias_report",
    
    # Corrective Actions
    "ActionStatus",
    "ActionType", 
    "CorrectiveAction",
    "CorrectiveActionLogger",
    "CorrectiveActionSummary",
    "TriggerType",
    
    # Cybersecurity
    "ComplianceStatus",
    "CybersecurityAssessment",
    "CybersecurityComplianceEngine",
    "SecurityControl",
    "SecurityControlImplementation",
    "SecurityFramework",
    "SecurityLevel",
    
    # Stakeholder Impact
    "ComprehensiveStakeholderImpactAssessment",
    "ImpactAssessment",
    "ImpactCategory",
    "ImpactSeverity",
    "ImpactTimeline",
    "StakeholderGroup",
    "StakeholderImpactAssessmentEngine",
    "StakeholderType",
    
    # Uncertainty Quantification
    "ConfidenceInterval",
    "UncertaintyMethod",
    "UncertaintyMetrics",
    "UncertaintyQuantifier",
    
    # Visualization
    "CIAFVisualizationEngine",
    "ExportFormat",
    "NodeType",
    "VisualizationConfig",
    "VisualizationEdge",
    "VisualizationNode",
    "VisualizationType",
    
    # Feature availability flags
    "PROTOCOL_IMPLEMENTATIONS_AVAILABLE",
    "HUMAN_OVERSIGHT_AVAILABLE",
    "WEB_DASHBOARD_AVAILABLE",
    "ROBUSTNESS_TESTING_AVAILABLE",
] + (["HumanOversightEngine",
    "OversightAlert", 
    "OversightReview",
    "AlertType",
    "ReviewStatus"] if HUMAN_OVERSIGHT_AVAILABLE else []) + (
    ["CIAFDashboard",
    "DashboardData",
    "create_dashboard"] if WEB_DASHBOARD_AVAILABLE else []) + (
    ["RobustnessTestSuite",
    "TestResult",
    "RobustnessReport",
    "TestType",
    "TestSeverity",
    "AdversarialTester",
    "DistributionShiftTester",
    "StressTester"] if ROBUSTNESS_TESTING_AVAILABLE else [])
