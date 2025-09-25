"""
Regulatory Framework Mapping for CIAF Compliance

This module maps CIAF capabilities to various regulatory frameworks and provides
compliance requirement checklists for different jurisdictions and industries.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""

    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    mandatory: bool
    ciaf_capabilities: List[str]
    implementation_notes: str
    verification_method: str
    documentation_required: List[str]
    risk_level: str = "medium"

    def is_satisfied_by_ciaf(self) -> bool:
        """Check if this requirement can be satisfied by CIAF capabilities."""
        return len(self.ciaf_capabilities) > 0


class RegulatoryMapper:
    """Maps CIAF capabilities to regulatory requirements."""

    def __init__(self):
        """Initialize with predefined regulatory mappings."""
        self.requirements: Dict[ComplianceFramework, List[ComplianceRequirement]] = {}
        self._initialize_requirements()

    def _initialize_requirements(self):
        """Initialize all regulatory framework requirements."""
        self._initialize_eu_ai_act()
        self._initialize_nist_ai_rmf()
        self._initialize_gdpr()
        self._initialize_hipaa()
        self._initialize_sox()
        self._initialize_iso_27001()
        self._initialize_pci_dss()
        self._initialize_ccpa()
        self._initialize_fda_ai_ml()
        self._initialize_fair_lending()
        self._initialize_general()

    def _initialize_eu_ai_act(self):
        """Initialize EU AI Act requirements."""
        self.requirements[ComplianceFramework.EU_AI_ACT] = [
            ComplianceRequirement(
                requirement_id="EU_AI_ACT_001",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Risk Management System",
                description="Establish and maintain a risk management system for high-risk AI systems",
                category="Risk Management",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "risk_assessment",
                    "provenance_tracking",
                ],
                implementation_notes="CIAF provides comprehensive audit trails and risk assessment capabilities",
                verification_method="Document risk management processes and audit trail integrity",
                documentation_required=[
                    "risk_assessment_report",
                    "audit_trail_documentation",
                    "provenance_records",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="EU_AI_ACT_002",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Data Governance and Management",
                description="Implement appropriate data governance and management practices",
                category="Data Governance",
                mandatory=True,
                ciaf_capabilities=[
                    "dataset_anchoring",
                    "provenance_capsules",
                    "cryptographic_integrity",
                ],
                implementation_notes="CIAF dataset anchoring ensures data integrity and provenance",
                verification_method="Verify dataset fingerprints and provenance capsule integrity",
                documentation_required=[
                    "dataset_documentation",
                    "data_quality_reports",
                    "provenance_validation",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="EU_AI_ACT_003",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Transparency and Provision of Information",
                description="Ensure transparency and provide adequate information to users",
                category="Transparency",
                mandatory=True,
                ciaf_capabilities=[
                    "inference_receipts",
                    "training_snapshots",
                    "transparency_reports",
                ],
                implementation_notes="CIAF inference receipts provide verifiable transparency",
                verification_method="Generate and validate transparency reports",
                documentation_required=[
                    "transparency_reports",
                    "user_documentation",
                    "inference_explanations",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="EU_AI_ACT_004",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Record Keeping",
                description="Maintain detailed records of AI system operation",
                category="Documentation",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "inference_receipts",
                    "training_snapshots",
                ],
                implementation_notes="CIAF automatically maintains comprehensive records",
                verification_method="Audit trail completeness and integrity verification",
                documentation_required=[
                    "operational_records",
                    "audit_logs",
                    "compliance_reports",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="EU_AI_ACT_005",
                framework=ComplianceFramework.EU_AI_ACT,
                title="Human Oversight",
                description="Ensure appropriate human oversight of AI systems",
                category="Human Oversight",
                mandatory=True,
                ciaf_capabilities=[
                    "inference_receipts",
                    "audit_trails",
                    "manual_review_flags",
                ],
                implementation_notes="CIAF supports human oversight through verifiable audit trails",
                verification_method="Document human oversight procedures and decision points",
                documentation_required=[
                    "oversight_procedures",
                    "human_review_logs",
                    "escalation_protocols",
                ],
                risk_level="high",
            ),
        ]

    def _initialize_nist_ai_rmf(self):
        """Initialize NIST AI Risk Management Framework requirements."""
        self.requirements[ComplianceFramework.NIST_AI_RMF] = [
            ComplianceRequirement(
                requirement_id="NIST_AI_RMF_001",
                framework=ComplianceFramework.NIST_AI_RMF,
                title="AI Risk Management Strategy",
                description="Develop and implement AI risk management strategy",
                category="Governance",
                mandatory=True,
                ciaf_capabilities=[
                    "risk_assessment",
                    "audit_trails",
                    "compliance_validation",
                ],
                implementation_notes="CIAF supports comprehensive risk management through audit trails",
                verification_method="Risk management documentation and validation reports",
                documentation_required=[
                    "risk_management_strategy",
                    "risk_assessment_reports",
                    "mitigation_strategies",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="NIST_AI_RMF_002",
                framework=ComplianceFramework.NIST_AI_RMF,
                title="AI System Inventory and Mapping",
                description="Maintain inventory and mapping of AI systems",
                category="Inventory",
                mandatory=True,
                ciaf_capabilities=[
                    "model_versioning",
                    "training_snapshots",
                    "audit_trails",
                ],
                implementation_notes="CIAF provides comprehensive model and data inventory",
                verification_method="System inventory completeness and accuracy verification",
                documentation_required=[
                    "ai_system_inventory",
                    "system_dependencies",
                    "version_control_records",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="NIST_AI_RMF_003",
                framework=ComplianceFramework.NIST_AI_RMF,
                title="Impact Assessment",
                description="Conduct impact assessments for AI systems",
                category="Assessment",
                mandatory=True,
                ciaf_capabilities=[
                    "risk_assessment",
                    "inference_receipts",
                    "audit_trails",
                ],
                implementation_notes="CIAF enables comprehensive impact assessment through provenance tracking",
                verification_method="Impact assessment documentation and validation",
                documentation_required=[
                    "impact_assessment_reports",
                    "stakeholder_analysis",
                    "risk_mitigation_plans",
                ],
                risk_level="high",
            ),
        ]

    def _initialize_gdpr(self):
        """Initialize GDPR requirements."""
        self.requirements[ComplianceFramework.GDPR] = [
            ComplianceRequirement(
                requirement_id="GDPR_001",
                framework=ComplianceFramework.GDPR,
                title="Data Processing Records",
                description="Maintain records of processing activities",
                category="Documentation",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "data_access_logging",
                    "provenance_tracking",
                ],
                implementation_notes="CIAF audit trails provide comprehensive data processing records",
                verification_method="Audit trail completeness for data processing activities",
                documentation_required=[
                    "processing_records",
                    "data_flow_documentation",
                    "purpose_limitation_evidence",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="GDPR_002",
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="Enable data subject rights (access, rectification, erasure)",
                category="Rights Management",
                mandatory=True,
                ciaf_capabilities=[
                    "provenance_tracking",
                    "audit_trails",
                    "data_lineage",
                ],
                implementation_notes="CIAF provenance enables data subject rights compliance",
                verification_method="Demonstrate data subject rights fulfillment capabilities",
                documentation_required=[
                    "rights_fulfillment_procedures",
                    "data_subject_request_logs",
                    "erasure_verification",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="GDPR_003",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design and Default",
                description="Implement data protection by design and default",
                category="Privacy Engineering",
                mandatory=True,
                ciaf_capabilities=[
                    "cryptographic_integrity",
                    "privacy_preserving_provenance",
                    "minimal_data_exposure",
                ],
                implementation_notes="CIAF implements privacy by design through cryptographic provenance",
                verification_method="Privacy engineering assessment and validation",
                documentation_required=[
                    "privacy_impact_assessment",
                    "design_documentation",
                    "privacy_controls_verification",
                ],
                risk_level="high",
            ),
        ]

    def _initialize_hipaa(self):
        """Initialize HIPAA requirements."""
        self.requirements[ComplianceFramework.HIPAA] = [
            ComplianceRequirement(
                requirement_id="HIPAA_001",
                framework=ComplianceFramework.HIPAA,
                title="Administrative Safeguards",
                description="Implement administrative safeguards for PHI",
                category="Administrative Controls",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "access_controls",
                    "user_authentication",
                ],
                implementation_notes="CIAF audit trails support HIPAA administrative safeguards",
                verification_method="Administrative controls documentation and audit",
                documentation_required=[
                    "administrative_procedures",
                    "access_control_policies",
                    "audit_procedures",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="HIPAA_002",
                framework=ComplianceFramework.HIPAA,
                title="Physical Safeguards",
                description="Implement physical safeguards for PHI",
                category="Physical Controls",
                mandatory=True,
                ciaf_capabilities=[
                    "cryptographic_protection",
                    "secure_storage",
                    "access_logging",
                ],
                implementation_notes="CIAF cryptographic controls support physical safeguards",
                verification_method="Physical controls assessment and validation",
                documentation_required=[
                    "physical_security_procedures",
                    "facility_access_controls",
                    "media_controls",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="HIPAA_003",
                framework=ComplianceFramework.HIPAA,
                title="Technical Safeguards",
                description="Implement technical safeguards for PHI",
                category="Technical Controls",
                mandatory=True,
                ciaf_capabilities=["encryption", "audit_trails", "integrity_controls"],
                implementation_notes="CIAF provides comprehensive technical safeguards",
                verification_method="Technical controls testing and validation",
                documentation_required=[
                    "technical_procedures",
                    "encryption_documentation",
                    "integrity_verification",
                ],
                risk_level="high",
            ),
        ]

    def _initialize_sox(self):
        """Initialize Sarbanes-Oxley requirements."""
        self.requirements[ComplianceFramework.SOX] = [
            ComplianceRequirement(
                requirement_id="SOX_001",
                framework=ComplianceFramework.SOX,
                title="Internal Controls Over Financial Reporting",
                description="Maintain internal controls over financial reporting",
                category="Internal Controls",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "integrity_controls",
                    "change_management",
                ],
                implementation_notes="CIAF audit trails support SOX internal controls",
                verification_method="Internal controls testing and documentation",
                documentation_required=[
                    "internal_control_documentation",
                    "testing_procedures",
                    "deficiency_reports",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="SOX_002",
                framework=ComplianceFramework.SOX,
                title="Documentation and Retention",
                description="Maintain documentation and records retention",
                category="Documentation",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "immutable_records",
                    "long_term_storage",
                ],
                implementation_notes="CIAF provides immutable audit trails for SOX compliance",
                verification_method="Documentation completeness and retention verification",
                documentation_required=[
                    "records_retention_policy",
                    "documentation_procedures",
                    "audit_trail_reports",
                ],
                risk_level="medium",
            ),
        ]

    def _initialize_iso_27001(self):
        """Initialize ISO 27001 Information Security Management requirements."""
        self.requirements[ComplianceFramework.ISO_27001] = [
            ComplianceRequirement(
                requirement_id="ISO_27001_001",
                framework=ComplianceFramework.ISO_27001,
                title="Information Security Management System (ISMS)",
                description="Establish, implement, maintain and continually improve an ISMS",
                category="Management System",
                mandatory=True,
                ciaf_capabilities=[
                    "audit_trails",
                    "risk_assessment",
                    "security_controls",
                ],
                implementation_notes="CIAF provides comprehensive audit trails supporting ISMS requirements",
                verification_method="ISMS documentation and effectiveness assessment",
                documentation_required=[
                    "isms_documentation",
                    "security_policy",
                    "risk_assessment_report",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="ISO_27001_002",
                framework=ComplianceFramework.ISO_27001,
                title="Risk Assessment and Treatment",
                description="Conduct information security risk assessments and implement appropriate treatments",
                category="Risk Management",
                mandatory=True,
                ciaf_capabilities=[
                    "risk_assessment",
                    "security_monitoring",
                    "audit_trails",
                ],
                implementation_notes="CIAF risk assessment capabilities align with ISO 27001 requirements",
                verification_method="Risk assessment process validation and documentation",
                documentation_required=[
                    "risk_assessment_methodology",
                    "risk_treatment_plan",
                    "risk_register",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="ISO_27001_003",
                framework=ComplianceFramework.ISO_27001,
                title="Access Control",
                description="Manage access to information and information processing facilities",
                category="Access Control",
                mandatory=True,
                ciaf_capabilities=[
                    "access_control",
                    "audit_trails",
                    "identity_management",
                ],
                implementation_notes="CIAF audit trails provide access control monitoring",
                verification_method="Access control effectiveness testing",
                documentation_required=[
                    "access_control_policy",
                    "user_access_matrix",
                    "access_logs",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="ISO_27001_004",
                framework=ComplianceFramework.ISO_27001,
                title="Cryptography",
                description="Implement appropriate cryptographic controls",
                category="Cryptography",
                mandatory=True,
                ciaf_capabilities=[
                    "cryptographic_integrity",
                    "key_management",
                    "data_encryption",
                ],
                implementation_notes="CIAF uses cryptographic techniques for data integrity and security",
                verification_method="Cryptographic implementation validation",
                documentation_required=[
                    "cryptographic_policy",
                    "key_management_procedures",
                    "encryption_standards",
                ],
                risk_level="high",
            ),
            ComplianceRequirement(
                requirement_id="ISO_27001_005",
                framework=ComplianceFramework.ISO_27001,
                title="Monitoring and Measurement",
                description="Monitor, measure, analyze and evaluate information security performance",
                category="Monitoring",
                mandatory=True,
                ciaf_capabilities=[
                    "continuous_monitoring",
                    "security_metrics",
                    "audit_trails",
                ],
                implementation_notes="CIAF provides continuous monitoring and security metrics",
                verification_method="Monitoring process effectiveness assessment",
                documentation_required=[
                    "monitoring_procedures",
                    "security_metrics_report",
                    "performance_indicators",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="ISO_27001_006",
                framework=ComplianceFramework.ISO_27001,
                title="Incident Management",
                description="Manage information security incidents effectively",
                category="Incident Management",
                mandatory=True,
                ciaf_capabilities=[
                    "incident_detection",
                    "audit_trails",
                    "forensic_analysis",
                ],
                implementation_notes="CIAF audit trails support incident detection and forensic analysis",
                verification_method="Incident management process validation",
                documentation_required=[
                    "incident_response_plan",
                    "incident_logs",
                    "lessons_learned_report",
                ],
                risk_level="medium",
            ),
        ]

    def _initialize_pci_dss(self):
        """Initialize PCI DSS Payment Card Industry requirements."""
        self.requirements[ComplianceFramework.PCI_DSS] = [
            ComplianceRequirement(
                requirement_id="PCI_DSS_001",
                framework=ComplianceFramework.PCI_DSS,
                title="Build and Maintain Secure Network",
                description="Install and maintain a firewall configuration to protect cardholder data",
                category="Network Security",
                mandatory=True,
                ciaf_capabilities=[
                    "network_monitoring",
                    "security_controls",
                    "audit_trails",
                ],
                implementation_notes="CIAF security monitoring supports PCI DSS network requirements",
                verification_method="Network security configuration validation",
                documentation_required=[
                    "firewall_configuration",
                    "network_diagram",
                    "security_standards",
                ],
                risk_level="high",
            )
        ]

    def _initialize_ccpa(self):
        """Initialize California Consumer Privacy Act requirements."""
        self.requirements[ComplianceFramework.CCPA] = [
            ComplianceRequirement(
                requirement_id="CCPA_001",
                framework=ComplianceFramework.CCPA,
                title="Consumer Rights",
                description="Provide consumers with rights regarding their personal information",
                category="Consumer Rights",
                mandatory=True,
                ciaf_capabilities=[
                    "data_governance",
                    "audit_trails",
                    "privacy_controls",
                ],
                implementation_notes="CIAF data governance supports CCPA consumer rights",
                verification_method="Consumer rights implementation validation",
                documentation_required=[
                    "privacy_policy",
                    "consumer_request_procedures",
                    "data_inventory",
                ],
                risk_level="medium",
            )
        ]

    def _initialize_fda_ai_ml(self):
        """Initialize FDA AI/ML Device requirements."""
        self.requirements[ComplianceFramework.FDA_AI_ML] = [
            ComplianceRequirement(
                requirement_id="FDA_AI_ML_001",
                framework=ComplianceFramework.FDA_AI_ML,
                title="Software as Medical Device (SaMD)",
                description="Implement quality management for AI/ML-based medical devices",
                category="Medical Device",
                mandatory=True,
                ciaf_capabilities=[
                    "model_validation",
                    "audit_trails",
                    "quality_management",
                ],
                implementation_notes="CIAF validation processes support FDA AI/ML requirements",
                verification_method="Medical device validation documentation",
                documentation_required=[
                    "software_validation_plan",
                    "clinical_evaluation",
                    "risk_analysis",
                ],
                risk_level="high",
            )
        ]

    def _initialize_fair_lending(self):
        """Initialize Fair Lending Act requirements."""
        self.requirements[ComplianceFramework.FAIR_LENDING] = [
            ComplianceRequirement(
                requirement_id="FAIR_LENDING_001",
                framework=ComplianceFramework.FAIR_LENDING,
                title="Discriminatory Practices Prevention",
                description="Prevent discriminatory lending practices",
                category="Fair Lending",
                mandatory=True,
                ciaf_capabilities=[
                    "bias_detection",
                    "fairness_assessment",
                    "audit_trails",
                ],
                implementation_notes="CIAF bias detection supports fair lending compliance",
                verification_method="Bias assessment and fairness testing",
                documentation_required=[
                    "fair_lending_policy",
                    "bias_testing_report",
                    "monitoring_procedures",
                ],
                risk_level="high",
            )
        ]

    def _initialize_general(self):
        """Initialize general best practices requirements."""
        self.requirements[ComplianceFramework.GENERAL] = [
            ComplianceRequirement(
                requirement_id="GENERAL_001",
                framework=ComplianceFramework.GENERAL,
                title="Model Governance",
                description="Implement comprehensive model governance",
                category="Governance",
                mandatory=False,
                ciaf_capabilities=[
                    "model_versioning",
                    "training_snapshots",
                    "audit_trails",
                ],
                implementation_notes="CIAF provides comprehensive model governance capabilities",
                verification_method="Model governance process documentation and validation",
                documentation_required=[
                    "model_governance_policy",
                    "version_control_procedures",
                    "change_management_logs",
                ],
                risk_level="medium",
            ),
            ComplianceRequirement(
                requirement_id="GENERAL_002",
                framework=ComplianceFramework.GENERAL,
                title="Explainability and Transparency",
                description="Provide model explainability and transparency",
                category="Transparency",
                mandatory=False,
                ciaf_capabilities=[
                    "inference_receipts",
                    "provenance_tracking",
                    "transparency_reports",
                ],
                implementation_notes="CIAF enables model transparency through provenance tracking",
                verification_method="Transparency documentation and validation",
                documentation_required=[
                    "explainability_reports",
                    "transparency_documentation",
                    "user_guides",
                ],
                risk_level="low",
            ),
        ]

    def get_requirements(
        self,
        frameworks: List[ComplianceFramework],
        mandatory_only: bool = False,
        category: Optional[str] = None,
    ) -> List[ComplianceRequirement]:
        """Get requirements for specified frameworks."""

        requirements = []
        for framework in frameworks:
            if framework in self.requirements:
                framework_reqs = self.requirements[framework]

                if mandatory_only:
                    framework_reqs = [r for r in framework_reqs if r.mandatory]

                if category:
                    framework_reqs = [
                        r for r in framework_reqs if r.category == category
                    ]

                requirements.extend(framework_reqs)

        return requirements

    def get_ciaf_coverage(
        self, frameworks: List[ComplianceFramework]
    ) -> Dict[str, Any]:
        """Analyze CIAF coverage for specified frameworks."""

        requirements = self.get_requirements(frameworks)

        total_requirements = len(requirements)
        satisfied_requirements = len(
            [r for r in requirements if r.is_satisfied_by_ciaf()]
        )
        mandatory_requirements = len([r for r in requirements if r.mandatory])
        satisfied_mandatory = len(
            [r for r in requirements if r.mandatory and r.is_satisfied_by_ciaf()]
        )

        coverage_by_framework = {}
        for framework in frameworks:
            framework_reqs = self.get_requirements([framework])
            framework_satisfied = len(
                [r for r in framework_reqs if r.is_satisfied_by_ciaf()]
            )
            coverage_by_framework[framework.value] = {
                "total_requirements": len(framework_reqs),
                "satisfied_requirements": framework_satisfied,
                "coverage_percentage": (
                    (framework_satisfied / len(framework_reqs)) * 100
                    if framework_reqs
                    else 0
                ),
            }

        return {
            "overall_coverage": {
                "total_requirements": total_requirements,
                "satisfied_requirements": satisfied_requirements,
                "coverage_percentage": (
                    (satisfied_requirements / total_requirements) * 100
                    if total_requirements
                    else 0
                ),
                "mandatory_coverage": (
                    (satisfied_mandatory / mandatory_requirements) * 100
                    if mandatory_requirements
                    else 0
                ),
            },
            "framework_coverage": coverage_by_framework,
            "unsatisfied_requirements": [
                {
                    "requirement_id": r.requirement_id,
                    "framework": r.framework.value,
                    "title": r.title,
                    "mandatory": r.mandatory,
                }
                for r in requirements
                if not r.is_satisfied_by_ciaf()
            ],
        }

    def generate_compliance_checklist(
        self, frameworks: List[ComplianceFramework], output_format: str = "json"
    ) -> str:
        """Generate compliance checklist for specified frameworks."""

        requirements = self.get_requirements(frameworks)

        checklist = {
            "frameworks": [f.value for f in frameworks],
            "total_requirements": len(requirements),
            "mandatory_requirements": len([r for r in requirements if r.mandatory]),
            "requirements": [],
        }

        for req in requirements:
            checklist["requirements"].append(
                {
                    "requirement_id": req.requirement_id,
                    "framework": req.framework.value,
                    "title": req.title,
                    "description": req.description,
                    "category": req.category,
                    "mandatory": req.mandatory,
                    "risk_level": req.risk_level,
                    "ciaf_satisfied": req.is_satisfied_by_ciaf(),
                    "ciaf_capabilities": req.ciaf_capabilities,
                    "implementation_notes": req.implementation_notes,
                    "verification_method": req.verification_method,
                    "documentation_required": req.documentation_required,
                }
            )

        if output_format.lower() == "json":
            return json.dumps(checklist, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def get_implementation_guidance(
        self, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Get implementation guidance for a specific framework."""

        if framework not in self.requirements:
            raise ValueError(f"Framework {framework.value} not supported")

        requirements = self.requirements[framework]

        guidance = {
            "framework": framework.value,
            "overview": self._get_framework_overview(framework),
            "implementation_steps": self._get_implementation_steps(framework),
            "ciaf_capabilities_needed": list(
                set([cap for req in requirements for cap in req.ciaf_capabilities])
            ),
            "documentation_templates": self._get_documentation_templates(framework),
            "validation_procedures": self._get_validation_procedures(framework),
        }

        return guidance

    def _get_framework_overview(self, framework: ComplianceFramework) -> str:
        """Get overview for a specific framework."""
        overviews = {
            ComplianceFramework.EU_AI_ACT: "The EU AI Act is comprehensive legislation regulating AI systems based on risk levels.",
            ComplianceFramework.NIST_AI_RMF: "NIST AI RMF provides a framework for managing AI risks throughout the AI lifecycle.",
            ComplianceFramework.GDPR: "GDPR regulates data protection and privacy for individuals within the EU.",
            ComplianceFramework.HIPAA: "HIPAA establishes national standards for protecting health information.",
            ComplianceFramework.SOX: "SOX establishes requirements for financial reporting and internal controls.",
        }
        return overviews.get(framework, "Compliance framework overview not available.")

    def _get_implementation_steps(self, framework: ComplianceFramework) -> List[str]:
        """Get implementation steps for a specific framework."""
        # This would be expanded with detailed implementation steps
        return [
            "1. Assess current compliance posture",
            "2. Identify gaps and requirements",
            "3. Implement CIAF capabilities",
            "4. Document compliance procedures",
            "5. Validate compliance implementation",
            "6. Maintain ongoing compliance monitoring",
        ]

    def _get_documentation_templates(self, framework: ComplianceFramework) -> List[str]:
        """Get documentation templates for a specific framework."""
        return [
            "compliance_policy_template",
            "risk_assessment_template",
            "audit_procedures_template",
            "incident_response_template",
        ]

    def _get_validation_procedures(self, framework: ComplianceFramework) -> List[str]:
        """Get validation procedures for a specific framework."""
        return [
            "compliance_audit_procedure",
            "risk_assessment_validation",
            "documentation_review_process",
            "ongoing_monitoring_procedures",
        ]
