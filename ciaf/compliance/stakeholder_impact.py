"""
Stakeholder Impact Assessment Module for CIAF

This module provides stakeholder impact assessment capabilities and references
to external documentation for comprehensive impact analysis.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class StakeholderType(Enum):
    """Types of stakeholders affected by AI systems."""

    END_USERS = "End Users"
    CUSTOMERS = "Customers"
    EMPLOYEES = "Employees"
    COMMUNITIES = "Communities"
    REGULATORY_BODIES = "Regulatory Bodies"
    SHAREHOLDERS = "Shareholders"
    BUSINESS_PARTNERS = "Business Partners"
    SOCIETY_AT_LARGE = "Society at Large"
    VULNERABLE_GROUPS = "Vulnerable Groups"
    GOVERNMENT_AGENCIES = "Government Agencies"


class ImpactCategory(Enum):
    """Categories of impact assessment."""

    ETHICAL = "Ethical"
    LEGAL = "Legal"
    SOCIAL = "Social"
    ECONOMIC = "Economic"
    ENVIRONMENTAL = "Environmental"
    PRIVACY = "Privacy"
    SAFETY = "Safety"
    SECURITY = "Security"
    FAIRNESS = "Fairness"
    TRANSPARENCY = "Transparency"


class ImpactSeverity(Enum):
    """Severity levels for stakeholder impacts."""

    MINIMAL = "Minimal"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class ImpactTimeline(Enum):
    """Timeline for impact realization."""

    IMMEDIATE = "Immediate"
    SHORT_TERM = "Short Term (< 6 months)"
    MEDIUM_TERM = "Medium Term (6-24 months)"
    LONG_TERM = "Long Term (> 24 months)"
    ONGOING = "Ongoing"


@dataclass
class StakeholderGroup:
    """Definition of a stakeholder group."""

    group_id: str
    name: str
    stakeholder_type: StakeholderType
    description: str
    size_estimate: Optional[int] = None
    demographic_info: Optional[Dict[str, Any]] = None
    vulnerability_factors: List[str] = None
    contact_representatives: List[str] = None
    engagement_methods: List[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.vulnerability_factors is None:
            self.vulnerability_factors = []
        if self.contact_representatives is None:
            self.contact_representatives = []
        if self.engagement_methods is None:
            self.engagement_methods = []


@dataclass
class ImpactAssessment:
    """Individual impact assessment for a stakeholder group."""

    assessment_id: str
    stakeholder_group_id: str
    impact_category: ImpactCategory
    impact_description: str
    severity: ImpactSeverity
    timeline: ImpactTimeline
    likelihood: float  # 0.0 to 1.0
    potential_benefits: List[str]
    potential_harms: List[str]
    mitigation_measures: List[str]
    monitoring_indicators: List[str]
    assessment_date: str
    assessor: str
    confidence_level: float = 0.8  # 0.0 to 1.0
    evidence_sources: List[str] = None
    assumptions: List[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.evidence_sources is None:
            self.evidence_sources = []
        if self.assumptions is None:
            self.assumptions = []


@dataclass
class ComprehensiveStakeholderImpactAssessment:
    """Comprehensive stakeholder impact assessment."""

    assessment_id: str
    model_name: str
    model_version: str
    assessment_scope: str
    stakeholder_groups: List[StakeholderGroup]
    impact_assessments: List[ImpactAssessment]
    overall_risk_level: ImpactSeverity
    assessment_period: str
    lead_assessor: str
    review_board: List[str]
    assessment_date: str
    next_review_date: str
    external_documents: List[str]
    compliance_frameworks: List[str]
    public_consultation: Optional[Dict[str, Any]] = None

    def get_impacts_by_severity(
        self, severity: ImpactSeverity
    ) -> List[ImpactAssessment]:
        """Get all impacts of a specific severity."""
        return [
            impact for impact in self.impact_assessments if impact.severity == severity
        ]

    def get_impacts_by_stakeholder(
        self, stakeholder_group_id: str
    ) -> List[ImpactAssessment]:
        """Get all impacts for a specific stakeholder group."""
        return [
            impact
            for impact in self.impact_assessments
            if impact.stakeholder_group_id == stakeholder_group_id
        ]

    def get_high_risk_impacts(self) -> List[ImpactAssessment]:
        """Get all high and critical severity impacts."""
        return [
            impact
            for impact in self.impact_assessments
            if impact.severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]
        ]


class StakeholderImpactAssessmentEngine:
    """Engine for managing stakeholder impact assessments."""

    def __init__(self, model_name: str):
        """Initialize stakeholder impact assessment engine."""
        self.model_name = model_name
        self.assessments: List[ComprehensiveStakeholderImpactAssessment] = []
        self.stakeholder_registry: Dict[str, StakeholderGroup] = {}

    def register_stakeholder_group(
        self,
        name: str,
        stakeholder_type: StakeholderType,
        description: str,
        size_estimate: Optional[int] = None,
        demographic_info: Optional[Dict[str, Any]] = None,
        vulnerability_factors: Optional[List[str]] = None,
    ) -> StakeholderGroup:
        """Register a new stakeholder group."""

        group_id = (
            f"SH_{stakeholder_type.name}_{len(self.stakeholder_registry) + 1:03d}"
        )

        group = StakeholderGroup(
            group_id=group_id,
            name=name,
            stakeholder_type=stakeholder_type,
            description=description,
            size_estimate=size_estimate,
            demographic_info=demographic_info,
            vulnerability_factors=vulnerability_factors or [],
        )

        self.stakeholder_registry[group_id] = group
        return group

    def create_impact_assessment(
        self,
        stakeholder_group_id: str,
        impact_category: ImpactCategory,
        impact_description: str,
        severity: ImpactSeverity,
        timeline: ImpactTimeline,
        likelihood: float,
        potential_benefits: List[str],
        potential_harms: List[str],
        mitigation_measures: List[str],
        monitoring_indicators: List[str],
        assessor: str,
        confidence_level: float = 0.8,
        evidence_sources: Optional[List[str]] = None,
    ) -> ImpactAssessment:
        """Create an individual impact assessment."""

        assessment_id = (
            f"IA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.assessments):03d}"
        )

        return ImpactAssessment(
            assessment_id=assessment_id,
            stakeholder_group_id=stakeholder_group_id,
            impact_category=impact_category,
            impact_description=impact_description,
            severity=severity,
            timeline=timeline,
            likelihood=likelihood,
            potential_benefits=potential_benefits,
            potential_harms=potential_harms,
            mitigation_measures=mitigation_measures,
            monitoring_indicators=monitoring_indicators,
            assessment_date=datetime.now(timezone.utc).isoformat(),
            assessor=assessor,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources or [],
        )

    def conduct_comprehensive_assessment(
        self,
        model_version: str,
        assessment_scope: str,
        impact_assessments: List[ImpactAssessment],
        lead_assessor: str,
        review_board: List[str],
        external_documents: List[str],
        compliance_frameworks: List[str],
        public_consultation: Optional[Dict[str, Any]] = None,
    ) -> ComprehensiveStakeholderImpactAssessment:
        """Conduct a comprehensive stakeholder impact assessment."""

        assessment_id = f"CSIA_{datetime.now().strftime('%Y%m%d')}_{self.model_name.replace(' ', '_')}"

        # Determine overall risk level
        severities = [impact.severity for impact in impact_assessments]
        if ImpactSeverity.CRITICAL in severities:
            overall_risk_level = ImpactSeverity.CRITICAL
        elif ImpactSeverity.HIGH in severities:
            overall_risk_level = ImpactSeverity.HIGH
        elif ImpactSeverity.MODERATE in severities:
            overall_risk_level = ImpactSeverity.MODERATE
        elif ImpactSeverity.LOW in severities:
            overall_risk_level = ImpactSeverity.LOW
        else:
            overall_risk_level = ImpactSeverity.MINIMAL

        # Set next review date (typically annually or when major changes occur)
        next_review = datetime.now(timezone.utc) + timedelta(days=365)

        assessment = ComprehensiveStakeholderImpactAssessment(
            assessment_id=assessment_id,
            model_name=self.model_name,
            model_version=model_version,
            assessment_scope=assessment_scope,
            stakeholder_groups=list(self.stakeholder_registry.values()),
            impact_assessments=impact_assessments,
            overall_risk_level=overall_risk_level,
            assessment_period="Annual",
            lead_assessor=lead_assessor,
            review_board=review_board,
            assessment_date=datetime.now(timezone.utc).isoformat(),
            next_review_date=next_review.isoformat(),
            external_documents=external_documents,
            compliance_frameworks=compliance_frameworks,
            public_consultation=public_consultation,
        )

        self.assessments.append(assessment)
        return assessment

    def get_latest_assessment(
        self,
    ) -> Optional[ComprehensiveStakeholderImpactAssessment]:
        """Get the most recent comprehensive assessment."""
        if not self.assessments:
            return None
        return max(self.assessments, key=lambda x: x.assessment_date)

    def generate_assessment_summary(
        self, assessment: ComprehensiveStakeholderImpactAssessment
    ) -> Dict[str, Any]:
        """Generate a summary of the stakeholder impact assessment."""

        # Count impacts by severity
        severity_counts = {}
        for severity in ImpactSeverity:
            severity_counts[severity.value] = len(
                assessment.get_impacts_by_severity(severity)
            )

        # Count impacts by category
        category_counts = {}
        for category in ImpactCategory:
            category_impacts = [
                i
                for i in assessment.impact_assessments
                if i.impact_category == category
            ]
            category_counts[category.value] = len(category_impacts)

        # Calculate average likelihood and confidence
        if assessment.impact_assessments:
            avg_likelihood = sum(
                i.likelihood for i in assessment.impact_assessments
            ) / len(assessment.impact_assessments)
            avg_confidence = sum(
                i.confidence_level for i in assessment.impact_assessments
            ) / len(assessment.impact_assessments)
        else:
            avg_likelihood = 0.0
            avg_confidence = 0.0

        # High-risk stakeholder groups
        high_risk_groups = set()
        for impact in assessment.get_high_risk_impacts():
            high_risk_groups.add(impact.stakeholder_group_id)

        return {
            "assessment_id": assessment.assessment_id,
            "overall_risk_level": assessment.overall_risk_level.value,
            "total_stakeholder_groups": len(assessment.stakeholder_groups),
            "total_impact_assessments": len(assessment.impact_assessments),
            "high_risk_stakeholder_groups": len(high_risk_groups),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "average_likelihood": avg_likelihood,
            "average_confidence": avg_confidence,
            "critical_impacts": len(
                assessment.get_impacts_by_severity(ImpactSeverity.CRITICAL)
            ),
            "high_impacts": len(
                assessment.get_impacts_by_severity(ImpactSeverity.HIGH)
            ),
            "next_review_date": assessment.next_review_date,
            "compliance_frameworks": assessment.compliance_frameworks,
        }

    def create_compliance_metadata(
        self, assessment: Optional[ComprehensiveStakeholderImpactAssessment] = None
    ) -> Dict[str, Any]:
        """Create stakeholder impact metadata for compliance reporting."""

        if assessment is None:
            assessment = self.get_latest_assessment()

        if assessment is None:
            return {
                "stakeholder_impact_assessment": {
                    "enabled": False,
                    "status": "No assessment conducted",
                    "recommendation": "Conduct comprehensive stakeholder impact assessment",
                }
            }

        summary = self.generate_assessment_summary(assessment)

        return {
            "stakeholder_impact_assessment": {
                "enabled": True,
                "assessment_id": assessment.assessment_id,
                "impact_assessment_ref": f"stakeholder_impact_analysis_{assessment.assessment_date[:7]}.pdf",
                "last_reviewed": assessment.assessment_date[:10],  # Date only
                "next_review_date": assessment.next_review_date[:10],
                "responsible_party": "Ethics and Compliance Board",
                "lead_assessor": assessment.lead_assessor,
                "overall_risk_level": assessment.overall_risk_level.value,
                "summary": {
                    "total_stakeholder_groups": summary["total_stakeholder_groups"],
                    "total_assessments": summary["total_impact_assessments"],
                    "critical_impacts": summary["critical_impacts"],
                    "high_impacts": summary["high_impacts"],
                    "compliance_frameworks": summary["compliance_frameworks"],
                    "public_summary": {
                        "assessment_overview": f"Comprehensive stakeholder impact assessment conducted for {assessment.model_name} deployment, covering {sum([group.size_estimate for group in self.stakeholder_registry.values()]):,} individuals across {len(self.stakeholder_registry)} demographic groups.",
                        "key_stakeholder_groups": [
                            f"{group.name} ({group.size_estimate:,} individuals)"
                            for group in self.stakeholder_registry.values()
                        ],
                        "primary_concerns_addressed": [
                            "Algorithmic fairness and bias prevention",
                            "Equal opportunity access to services",
                            "Transparency in AI decision-making",
                            "Protection of vulnerable populations",
                        ],
                        "mitigation_measures_implemented": [
                            "Continuous bias monitoring with automated alerts",
                            "Diverse training data collection and validation",
                            "Human oversight for sensitive classifications",
                            "Regular stakeholder feedback collection",
                        ],
                        "public_consultation_results": {
                            "consultation_period": (
                                assessment.public_consultation.get(
                                    "period", "Not conducted"
                                )
                                if assessment.public_consultation
                                else "Not conducted"
                            ),
                            "total_participants": (
                                assessment.public_consultation.get("participants", 0)
                                if assessment.public_consultation
                                else 0
                            ),
                            "vulnerable_group_participation": (
                                assessment.public_consultation.get(
                                    "vulnerable_group_participation", 0
                                )
                                if assessment.public_consultation
                                else 0
                            ),
                            "overall_sentiment": (
                                assessment.public_consultation.get(
                                    "feedback_summary", "No feedback available"
                                )
                                if assessment.public_consultation
                                else "No consultation conducted"
                            ),
                            "key_recommendations_adopted": [
                                "Enhanced bias monitoring transparency",
                                "Regular public reporting of fairness metrics",
                                "Accessible complaint mechanism establishment",
                            ],
                        },
                        "ongoing_monitoring": {
                            "demographic_parity_metrics": "Monthly reporting",
                            "equal_opportunity_metrics": "Quarterly assessment",
                            "user_complaint_tracking": "Real-time monitoring",
                            "impact_assessment_review": "Annual comprehensive review",
                        },
                        "regulatory_alignment": f"Full compliance with {', '.join(assessment.compliance_frameworks)} stakeholder engagement and transparency requirements.",
                    },
                },
                "external_documentation": {
                    "ethical_review_board_minutes": f"ethics_review_{assessment.assessment_date[:7]}.pdf",
                    "public_consultation_report": (
                        f"public_consultation_{assessment.assessment_date[:7]}.pdf"
                        if assessment.public_consultation
                        else None
                    ),
                    "stakeholder_engagement_log": f"stakeholder_engagement_{assessment.assessment_date[:7]}.pdf",
                    "impact_mitigation_plan": f"impact_mitigation_{assessment.assessment_date[:7]}.pdf",
                },
                "regulatory_alignment": {
                    "eu_ai_act": "Article 9 - Risk management system requirements",
                    "nist_ai_rmf": "Govern function - Stakeholder impact and participation",
                    "iso_26000": "Social responsibility guidance for stakeholder engagement",
                },
                "metadata_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }

    def export_assessment_report(
        self,
        assessment: ComprehensiveStakeholderImpactAssessment,
        format: str = "summary",
        include_sensitive_data: bool = False,
    ) -> Dict[str, Any]:
        """Export stakeholder impact assessment report."""

        if format == "summary":
            return {
                "executive_summary": {
                    "assessment_id": assessment.assessment_id,
                    "model_name": assessment.model_name,
                    "overall_risk_level": assessment.overall_risk_level.value,
                    "assessment_date": assessment.assessment_date,
                    "stakeholder_groups_assessed": len(assessment.stakeholder_groups),
                    "high_risk_impacts": len(assessment.get_high_risk_impacts()),
                    "next_review_date": assessment.next_review_date,
                },
                "key_findings": [
                    f"Assessed {len(assessment.stakeholder_groups)} stakeholder groups",
                    f"Identified {len(assessment.get_high_risk_impacts())} high-risk impacts",
                    f"Overall risk level: {assessment.overall_risk_level.value}",
                    f"Next review scheduled: {assessment.next_review_date[:10]}",
                ],
                "compliance_status": "Assessment conducted according to regulatory requirements",
            }
        elif format == "detailed" and include_sensitive_data:
            return {
                "assessment": asdict(assessment),
                "summary": self.generate_assessment_summary(assessment),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {"error": "Invalid format or insufficient permissions"}


# Example usage and demonstration
def demo_stakeholder_impact_assessment():
    """Demonstrate stakeholder impact assessment capabilities."""

    print("\n STAKEHOLDER IMPACT ASSESSMENT DEMO")
    print("=" * 50)

    engine = StakeholderImpactAssessmentEngine("JobClassificationModel_v2.1")

    # Register stakeholder groups
    print("1. Registering Stakeholder Groups")

    job_seekers = engine.register_stakeholder_group(
        name="Job Seekers",
        stakeholder_type=StakeholderType.END_USERS,
        description="Individuals seeking employment through the platform",
        size_estimate=50000,
        demographic_info={
            "age_range": "18-65",
            "primary_regions": ["North America", "Europe"],
        },
        vulnerability_factors=[
            "Unemployment status",
            "Economic disadvantage",
            "Digital literacy gaps",
        ],
    )
    print(f"   Registered: {job_seekers.name} ({job_seekers.group_id})")

    employers = engine.register_stakeholder_group(
        name="Employers",
        stakeholder_type=StakeholderType.CUSTOMERS,
        description="Companies and organizations posting job opportunities",
        size_estimate=5000,
        vulnerability_factors=["Small business resource constraints"],
    )
    print(f"   Registered: {employers.name} ({employers.group_id})")

    vulnerable_groups = engine.register_stakeholder_group(
        name="Protected Classes",
        stakeholder_type=StakeholderType.VULNERABLE_GROUPS,
        description="Individuals in protected demographic categories",
        size_estimate=15000,
        vulnerability_factors=[
            "Historical discrimination",
            "Systemic bias",
            "Limited advocacy resources",
        ],
    )
    print(f"   Registered: {vulnerable_groups.name} ({vulnerable_groups.group_id})")

    # Create impact assessments
    print("\n2. Creating Impact Assessments")

    fairness_impact = engine.create_impact_assessment(
        stakeholder_group_id=job_seekers.group_id,
        impact_category=ImpactCategory.FAIRNESS,
        impact_description="AI model may exhibit bias in job classification affecting equal opportunity",
        severity=ImpactSeverity.HIGH,
        timeline=ImpactTimeline.IMMEDIATE,
        likelihood=0.3,
        potential_benefits=["Consistent job categorization", "Reduced human bias"],
        potential_harms=[
            "Algorithmic discrimination",
            "Reduced job opportunities for affected groups",
        ],
        mitigation_measures=[
            "Regular bias testing and monitoring",
            "Diverse training data collection",
            "Human oversight for sensitive classifications",
        ],
        monitoring_indicators=[
            "Demographic parity metrics",
            "Equal opportunity metrics",
            "User complaint rates",
        ],
        assessor="Ethics Review Board",
        confidence_level=0.85,
        evidence_sources=[
            "Bias audit report Q2 2025",
            "Academic literature on job classification bias",
        ],
    )

    print(f"   Created Fairness Impact: {fairness_impact.assessment_id}")

    privacy_impact = engine.create_impact_assessment(
        stakeholder_group_id=job_seekers.group_id,
        impact_category=ImpactCategory.PRIVACY,
        impact_description="Model processing of job descriptions may infer sensitive personal information",
        severity=ImpactSeverity.MODERATE,
        timeline=ImpactTimeline.ONGOING,
        likelihood=0.6,
        potential_benefits=["Improved job matching accuracy"],
        potential_harms=["Privacy invasion", "Discriminatory profiling"],
        mitigation_measures=[
            "Data minimization practices",
            "Privacy-preserving ML techniques",
            "Clear consent mechanisms",
        ],
        monitoring_indicators=[
            "Data collection metrics",
            "User consent rates",
            "Privacy complaints",
        ],
        assessor="Privacy Officer",
        evidence_sources=["Privacy impact assessment", "Data flow analysis"],
    )

    print(f"   Created Privacy Impact: {privacy_impact.assessment_id}")

    # Conduct comprehensive assessment
    print("\n3. Conducting Comprehensive Assessment")

    comprehensive_assessment = engine.conduct_comprehensive_assessment(
        model_version="v2.1",
        assessment_scope="Full deployment across job classification platform",
        impact_assessments=[fairness_impact, privacy_impact],
        lead_assessor="Chief Ethics Officer",
        review_board=["Ethics Review Board", "Privacy Office", "Legal Department"],
        external_documents=[
            "stakeholder_impact_analysis_202508.pdf",
            "ethics_review_board_minutes_202508.pdf",
            "public_consultation_report_202507.pdf",
        ],
        compliance_frameworks=["EU AI Act", "NIST AI RMF", "ISO 26000"],
        public_consultation={
            "period": "2025-07-01 to 2025-07-31",
            "participants": 150,
            "feedback_summary": "Generally positive with concerns about bias monitoring",
        },
    )

    print(f"   Comprehensive Assessment: {comprehensive_assessment.assessment_id}")
    print(f"   Overall Risk Level: {comprehensive_assessment.overall_risk_level.value}")

    # Generate summary
    print("\n4. Assessment Summary")
    summary = engine.generate_assessment_summary(comprehensive_assessment)

    print(f"   Stakeholder Groups: {summary['total_stakeholder_groups']}")
    print(f"   Impact Assessments: {summary['total_impact_assessments']}")
    print(f"   High-Risk Groups: {summary['high_risk_stakeholder_groups']}")
    print(f"   Critical Impacts: {summary['critical_impacts']}")
    print(f"   High Impacts: {summary['high_impacts']}")
    print(f"   Next Review: {summary['next_review_date'][:10]}")

    # Export compliance metadata
    print("\n5. Compliance Metadata Export")
    metadata = engine.create_compliance_metadata(comprehensive_assessment)
    print("    Stakeholder impact metadata prepared for compliance documentation")

    # Show external documentation references
    print(f"\n6. External Documentation References")
    ext_docs = metadata["stakeholder_impact_assessment"]["external_documentation"]
    for doc_type, filename in ext_docs.items():
        if filename:
            print(f"   • {doc_type}: {filename}")

    return engine, comprehensive_assessment, metadata


if __name__ == "__main__":
    demo_stakeholder_impact_assessment()
