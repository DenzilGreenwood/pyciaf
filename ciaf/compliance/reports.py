"""
Compliance Report Generation for CIAF

This module generates comprehensive compliance reports for various regulatory
frameworks, including executive summaries, detailed findings, and recommendations.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .audit_trails import AuditTrailGenerator, ComplianceAuditRecord
from .regulatory_mapping import ComplianceFramework, RegulatoryMapper


class ReportType(Enum):
    """Types of compliance reports."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ASSESSMENT = "detailed_assessment"
    GAP_ANALYSIS = "gap_analysis"
    AUDIT_REPORT = "audit_report"
    RISK_ASSESSMENT = "risk_assessment"
    QUARTERLY_REVIEW = "quarterly_review"
    ANNUAL_REPORT = "annual_report"


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""

    # Report metadata
    report_id: str
    report_type: ReportType
    frameworks: List[str]
    model_name: str
    model_version: str
    generated_date: str
    reporting_period_start: str
    reporting_period_end: str
    generated_by: str

    # Executive summary
    executive_summary: Dict[str, Any]

    # Detailed findings
    compliance_status: Dict[str, Any]
    requirements_assessment: List[Dict[str, Any]]
    audit_findings: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]

    # Recommendations and actions
    recommendations: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]

    # Supporting data
    audit_statistics: Dict[str, Any]
    compliance_metrics: Dict[str, Any]
    appendices: Dict[str, Any]

    # Report integrity
    report_hash: str = ""

    def __post_init__(self):
        """Compute report hash for integrity."""
        report_data = {
            "report_id": self.report_id,
            "model_name": self.model_name,
            "generated_date": self.generated_date,
            "frameworks": self.frameworks,
            "compliance_status": self.compliance_status,
        }
        import hashlib

        report_str = json.dumps(report_data, sort_keys=True)
        self.report_hash = hashlib.sha256(report_str.encode()).hexdigest()


class ComplianceReportGenerator:
    """Generates comprehensive compliance reports."""

    def __init__(self, model_name: str):
        """Initialize report generator."""
        self.model_name = model_name
        self.regulatory_mapper = RegulatoryMapper()

    def generate_executive_summary_report(
        self,
        frameworks: List[ComplianceFramework],
        audit_generator: AuditTrailGenerator,
        model_version: str = "current",
        reporting_period_days: int = 90,
    ) -> ComplianceReport:
        """Generate executive summary compliance report."""

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=reporting_period_days)

        # Get compliance coverage
        coverage = self.regulatory_mapper.get_ciaf_coverage(frameworks)

        # Get audit statistics
        audit_records = audit_generator.get_audit_trail(start_date, end_date)
        audit_stats = self._analyze_audit_records(audit_records)

        # Generate executive summary
        executive_summary = {
            "overall_compliance_score": coverage["overall_coverage"][
                "coverage_percentage"
            ],
            "frameworks_assessed": [f.value for f in frameworks],
            "total_requirements": coverage["overall_coverage"]["total_requirements"],
            "satisfied_requirements": coverage["overall_coverage"][
                "satisfied_requirements"
            ],
            "audit_events_period": len(audit_records),
            "high_risk_events": audit_stats["high_risk_count"],
            "compliance_trend": "stable",  # Would be calculated from historical data
            "key_findings": self._generate_key_findings(coverage, audit_stats),
            "action_required": coverage["overall_coverage"]["coverage_percentage"] < 90,
        }

        # Generate compliance status
        compliance_status = {
            "overall_status": (
                "compliant"
                if coverage["overall_coverage"]["coverage_percentage"] >= 90
                else "non_compliant"
            ),
            "framework_status": {},
            "critical_gaps": [],
            "improvement_areas": [],
        }

        for framework_name, framework_coverage in coverage[
            "framework_coverage"
        ].items():
            compliance_status["framework_status"][framework_name] = {
                "status": (
                    "compliant"
                    if framework_coverage["coverage_percentage"] >= 90
                    else "non_compliant"
                ),
                "coverage_percentage": framework_coverage["coverage_percentage"],
                "satisfied_requirements": framework_coverage["satisfied_requirements"],
                "total_requirements": framework_coverage["total_requirements"],
            }

        # Add critical gaps
        for req in coverage["unsatisfied_requirements"]:
            if req["mandatory"]:
                compliance_status["critical_gaps"].append(req)

        report = ComplianceReport(
            report_id=f"exec_summary_{int(datetime.now().timestamp())}",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            frameworks=[f.value for f in frameworks],
            model_name=self.model_name,
            model_version=model_version,
            generated_date=datetime.now(timezone.utc).isoformat(),
            reporting_period_start=start_date.isoformat(),
            reporting_period_end=end_date.isoformat(),
            generated_by="CIAF_ComplianceReportGenerator",
            executive_summary=executive_summary,
            compliance_status=compliance_status,
            requirements_assessment=[],
            audit_findings=self._summarize_audit_findings(audit_records),
            risk_assessment=self._assess_compliance_risk(coverage, audit_stats),
            recommendations=self._generate_recommendations(coverage),
            action_items=self._generate_action_items(coverage),
            audit_statistics=audit_stats,
            compliance_metrics=self._calculate_compliance_metrics(
                coverage, audit_stats
            ),
            appendices={},
        )

        return report

    def generate_detailed_assessment_report(
        self,
        frameworks: List[ComplianceFramework],
        audit_generator: AuditTrailGenerator,
        model_version: str = "current",
    ) -> ComplianceReport:
        """Generate detailed compliance assessment report."""

        # Get all requirements for detailed assessment
        requirements_assessment = []
        for framework in frameworks:
            framework_requirements = self.regulatory_mapper.get_requirements(
                [framework]
            )

            for req in framework_requirements:
                assessment = {
                    "requirement_id": req.requirement_id,
                    "framework": req.framework.value,
                    "title": req.title,
                    "description": req.description,
                    "category": req.category,
                    "mandatory": req.mandatory,
                    "risk_level": req.risk_level,
                    "ciaf_satisfied": req.is_satisfied_by_ciaf(),
                    "ciaf_capabilities": req.ciaf_capabilities,
                    "implementation_status": (
                        "implemented" if req.is_satisfied_by_ciaf() else "gap"
                    ),
                    "evidence": self._gather_requirement_evidence(req, audit_generator),
                    "gaps": (
                        []
                        if req.is_satisfied_by_ciaf()
                        else ["Implementation required"]
                    ),
                    "recommendations": (
                        req.implementation_notes
                        if not req.is_satisfied_by_ciaf()
                        else "Maintain current implementation"
                    ),
                }
                requirements_assessment.append(assessment)

        # Generate base report structure (similar to executive summary)
        exec_report = self.generate_executive_summary_report(
            frameworks, audit_generator, model_version
        )

        # Update with detailed information
        exec_report.report_type = ReportType.DETAILED_ASSESSMENT
        exec_report.report_id = f"detailed_assessment_{int(datetime.now().timestamp())}"
        exec_report.requirements_assessment = requirements_assessment

        # Add detailed appendices
        exec_report.appendices = {
            "requirement_details": {
                req["requirement_id"]: req for req in requirements_assessment
            },
            "ciaf_capability_mapping": self._generate_capability_mapping(frameworks),
            "implementation_timeline": self._generate_implementation_timeline(
                requirements_assessment
            ),
            "cost_benefit_analysis": self._generate_cost_benefit_analysis(
                requirements_assessment
            ),
        }

        return exec_report

    def generate_gap_analysis_report(
        self,
        frameworks: List[ComplianceFramework],
        audit_generator: AuditTrailGenerator,
        model_version: str = "current",
    ) -> ComplianceReport:
        """Generate gap analysis report focusing on compliance gaps."""

        coverage = self.regulatory_mapper.get_ciaf_coverage(frameworks)

        # Focus on gaps
        gap_analysis = {
            "total_gaps": len(coverage["unsatisfied_requirements"]),
            "critical_gaps": len(
                [r for r in coverage["unsatisfied_requirements"] if r["mandatory"]]
            ),
            "gaps_by_framework": {},
            "gaps_by_category": {},
            "priority_gaps": [],
            "implementation_effort": {},
        }

        # Analyze gaps by framework
        for framework in frameworks:
            framework_reqs = self.regulatory_mapper.get_requirements([framework])
            framework_gaps = [r for r in framework_reqs if not r.is_satisfied_by_ciaf()]

            gap_analysis["gaps_by_framework"][framework.value] = {
                "total_gaps": len(framework_gaps),
                "critical_gaps": len([r for r in framework_gaps if r.mandatory]),
                "gap_details": [
                    {
                        "requirement_id": r.requirement_id,
                        "title": r.title,
                        "mandatory": r.mandatory,
                        "risk_level": r.risk_level,
                    }
                    for r in framework_gaps
                ],
            }

        # Create gap analysis report
        base_report = self.generate_executive_summary_report(
            frameworks, audit_generator, model_version
        )
        base_report.report_type = ReportType.GAP_ANALYSIS
        base_report.report_id = f"gap_analysis_{int(datetime.now().timestamp())}"

        # Update with gap-specific information
        base_report.executive_summary["gap_analysis"] = gap_analysis
        base_report.recommendations = self._generate_gap_remediation_plan(coverage)

        return base_report

    def generate_audit_report(
        self,
        audit_generator: AuditTrailGenerator,
        frameworks: List[ComplianceFramework],
        start_date: datetime,
        end_date: datetime,
        model_version: str = "current",
    ) -> ComplianceReport:
        """Generate audit-focused compliance report."""

        # Get audit records for period
        audit_records = audit_generator.get_audit_trail(start_date, end_date)
        audit_integrity = audit_generator.verify_audit_integrity()

        # Analyze audit records
        audit_analysis = {
            "audit_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_days": (end_date - start_date).days,
            },
            "audit_integrity": audit_integrity,
            "event_summary": self._analyze_audit_records(audit_records),
            "compliance_events": self._analyze_compliance_events(audit_records),
            "risk_events": self._analyze_risk_events(audit_records),
            "access_patterns": self._analyze_access_patterns(audit_records),
            "anomalies": self._detect_audit_anomalies(audit_records),
        }

        # Generate base report
        base_report = self.generate_executive_summary_report(
            frameworks, audit_generator, model_version
        )
        base_report.report_type = ReportType.AUDIT_REPORT
        base_report.report_id = f"audit_report_{int(datetime.now().timestamp())}"
        base_report.reporting_period_start = start_date.isoformat()
        base_report.reporting_period_end = end_date.isoformat()

        # Update with audit-specific information
        base_report.audit_findings = self._generate_detailed_audit_findings(
            audit_records, audit_integrity
        )
        base_report.appendices["audit_analysis"] = audit_analysis

        return base_report

    def export_report(
        self,
        report: ComplianceReport,
        format: str = "json",
        include_appendices: bool = True,
    ) -> str:
        """Export compliance report in specified format."""

        if format.lower() == "json":
            report_dict = asdict(report)
            # Convert enum to string
            report_dict["report_type"] = report.report_type.value

            if not include_appendices:
                report_dict.pop("appendices", None)

            return json.dumps(report_dict, indent=2, default=str)

        elif format.lower() == "html":
            return self._generate_html_report(report, include_appendices)

        elif format.lower() == "pdf":
            return self._generate_pdf_report(report, include_appendices)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _analyze_audit_records(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> Dict[str, Any]:
        """Analyze audit records for statistics."""

        if not audit_records:
            return {
                "total_events": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "events_by_type": {},
                "events_by_day": {},
                "compliance_violations": 0,
            }

        stats = {
            "total_events": len(audit_records),
            "high_risk_count": len(
                [r for r in audit_records if r.risk_level == "high"]
            ),
            "medium_risk_count": len(
                [r for r in audit_records if r.risk_level == "medium"]
            ),
            "low_risk_count": len([r for r in audit_records if r.risk_level == "low"]),
            "events_by_type": {},
            "events_by_day": {},
            "compliance_violations": len(
                [r for r in audit_records if r.compliance_status != "compliant"]
            ),
        }

        # Count events by type
        for record in audit_records:
            event_type = record.event_type.value
            stats["events_by_type"][event_type] = (
                stats["events_by_type"].get(event_type, 0) + 1
            )

        # Count events by day
        for record in audit_records:
            date = record.timestamp.split("T")[0]
            stats["events_by_day"][date] = stats["events_by_day"].get(date, 0) + 1

        return stats

    def _generate_key_findings(
        self, coverage: Dict[str, Any], audit_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate key findings for executive summary."""

        findings = []

        # Coverage findings
        overall_coverage = coverage["overall_coverage"]["coverage_percentage"]
        if overall_coverage >= 95:
            findings.append(
                "Excellent compliance coverage achieved across all frameworks"
            )
        elif overall_coverage >= 80:
            findings.append("Good compliance coverage with minor gaps to address")
        else:
            findings.append("Significant compliance gaps require immediate attention")

        # Audit findings
        if audit_stats["compliance_violations"] > 0:
            findings.append(
                f"Identified {audit_stats['compliance_violations']} compliance violations requiring remediation"
            )

        if audit_stats["high_risk_count"] > 0:
            findings.append(
                f"Detected {audit_stats['high_risk_count']} high-risk events requiring review"
            )

        # Mandatory requirements
        unsatisfied_mandatory = len(
            [r for r in coverage["unsatisfied_requirements"] if r["mandatory"]]
        )
        if unsatisfied_mandatory > 0:
            findings.append(
                f"Critical: {unsatisfied_mandatory} mandatory requirements not satisfied"
            )

        return findings

    def _summarize_audit_findings(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> List[Dict[str, Any]]:
        """Summarize audit findings for report."""

        findings = []

        # Group by risk level
        high_risk_events = [r for r in audit_records if r.risk_level == "high"]
        if high_risk_events:
            findings.append(
                {
                    "finding_type": "high_risk_events",
                    "severity": "high",
                    "count": len(high_risk_events),
                    "description": f"Identified {len(high_risk_events)} high-risk events requiring immediate attention",
                    "events": [r.event_id for r in high_risk_events[:10]],  # First 10
                }
            )

        # Compliance violations
        violations = [r for r in audit_records if r.compliance_status != "compliant"]
        if violations:
            findings.append(
                {
                    "finding_type": "compliance_violations",
                    "severity": "high",
                    "count": len(violations),
                    "description": f"Found {len(violations)} compliance violations",
                    "events": [r.event_id for r in violations[:10]],
                }
            )

        return findings

    def _assess_compliance_risk(
        self, coverage: Dict[str, Any], audit_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall compliance risk."""

        risk_factors = []
        overall_risk = "low"

        # Coverage-based risk
        coverage_pct = coverage["overall_coverage"]["coverage_percentage"]
        if coverage_pct < 70:
            risk_factors.append("Low compliance coverage")
            overall_risk = "high"
        elif coverage_pct < 90:
            risk_factors.append("Moderate compliance gaps")
            overall_risk = "medium"

        # Audit-based risk
        if audit_stats["high_risk_count"] > 0:
            risk_factors.append("High-risk events detected")
            overall_risk = "high"

        if audit_stats["compliance_violations"] > 0:
            risk_factors.append("Compliance violations found")
            overall_risk = "high"

        return {
            "overall_risk_level": overall_risk,
            "risk_factors": risk_factors,
            "risk_score": self._calculate_risk_score(coverage, audit_stats),
            "mitigation_required": overall_risk in ["high", "medium"],
        }

    def _generate_recommendations(
        self, coverage: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate compliance recommendations."""

        recommendations = []

        # Address unsatisfied requirements
        for req in coverage["unsatisfied_requirements"]:
            if req["mandatory"]:
                recommendations.append(
                    {
                        "priority": "high",
                        "type": "mandatory_requirement",
                        "requirement_id": req["requirement_id"],
                        "framework": req["framework"],
                        "title": f"Implement {req['title']}",
                        "description": "This is a mandatory requirement that must be addressed for compliance",
                        "effort": "medium",
                        "timeline": "immediate",
                    }
                )

        # General improvements
        coverage_pct = coverage["overall_coverage"]["coverage_percentage"]
        if coverage_pct < 90:
            recommendations.append(
                {
                    "priority": "medium",
                    "type": "coverage_improvement",
                    "title": "Improve overall compliance coverage",
                    "description": f"Current coverage is {coverage_pct:.1f}%. Target 90%+ for strong compliance posture",
                    "effort": "medium",
                    "timeline": "3-6 months",
                }
            )

        return recommendations

    def _generate_action_items(self, coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific action items."""

        action_items = []

        for req in coverage["unsatisfied_requirements"]:
            action_items.append(
                {
                    "action_id": f"action_{req['requirement_id']}",
                    "requirement_id": req["requirement_id"],
                    "title": f"Implement {req['title']}",
                    "priority": "high" if req["mandatory"] else "medium",
                    "status": "open",
                    "assigned_to": "compliance_team",
                    "due_date": "TBD",
                    "estimated_effort": "TBD",
                }
            )

        return action_items

    def _calculate_compliance_metrics(
        self, coverage: Dict[str, Any], audit_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate compliance metrics."""

        return {
            "compliance_score": coverage["overall_coverage"]["coverage_percentage"],
            "mandatory_compliance_score": coverage["overall_coverage"][
                "mandatory_coverage"
            ],
            "audit_integrity_score": (
                100 if audit_stats["compliance_violations"] == 0 else 80
            ),
            "risk_score": self._calculate_risk_score(coverage, audit_stats),
            "trend": "stable",  # Would be calculated from historical data
        }

    def _calculate_risk_score(
        self, coverage: Dict[str, Any], audit_stats: Dict[str, Any]
    ) -> float:
        """Calculate numerical risk score (0-100, lower is better)."""

        # Base risk from coverage gaps
        coverage_risk = 100 - coverage["overall_coverage"]["coverage_percentage"]

        # Additional risk from violations
        violation_risk = min(audit_stats["compliance_violations"] * 10, 50)

        # High-risk events
        high_risk_penalty = min(audit_stats["high_risk_count"] * 5, 30)

        total_risk = min(coverage_risk + violation_risk + high_risk_penalty, 100)
        return total_risk

    def _gather_requirement_evidence(
        self, requirement, audit_generator: AuditTrailGenerator
    ) -> List[str]:
        """Gather evidence for requirement satisfaction."""

        # This would gather specific evidence from audit trails
        evidence = []

        if requirement.is_satisfied_by_ciaf():
            evidence.append("CIAF capabilities satisfy this requirement")
            evidence.extend(
                [f"Capability: {cap}" for cap in requirement.ciaf_capabilities]
            )
        else:
            evidence.append("No CIAF capabilities mapped to this requirement")

        return evidence

    def _generate_capability_mapping(
        self, frameworks: List[ComplianceFramework]
    ) -> Dict[str, Any]:
        """Generate mapping of CIAF capabilities to requirements."""

        capability_map = {}

        for framework in frameworks:
            requirements = self.regulatory_mapper.get_requirements([framework])
            for req in requirements:
                for capability in req.ciaf_capabilities:
                    if capability not in capability_map:
                        capability_map[capability] = []
                    capability_map[capability].append(
                        {
                            "requirement_id": req.requirement_id,
                            "framework": req.framework.value,
                            "title": req.title,
                        }
                    )

        return capability_map

    def _generate_implementation_timeline(
        self, requirements_assessment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate implementation timeline for gaps."""

        timeline = {
            "immediate": [],
            "short_term": [],
            "medium_term": [],
            "long_term": [],
        }

        for req in requirements_assessment:
            if req["implementation_status"] == "gap":
                if req["mandatory"]:
                    timeline["immediate"].append(req["requirement_id"])
                elif req["risk_level"] == "high":
                    timeline["short_term"].append(req["requirement_id"])
                else:
                    timeline["medium_term"].append(req["requirement_id"])

        return timeline

    def _generate_cost_benefit_analysis(
        self, requirements_assessment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate cost-benefit analysis for implementation."""

        # This would include more detailed cost modeling
        return {
            "total_gaps": len(
                [
                    r
                    for r in requirements_assessment
                    if r["implementation_status"] == "gap"
                ]
            ),
            "mandatory_gaps": len(
                [
                    r
                    for r in requirements_assessment
                    if r["implementation_status"] == "gap" and r["mandatory"]
                ]
            ),
            "estimated_effort": "Medium - Most requirements can be addressed with CIAF configuration",
            "business_risk": "High for mandatory requirements, Medium for optional",
            "implementation_cost": "Low to Medium - Primarily configuration and documentation",
        }

    def _generate_gap_remediation_plan(
        self, coverage: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific remediation plan for gaps."""

        plan = []

        for req in coverage["unsatisfied_requirements"]:
            plan.append(
                {
                    "requirement_id": req["requirement_id"],
                    "framework": req["framework"],
                    "title": req["title"],
                    "priority": "high" if req["mandatory"] else "medium",
                    "remediation_approach": "Implement additional CIAF configuration or external controls",
                    "estimated_timeline": "1-3 months",
                    "dependencies": [],
                    "success_criteria": "Requirement fully satisfied and documented",
                }
            )

        return plan

    def _analyze_compliance_events(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> Dict[str, Any]:
        """Analyze compliance-specific events."""

        compliance_events = [
            r for r in audit_records if "compliance" in r.event_type.value
        ]

        return {
            "total_compliance_events": len(compliance_events),
            "compliance_checks": len(
                [r for r in compliance_events if "check" in r.event_type.value]
            ),
            "violations": len(
                [r for r in audit_records if r.compliance_status != "compliant"]
            ),
            "remediation_actions": 0,  # Would track remediation events
        }

    def _analyze_risk_events(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> Dict[str, Any]:
        """Analyze risk-related events."""

        return {
            "high_risk_events": len(
                [r for r in audit_records if r.risk_level == "high"]
            ),
            "medium_risk_events": len(
                [r for r in audit_records if r.risk_level == "medium"]
            ),
            "low_risk_events": len([r for r in audit_records if r.risk_level == "low"]),
            "risk_trends": "stable",  # Would analyze trends over time
        }

    def _analyze_access_patterns(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> Dict[str, Any]:
        """Analyze data access patterns."""

        access_events = [r for r in audit_records if "access" in r.event_type.value]

        return {
            "total_access_events": len(access_events),
            "unique_users": len(set(r.user_id for r in access_events if r.user_id)),
            "access_violations": len(
                [r for r in access_events if r.compliance_status != "compliant"]
            ),
            "after_hours_access": 0,  # Would analyze timestamps
        }

    def _detect_audit_anomalies(
        self, audit_records: List[ComplianceAuditRecord]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in audit records."""

        anomalies = []

        # Simple anomaly detection - would be more sophisticated in practice
        high_risk_events = [r for r in audit_records if r.risk_level == "high"]
        if len(high_risk_events) > 10:  # Threshold
            anomalies.append(
                {
                    "type": "high_risk_spike",
                    "description": f"Unusually high number of high-risk events: {len(high_risk_events)}",
                    "severity": "medium",
                }
            )

        return anomalies

    def _generate_detailed_audit_findings(
        self,
        audit_records: List[ComplianceAuditRecord],
        audit_integrity: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate detailed audit findings."""

        findings = []

        # Integrity findings
        if not audit_integrity["integrity_verified"]:
            findings.append(
                {
                    "finding_type": "audit_integrity",
                    "severity": "critical",
                    "description": "Audit trail integrity verification failed",
                    "details": audit_integrity,
                    "recommendation": "Investigate audit trail corruption and restore from backup",
                }
            )

        # Add other detailed findings
        findings.extend(self._summarize_audit_findings(audit_records))

        return findings

    def _generate_html_report(
        self, report: ComplianceReport, include_appendices: bool
    ) -> str:
        """Generate HTML version of compliance report."""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CIAF Compliance Report - {report.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .finding {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
                .critical {{ border-left-color: #d32f2f; }}
                .high {{ border-left-color: #f57c00; }}
                .medium {{ border-left-color: #fbc02d; }}
                .low {{ border-left-color: #388e3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CIAF Compliance Report</h1>
                <p><strong>Model:</strong> {report.model_name} v{report.model_version}</p>
                <p><strong>Report Type:</strong> {report.report_type.value}</p>
                <p><strong>Generated:</strong> {report.generated_date}</p>
                <p><strong>Frameworks:</strong> {', '.join(report.frameworks)}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Compliance Score:</strong> {report.executive_summary.get('overall_compliance_score', 'N/A'):.1f}%</p>
                <p><strong>Total Requirements:</strong> {report.executive_summary.get('total_requirements', 'N/A')}</p>
                <p><strong>Satisfied Requirements:</strong> {report.executive_summary.get('satisfied_requirements', 'N/A')}</p>
                
                <h3>Key Findings</h3>
                <ul>
        """

        for finding in report.executive_summary.get("key_findings", []):
            html += f"<li>{finding}</li>"

        html += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Compliance Status</h2>
        """

        for framework, status in report.compliance_status.get(
            "framework_status", {}
        ).items():
            html += f"""
                <div class="finding">
                    <h4>{framework.upper()}</h4>
                    <p><strong>Status:</strong> {status['status']}</p>
                    <p><strong>Coverage:</strong> {status['coverage_percentage']:.1f}%</p>
                </div>
            """

        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """

        for rec in report.recommendations:
            priority_class = rec.get("priority", "medium")
            html += f"""
                <div class="finding {priority_class}">
                    <h4>{rec.get('title', 'Recommendation')}</h4>
                    <p><strong>Priority:</strong> {rec.get('priority', 'medium')}</p>
                    <p>{rec.get('description', '')}</p>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _generate_pdf_report(
        self, report: ComplianceReport, include_appendices: bool = True
    ) -> str:
        """Generate PDF report from HTML."""
        try:
            # Try to use weasyprint for PDF generation
            try:
                from weasyprint import HTML, CSS
                html_content = self._generate_html_report(report, include_appendices)
                
                # Basic CSS for PDF styling
                css_content = CSS(string="""
                    @page { margin: 2cm; }
                    body { font-family: Arial, sans-serif; font-size: 10pt; }
                    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                    h2 { color: #34495e; margin-top: 20px; }
                    .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .risk-high { color: #e74c3c; font-weight: bold; }
                    .risk-medium { color: #f39c12; }
                    .risk-low { color: #27ae60; }
                    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f8f9fa; }
                """)
                
                # Generate PDF
                html_doc = HTML(string=html_content)
                pdf_bytes = html_doc.write_pdf(stylesheets=[css_content])
                
                # Save to temporary file and return path
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(pdf_bytes)
                    return tmp_file.name
                    
            except ImportError:
                # Fallback: Try reportlab
                try:
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib import colors
                    import tempfile
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                        doc = SimpleDocTemplate(tmp_file.name, pagesize=A4)
                        styles = getSampleStyleSheet()
                        story = []
                        
                        # Title
                        title_style = ParagraphStyle(
                            'CustomTitle',
                            parent=styles['Heading1'],
                            fontSize=18,
                            spaceAfter=30,
                        )
                        story.append(Paragraph(f"Compliance Report: {report.model_name}", title_style))
                        story.append(Spacer(1, 12))
                        
                        # Summary
                        story.append(Paragraph("Executive Summary", styles['Heading2']))
                        story.append(Paragraph(f"Report ID: {report.report_id}", styles['Normal']))
                        story.append(Paragraph(f"Model: {report.model_name} v{report.model_version}", styles['Normal']))
                        story.append(Paragraph(f"Generated: {report.generation_timestamp}", styles['Normal']))
                        story.append(Paragraph(f"Total Events: {report.summary.total_events}", styles['Normal']))
                        story.append(Spacer(1, 12))
                        
                        # Risk Assessment
                        if hasattr(report, 'risk_assessment') and report.risk_assessment:
                            story.append(Paragraph("Risk Assessment", styles['Heading2']))
                            risk_data = [
                                ['Risk Level', 'Count', 'Percentage'],
                                ['High', str(report.summary.high_risk_events), f"{(report.summary.high_risk_events/max(report.summary.total_events,1)*100):.1f}%"],
                                ['Medium', str(report.summary.medium_risk_events), f"{(report.summary.medium_risk_events/max(report.summary.total_events,1)*100):.1f}%"],
                                ['Low', str(report.summary.low_risk_events), f"{(report.summary.low_risk_events/max(report.summary.total_events,1)*100):.1f}%"],
                            ]
                            
                            risk_table = Table(risk_data)
                            risk_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 14),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(risk_table)
                            story.append(Spacer(1, 12))
                        
                        # Compliance Status
                        story.append(Paragraph("Compliance Framework Status", styles['Heading2']))
                        if hasattr(report, 'framework_compliance'):
                            for framework, status in report.framework_compliance.items():
                                story.append(Paragraph(f"• {framework}: {status}", styles['Normal']))
                        story.append(Spacer(1, 12))
                        
                        # Recommendations
                        if hasattr(report, 'recommendations') and report.recommendations:
                            story.append(Paragraph("Recommendations", styles['Heading2']))
                            for i, rec in enumerate(report.recommendations[:10], 1):  # Limit to 10
                                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                        
                        # Build PDF
                        doc.build(story)
                        return tmp_file.name
                        
                except ImportError:
                    # Final fallback: Return HTML with instructions
                    html_content = self._generate_html_report(report, include_appendices)
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                        tmp_file.write(html_content)
                        html_path = tmp_file.name
                    
                    return f"""PDF generation libraries not available. 
HTML report saved to: {html_path}

To enable PDF generation, install one of:
- weasyprint: pip install weasyprint
- reportlab: pip install reportlab

Then convert HTML to PDF manually or programmatically."""
                        
        except Exception as e:
            return f"PDF generation failed: {str(e)}. Falling back to HTML format."
