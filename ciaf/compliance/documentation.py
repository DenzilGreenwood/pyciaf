"""
Compliance Documentation Generator for CIAF

This module provides automated generation of compliance documentation
for various regulatory frameworks and standards.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .audit_trails import AuditEventType, AuditTrailGenerator, ComplianceAuditRecord
from .regulatory_mapping import (
    ComplianceFramework,
    ComplianceRequirement,
    RegulatoryMapper,
)
from .validators import ComplianceValidator, ValidationResult, ValidationSeverity


class DocumentationType(Enum):
    """Types of compliance documents."""

    TECHNICAL_SPECIFICATION = "technical_specification"
    RISK_ASSESSMENT = "risk_assessment"
    PRIVACY_IMPACT_ASSESSMENT = "privacy_impact_assessment"
    ALGORITHMIC_IMPACT_ASSESSMENT = "algorithmic_impact_assessment"
    COMPLIANCE_MANUAL = "compliance_manual"
    AUDIT_REPORT = "audit_report"
    INCIDENT_RESPONSE_PLAN = "incident_response_plan"
    DATA_GOVERNANCE_POLICY = "data_governance_policy"
    TRANSPARENCY_REPORT = "transparency_report"
    USER_GUIDE = "user_guide"


@dataclass
class DocumentSection:
    """A section within a compliance document."""

    section_id: str
    title: str
    content: str
    subsections: List["DocumentSection"]
    references: List[str]
    compliance_mappings: List[str]

    def to_html(self, level: int = 1) -> str:
        """Convert section to HTML."""
        html = f"<h{level} id='{self.section_id}'>{self.title}</h{level}>\n"
        html += f"<div class='section-content'>\n{self.content}\n</div>\n"

        if self.references:
            html += "<div class='references'>\n<h5>References:</h5>\n<ul>\n"
            for ref in self.references:
                html += f"<li>{ref}</li>\n"
            html += "</ul>\n</div>\n"

        for subsection in self.subsections:
            html += subsection.to_html(level + 1)

        return html

    def to_markdown(self, level: int = 1) -> str:
        """Convert section to Markdown."""
        prefix = "#" * level
        md = f"{prefix} {self.title}\n\n"
        md += f"{self.content}\n\n"

        if self.references:
            md += "### References\n\n"
            for ref in self.references:
                md += f"- {ref}\n"
            md += "\n"

        for subsection in self.subsections:
            md += subsection.to_markdown(level + 1)

        return md


@dataclass
class ComplianceDocument:
    """A complete compliance document."""

    document_id: str
    title: str
    document_type: DocumentationType
    framework: ComplianceFramework
    model_name: str
    version: str
    sections: List[DocumentSection]
    metadata: Dict[str, Any]
    created_date: str
    last_updated: str

    def to_html(self) -> str:
        """Convert document to HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ border-bottom: 2px solid #ccc; padding-bottom: 20px; margin-bottom: 30px; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
        .section-content {{ margin-bottom: 20px; }}
        .references {{ background: #f9f9f9; padding: 10px; border-left: 4px solid #007acc; margin-top: 15px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #666; }}
        .toc {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .toc ul {{ list-style-type: none; padding-left: 20px; }}
        .toc a {{ text-decoration: none; color: #007acc; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p><strong>Framework:</strong> {self.framework.value}</p>
        <p><strong>Model:</strong> {self.model_name}</p>
        <p><strong>Version:</strong> {self.version}</p>
    </div>
    
    <div class="metadata">
        <h3>Document Information</h3>
        <p><strong>Document ID:</strong> {self.document_id}</p>
        <p><strong>Document Type:</strong> {self.document_type.value}</p>
        <p><strong>Created:</strong> {self.created_date}</p>
        <p><strong>Last Updated:</strong> {self.last_updated}</p>
    </div>
"""

        # Table of Contents
        html += self._generate_toc()

        # Sections
        for section in self.sections:
            html += section.to_html()

        html += """
</body>
</html>"""
        return html

    def to_markdown(self) -> str:
        """Convert document to Markdown."""
        md = f"# {self.title}\n\n"
        md += f"**Framework:** {self.framework.value}\n\n"
        md += f"**Model:** {self.model_name}\n\n"
        md += f"**Version:** {self.version}\n\n"
        md += f"**Document ID:** {self.document_id}\n\n"
        md += f"**Document Type:** {self.document_type.value}\n\n"
        md += f"**Created:** {self.created_date}\n\n"
        md += f"**Last Updated:** {self.last_updated}\n\n"
        md += "---\n\n"

        for section in self.sections:
            md += section.to_markdown()

        return md

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        html = "<div class='toc'>\n<h3>Table of Contents</h3>\n<ul>\n"

        for section in self.sections:
            html += f"<li><a href='#{section.section_id}'>{section.title}</a>\n"
            if section.subsections:
                html += "<ul>\n"
                for subsection in section.subsections:
                    html += f"<li><a href='#{subsection.section_id}'>{subsection.title}</a></li>\n"
                html += "</ul>\n"
            html += "</li>\n"

        html += "</ul>\n</div>\n"
        return html


class ComplianceDocumentationGenerator:
    """Automated compliance documentation generator."""

    def __init__(self, model_name: str):
        """Initialize documentation generator."""
        self.model_name = model_name
        self.regulatory_mapper = RegulatoryMapper()
        self.generated_documents: List[ComplianceDocument] = []

    def generate_technical_specification(
        self, framework: ComplianceFramework, model_version: str = "current"
    ) -> ComplianceDocument:
        """Generate technical specification document."""

        sections = []
        requirements = self.regulatory_mapper.get_requirements([framework])

        # Executive Summary
        sections.append(
            DocumentSection(
                section_id="executive_summary",
                title="Executive Summary",
                content=f"""This technical specification document outlines the compliance implementation 
            for the {self.model_name} AI model with respect to {framework.value} requirements. 
            The document provides detailed technical information about how the Cognitive Insight Audit Framework (CIAF) 
            addresses regulatory requirements through automated provenance tracking, cryptographic integrity, 
            and comprehensive audit capabilities.""",
                subsections=[],
                references=[],
                compliance_mappings=[framework.value],
            )
        )

        # System Architecture
        sections.append(
            DocumentSection(
                section_id="system_architecture",
                title="System Architecture",
                content=f"""The CIAF system implements a modular architecture designed for comprehensive 
            AI model compliance and transparency:
            
            **Core Components:**
            - **Provenance Engine**: Tracks complete data lineage and model decisions
            - **Cryptographic Layer**: Ensures integrity using AES-GCM, HMAC-SHA256, and PBKDF2
            - **Audit System**: Maintains tamper-evident audit trails with hash connections
            - **Compliance Validators**: Automated compliance checking against multiple frameworks
            - **Risk Assessment Engine**: Continuous risk monitoring and assessment
            
            **Key Features:**
            - Lazy Capsule Materialization (29,000x+ performance improvement)
            - Multi-framework compliance support ({', '.join([f.value for f in ComplianceFramework])})
            - Real-time monitoring and alerting
            - Comprehensive documentation generation""",
                subsections=[],
                references=[
                    "CIAF Technical Architecture v2.1",
                    "Cryptographic Standards FIPS 140-2",
                ],
                compliance_mappings=[framework.value],
            )
        )

        # Compliance Mapping
        compliance_content = f"The following table maps {framework.value} requirements to CIAF capabilities:\n\n"
        compliance_content += (
            "| Requirement | CIAF Capability | Implementation Status |\n"
        )
        compliance_content += (
            "|-------------|-----------------|----------------------|\n"
        )

        for req in requirements[:10]:  # Limit for readability
            capabilities = (
                ", ".join(req.ciaf_capabilities)
                if req.ciaf_capabilities
                else "Manual Implementation Required"
            )
            status = "✅ Automated" if req.ciaf_capabilities else "⚠️ Manual"
            compliance_content += f"| {req.title} | {capabilities} | {status} |\n"

        sections.append(
            DocumentSection(
                section_id="compliance_mapping",
                title="Compliance Mapping",
                content=compliance_content,
                subsections=[],
                references=[f"{framework.value} Official Requirements"],
                compliance_mappings=[framework.value],
            )
        )

        # Data Governance
        sections.append(
            DocumentSection(
                section_id="data_governance",
                title="Data Governance",
                content=f"""CIAF implements comprehensive data governance controls:
            
            **Data Lifecycle Management:**
            - Complete data lineage tracking from ingestion to disposal
            - Automated PII detection and protection
            - Data encryption at rest and in transit (AES-256-GCM)
            - Access control and permission management
            
            **Data Quality Assurance:**
            - Automated data validation and quality checks
            - Data integrity verification through cryptographic hashing
            - Version control for all datasets
            - Automated backup and recovery procedures
            
            **Privacy Protection:**
            - Privacy-preserving techniques (differential privacy, federated learning)
            - Consent management and tracking
            - Right to erasure implementation
            - Cross-border data transfer controls""",
                subsections=[],
                references=["Data Governance Policy v1.2", "Privacy Impact Assessment"],
                compliance_mappings=[framework.value, "GDPR", "CCPA"],
            )
        )

        # Risk Management
        sections.append(
            DocumentSection(
                section_id="risk_management",
                title="Risk Management",
                content=f"""CIAF incorporates continuous risk assessment and management:
            
            **Risk Assessment Framework:**
            - Automated bias detection and measurement
            - Performance monitoring and drift detection
            - Security vulnerability scanning
            - Impact assessment for model decisions
            
            **Risk Mitigation Strategies:**
            - Real-time model performance monitoring
            - Automated alerting for anomalous behavior
            - Circuit breaker patterns for high-risk scenarios
            - Rollback capabilities for model versions
            
            **Continuous Monitoring:**
            - 24/7 system health monitoring
            - Performance metrics tracking
            - Security event monitoring
            - Compliance status dashboard""",
                subsections=[],
                references=["Risk Management Framework", "ISO 31000:2018"],
                compliance_mappings=[framework.value, "NIST_AI_RMF"],
            )
        )

        # Audit and Transparency
        sections.append(
            DocumentSection(
                section_id="audit_transparency",
                title="Audit and Transparency",
                content=f"""CIAF provides comprehensive audit and transparency capabilities:
            
            **Audit Trail Generation:**
            - Immutable audit logs with cryptographic integrity
            - Complete event tracking (data access, model predictions, system changes)
            - Hash connections verification for tamper detection
            - Automated audit report generation
            
            **Transparency Features:**
            - Model decision explanations (SHAP, LIME integration)
            - Data source attribution
            - Processing history visualization
            - Public transparency reports
            
            **Compliance Monitoring:**
            - Real-time compliance status monitoring
            - Automated violation detection
            - Compliance dashboard and reporting
            - Integration with external audit systems""",
                subsections=[],
                references=["Audit Trail Specification", "Transparency Guidelines"],
                compliance_mappings=[framework.value],
            )
        )

        document = ComplianceDocument(
            document_id=f"TECH_SPEC_{framework.value}_{datetime.now().strftime('%Y%m%d')}",
            title=f"Technical Specification - {framework.value} Compliance",
            document_type=DocumentationType.TECHNICAL_SPECIFICATION,
            framework=framework,
            model_name=self.model_name,
            version=model_version,
            sections=sections,
            metadata={
                "requirements_count": len(requirements),
                "automated_requirements": len(
                    [r for r in requirements if r.ciaf_capabilities]
                ),
                "manual_requirements": len(
                    [r for r in requirements if not r.ciaf_capabilities]
                ),
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.generated_documents.append(document)
        return document

    def generate_risk_assessment(
        self,
        framework: ComplianceFramework,
        audit_generator: AuditTrailGenerator,
        assessment_period_days: int = 90,
    ) -> ComplianceDocument:
        """Generate comprehensive risk assessment document."""

        sections = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=assessment_period_days)

        # Get audit data for risk analysis
        all_events = audit_generator.get_audit_trail(start_date, end_date)
        high_risk_events = [e for e in all_events if e.risk_level == "high"]

        # Executive Summary
        sections.append(
            DocumentSection(
                section_id="risk_executive_summary",
                title="Executive Summary",
                content=f"""This risk assessment evaluates the {self.model_name} AI model over a {assessment_period_days}-day period 
            from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. 
            
            **Key Findings:**
            - Total events analyzed: {len(all_events):,}
            - High-risk events identified: {len(high_risk_events)}
            - Overall risk level: {'HIGH' if len(high_risk_events) > 50 else 'MEDIUM' if len(high_risk_events) > 10 else 'LOW'}
            - Compliance status: {'COMPLIANT' if len(high_risk_events) < 10 else 'REQUIRES ATTENTION'}""",
                subsections=[],
                references=[],
                compliance_mappings=[framework.value],
            )
        )

        # Risk Identification
        risk_types = {}
        for event in all_events:
            risk_type = event.metadata.get("risk_type", "unknown")
            if risk_type not in risk_types:
                risk_types[risk_type] = []
            risk_types[risk_type].append(event)

        risk_content = "The following risk categories have been identified:\n\n"
        for risk_type, events in risk_types.items():
            high_risk_count = len([e for e in events if e.risk_level == "high"])
            risk_content += f"**{risk_type.title()}:**\n"
            risk_content += f"- Total events: {len(events)}\n"
            risk_content += f"- High-risk events: {high_risk_count}\n"
            risk_content += f"- Risk level: {'HIGH' if high_risk_count > 5 else 'MEDIUM' if high_risk_count > 0 else 'LOW'}\n\n"

        sections.append(
            DocumentSection(
                section_id="risk_identification",
                title="Risk Identification",
                content=risk_content,
                subsections=[],
                references=["Risk Classification Framework"],
                compliance_mappings=[framework.value],
            )
        )

        # Risk Analysis
        sections.append(
            DocumentSection(
                section_id="risk_analysis",
                title="Risk Analysis",
                content=f"""**Quantitative Risk Analysis:**
            
            **Event Distribution by Risk Level:**
            - High Risk: {len([e for e in all_events if e.risk_level == 'high'])} ({len([e for e in all_events if e.risk_level == 'high'])/len(all_events)*100:.1f}%)
            - Medium Risk: {len([e for e in all_events if e.risk_level == 'medium'])} ({len([e for e in all_events if e.risk_level == 'medium'])/len(all_events)*100:.1f}%)
            - Low Risk: {len([e for e in all_events if e.risk_level == 'low'])} ({len([e for e in all_events if e.risk_level == 'low'])/len(all_events)*100:.1f}%)
            
            **Temporal Risk Patterns:**
            - Risk events are distributed across the assessment period
            - Peak risk periods identified for enhanced monitoring
            - Trend analysis indicates {'increasing' if len(high_risk_events) > 20 else 'stable'} risk profile
            
            **Impact Assessment:**
            - Data exposure risk: {'HIGH' if any(e.contains_pii for e in high_risk_events) else 'LOW'}
            - Model integrity risk: {'MEDIUM' if len(high_risk_events) > 10 else 'LOW'}
            - Compliance risk: {'HIGH' if len(high_risk_events) > 25 else 'MEDIUM' if len(high_risk_events) > 10 else 'LOW'}""",
                subsections=[],
                references=["Quantitative Risk Analysis Methodology"],
                compliance_mappings=[framework.value],
            )
        )

        # Risk Mitigation
        sections.append(
            DocumentSection(
                section_id="risk_mitigation",
                title="Risk Mitigation Strategies",
                content=f"""**Implemented Controls:**
            
            **Technical Controls:**
            - Automated risk monitoring and alerting
            - Real-time anomaly detection
            - Cryptographic integrity protection
            - Access control and authentication
            
            **Operational Controls:**
            - Regular security assessments
            - Incident response procedures
            - Staff training and awareness
            - Vendor management processes
            
            **Recommended Additional Controls:**
            {'- Implement additional monitoring for high-risk events' if len(high_risk_events) > 10 else '- Continue current monitoring practices'}
            {'- Review and update risk thresholds' if len(high_risk_events) > 25 else '- Maintain current risk thresholds'}
            - Regular review of risk assessment procedures
            - Enhanced logging for critical operations""",
                subsections=[],
                references=[
                    "Risk Mitigation Framework",
                    "Control Implementation Guide",
                ],
                compliance_mappings=[framework.value],
            )
        )

        document = ComplianceDocument(
            document_id=f"RISK_ASSESS_{framework.value}_{datetime.now().strftime('%Y%m%d')}",
            title=f"Risk Assessment - {framework.value}",
            document_type=DocumentationType.RISK_ASSESSMENT,
            framework=framework,
            model_name=self.model_name,
            version="current",
            sections=sections,
            metadata={
                "assessment_period_days": assessment_period_days,
                "total_events": len(all_events),
                "high_risk_events": len(high_risk_events),
                "risk_level": (
                    "HIGH"
                    if len(high_risk_events) > 50
                    else "MEDIUM" if len(high_risk_events) > 10 else "LOW"
                ),
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.generated_documents.append(document)
        return document

    def generate_compliance_manual(
        self, frameworks: List[ComplianceFramework]
    ) -> ComplianceDocument:
        """Generate comprehensive compliance manual."""

        sections = []

        # Introduction
        sections.append(
            DocumentSection(
                section_id="introduction",
                title="Introduction",
                content=f"""This compliance manual provides comprehensive guidance for ensuring 
            {self.model_name} AI model compliance with multiple regulatory frameworks. 
            
            **Covered Frameworks:**
            {chr(10).join([f'- {framework.value}' for framework in frameworks])}
            
            **Manual Scope:**
            - Implementation procedures
            - Compliance checklists
            - Monitoring guidelines
            - Incident response procedures
            - Documentation requirements""",
                subsections=[],
                references=[],
                compliance_mappings=[f.value for f in frameworks],
            )
        )

        # Implementation Procedures
        impl_content = "**Step-by-Step Implementation:**\n\n"
        impl_content += "1. **Environment Setup**\n"
        impl_content += "   - Install CIAF framework\n"
        impl_content += "   - Configure compliance settings\n"
        impl_content += "   - Initialize audit systems\n\n"
        impl_content += "2. **Model Integration**\n"
        impl_content += "   - Wrap model with CIAFModelWrapper\n"
        impl_content += "   - Configure provenance tracking\n"
        impl_content += "   - Enable audit logging\n\n"
        impl_content += "3. **Compliance Configuration**\n"
        impl_content += "   - Select applicable frameworks\n"
        impl_content += "   - Configure validation rules\n"
        impl_content += "   - Set up monitoring dashboards\n\n"
        impl_content += "4. **Testing and Validation**\n"
        impl_content += "   - Run compliance validators\n"
        impl_content += "   - Verify audit trail integrity\n"
        impl_content += "   - Generate compliance reports\n"

        sections.append(
            DocumentSection(
                section_id="implementation",
                title="Implementation Procedures",
                content=impl_content,
                subsections=[],
                references=["CIAF Installation Guide", "Quick Start Documentation"],
                compliance_mappings=[f.value for f in frameworks],
            )
        )

        # Compliance Checklists
        for framework in frameworks:
            requirements = self.regulatory_mapper.get_requirements([framework])
            checklist_content = f"**{framework.value} Compliance Checklist:**\n\n"

            for i, req in enumerate(requirements[:15], 1):  # Limit for readability
                status = "Good" if req.ciaf_capabilities else "Errror"
                checklist_content += f"{i}. {status} {req.title}\n"
                if req.ciaf_capabilities:
                    checklist_content += (
                        f"   - Automated by: {', '.join(req.ciaf_capabilities)}\n"
                    )
                else:
                    checklist_content += f"   - Manual implementation required\n"
                checklist_content += (
                    f"   - Priority: {'High' if req.mandatory else 'Medium'}\n\n"
                )

            sections.append(
                DocumentSection(
                    section_id=f"checklist_{framework.value.lower()}",
                    title=f"{framework.value} Checklist",
                    content=checklist_content,
                    subsections=[],
                    references=[f"{framework.value} Requirements Documentation"],
                    compliance_mappings=[framework.value],
                )
            )

        document = ComplianceDocument(
            document_id=f"COMPLIANCE_MANUAL_{datetime.now().strftime('%Y%m%d')}",
            title="Comprehensive Compliance Manual",
            document_type=DocumentationType.COMPLIANCE_MANUAL,
            framework=frameworks[0],  # Primary framework
            model_name=self.model_name,
            version="current",
            sections=sections,
            metadata={
                "frameworks_covered": [f.value for f in frameworks],
                "total_requirements": sum(
                    len(self.regulatory_mapper.get_requirements([f]))
                    for f in frameworks
                ),
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.generated_documents.append(document)
        return document

    def generate_audit_report(
        self,
        framework: ComplianceFramework,
        validator: ComplianceValidator,
        audit_generator: AuditTrailGenerator,
        reporting_period_days: int = 30,
    ) -> ComplianceDocument:
        """Generate comprehensive audit report."""

        sections = []
        validation_summary = validator.get_validation_summary()

        # Audit Summary
        sections.append(
            DocumentSection(
                section_id="audit_summary",
                title="Audit Summary",
                content=f"""**Audit Period:** {reporting_period_days} days
            **Framework:** {framework.value}
            **Model:** {self.model_name}
            
            **Compliance Status:**
            - Total Validations: {validation_summary.get('total_validations', 0)}
            - Passing: {validation_summary.get('passing', 0)}
            - Failing: {validation_summary.get('failing', 0)}
            - Warnings: {validation_summary.get('warnings', 0)}
            - Pass Rate: {validation_summary.get('pass_rate', 0):.1f}%
            - Overall Status: {validation_summary.get('overall_status', 'unknown').upper()}""",
                subsections=[],
                references=[],
                compliance_mappings=[framework.value],
            )
        )

        # Detailed Findings
        failing_validations = validator.get_failing_validations()
        findings_content = "**Critical Findings:**\n\n"

        if failing_validations:
            for validation in failing_validations:
                findings_content += f"**{validation.title}**\n"
                findings_content += f"- Severity: {validation.severity.value.upper()}\n"
                findings_content += f"- Status: {validation.status.upper()}\n"
                findings_content += f"- Message: {validation.message}\n"
                if validation.recommendations:
                    findings_content += (
                        f"- Recommendations: {'; '.join(validation.recommendations)}\n"
                    )
                findings_content += "\n"
        else:
            findings_content += (
                "No critical findings identified. All validations are passing.\n"
            )

        sections.append(
            DocumentSection(
                section_id="detailed_findings",
                title="Detailed Findings",
                content=findings_content,
                subsections=[],
                references=["Validation Results"],
                compliance_mappings=[framework.value],
            )
        )

        # Recommendations
        all_recommendations = []
        for validation in validator.validation_results:
            all_recommendations.extend(validation.recommendations)

        unique_recommendations = list(set(all_recommendations))

        rec_content = "**Recommended Actions:**\n\n"
        if unique_recommendations:
            for i, rec in enumerate(unique_recommendations, 1):
                rec_content += f"{i}. {rec}\n"
        else:
            rec_content += "No specific recommendations at this time. Continue current compliance practices.\n"

        sections.append(
            DocumentSection(
                section_id="recommendations",
                title="Recommendations",
                content=rec_content,
                subsections=[],
                references=[],
                compliance_mappings=[framework.value],
            )
        )

        document = ComplianceDocument(
            document_id=f"AUDIT_REPORT_{framework.value}_{datetime.now().strftime('%Y%m%d')}",
            title=f"Audit Report - {framework.value}",
            document_type=DocumentationType.AUDIT_REPORT,
            framework=framework,
            model_name=self.model_name,
            version="current",
            sections=sections,
            metadata={
                "reporting_period_days": reporting_period_days,
                "validation_summary": validation_summary,
                "critical_findings": len(failing_validations),
            },
            created_date=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.generated_documents.append(document)
        return document

    def save_document(
        self, document: ComplianceDocument, output_dir: str, format: str = "html"
    ) -> str:
        """Save document to file."""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{document.document_id}.{format.lower()}"
        filepath = os.path.join(output_dir, filename)

        if format.lower() == "html":
            content = document.to_html()
        elif format.lower() == "markdown" or format.lower() == "md":
            content = document.to_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def generate_document_index(self, output_dir: str) -> str:
        """Generate index of all documents."""

        index_html = (
            """<!DOCTYPE html>
<html>
<head>
    <title>Compliance Documentation Index</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .doc-link { color: #007acc; text-decoration: none; }
        .doc-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Compliance Documentation Index</h1>
    <p>Generated on: """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            + """</p>
    
    <table>
        <tr>
            <th>Document ID</th>
            <th>Title</th>
            <th>Type</th>
            <th>Framework</th>
            <th>Created</th>
            <th>Actions</th>
        </tr>
"""
        )

        for doc in self.generated_documents:
            index_html += f"""
        <tr>
            <td>{doc.document_id}</td>
            <td>{doc.title}</td>
            <td>{doc.document_type.value.replace('_', ' ').title()}</td>
            <td>{doc.framework.value}</td>
            <td>{doc.created_date[:10]}</td>
            <td><a href="{doc.document_id}.html" class="doc-link">View</a></td>
        </tr>"""

        index_html += """
    </table>
</body>
</html>"""

        index_path = os.path.join(output_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_html)

        return index_path

    def get_generated_documents(self) -> List[ComplianceDocument]:
        """Get list of all generated documents."""
        return self.generated_documents

    def clear_documents(self):
        """Clear all generated documents."""
        self.generated_documents.clear()
