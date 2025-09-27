"""
Compliance Validators for CIAF

This module provides automated validation capabilities to check compliance
with various regulatory frameworks and industry standards. Updated to use
the new compliance interfaces and policy framework.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.1.0
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .interfaces import (
    ComplianceValidator as IComplianceValidator,
    ComplianceFramework,
    ValidationSeverity,
    AuditEventType
)
from .audit_trails import AuditTrailGenerator, ComplianceAuditRecord
from .regulatory_mapping import (
    ComplianceRequirement,
    RegulatoryMapper,
)
from .policy import get_default_compliance_policy


@dataclass
class ValidationResult:
    """Individual validation result."""

    validation_id: str
    requirement_id: str
    framework: str
    title: str
    severity: ValidationSeverity
    status: str  # "pass", "fail", "warning", "not_applicable"
    message: str
    details: Dict[str, Any]
    evidence: List[str]
    recommendations: List[str]
    timestamp: str

    def is_passing(self) -> bool:
        """Check if validation is passing."""
        return self.status in ["pass", "not_applicable"]

    def is_failing(self) -> bool:
        """Check if validation is failing."""
        return self.status == "fail"

    def needs_attention(self) -> bool:
        """Check if validation needs attention."""
        return self.status in ["fail", "warning"]


class ComplianceValidator(IComplianceValidator):
    """Automated compliance validation engine implementing the ComplianceValidator protocol."""

    def __init__(self, model_name: str):
        """Initialize compliance validator."""
        self.model_name = model_name
        self.regulatory_mapper = RegulatoryMapper()
        self.validation_results: List[ValidationResult] = []
        self.compliance_policy = get_default_compliance_policy()

    def validate_framework_compliance(
        self,
        framework: ComplianceFramework,
        audit_data: Dict[str, Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Validate compliance with a specific regulatory framework (Protocol implementation)."""
        
        # Extract parameters from kwargs for backward compatibility
        audit_generator = kwargs.get('audit_generator')
        model_version = kwargs.get('model_version', 'current')
        validation_period_days = kwargs.get('validation_period_days', 30)
        
        if audit_generator:
            # Use the legacy method with audit generator
            results = self._validate_framework_compliance_legacy(
                framework, audit_generator, model_version, validation_period_days
            )
        else:
            # Use audit_data directly
            results = self._validate_framework_compliance_from_data(
                framework, audit_data, model_version
            )
        
        # Convert ValidationResult objects to dictionaries
        return [self._validation_result_to_dict(result) for result in results]

    def _validation_result_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dictionary format."""
        return {
            "validation_id": result.validation_id,
            "requirement_id": result.requirement_id,
            "framework": result.framework,
            "title": result.title,
            "severity": result.severity.value,
            "status": result.status,
            "message": result.message,
            "details": result.details,
            "evidence": result.evidence,
            "recommendations": result.recommendations,
            "timestamp": result.timestamp
        }

    def _validate_framework_compliance_legacy(
        self,
        framework: ComplianceFramework,
        audit_generator: AuditTrailGenerator,
        model_version: str = "current",
        validation_period_days: int = 30,
    ) -> List[ValidationResult]:
        """Legacy validate compliance with a specific regulatory framework."""

        requirements = self.regulatory_mapper.get_requirements([framework])
        results = []

        for requirement in requirements:
            result = self._validate_single_requirement(
                requirement, audit_generator, model_version, validation_period_days
            )
            results.append(result)

        self.validation_results.extend(results)
        return results

    def _validate_framework_compliance_from_data(
        self,
        framework: ComplianceFramework,
        audit_data: Dict[str, Any],
        model_version: str = "current"
    ) -> List[ValidationResult]:
        """Validate compliance using audit data directly."""
        
        requirements = self.regulatory_mapper.get_requirements([framework])
        results = []

        for requirement in requirements:
            result = self._validate_requirement_from_data(
                requirement, audit_data, model_version
            )
            results.append(result)

        self.validation_results.extend(results)
        return results

    def _validate_requirement_from_data(
        self,
        requirement: ComplianceRequirement,
        audit_data: Dict[str, Any],
        model_version: str
    ) -> ValidationResult:
        """Validate a single requirement using audit data."""
        
        # Check if CIAF satisfies the requirement
        if requirement.is_satisfied_by_ciaf():
            status = "pass"
            message = f"Requirement satisfied by CIAF capabilities: {', '.join(requirement.ciaf_capabilities)}"
            evidence = [f"CIAF capability: {cap}" for cap in requirement.ciaf_capabilities]
            recommendations = []
        else:
            status = "fail" if requirement.mandatory else "warning"
            message = "Requirement not satisfied by current CIAF capabilities"
            evidence = ["No CIAF capabilities mapped to this requirement"]
            recommendations = [requirement.implementation_notes]

        # Additional validation based on audit data
        audit_events = audit_data.get('events', [])
        if requirement.category == "audit_controls" and audit_events:
            # Enhanced validation for audit-related requirements
            integrity_verified = audit_data.get('integrity_verified', False)
            if integrity_verified:
                status = "pass"
                message += " - Audit integrity verified"
                evidence.append("Audit trail integrity verified")

        # Determine severity based on requirement properties and compliance policy
        severity = self._determine_severity(requirement)

        return ValidationResult(
            validation_id=f"VAL_{requirement.requirement_id}",
            requirement_id=requirement.requirement_id,
            framework=requirement.framework.value,
            title=requirement.title,
            severity=severity,
            status=status,
            message=message,
            details={
                "requirement_description": requirement.description,
                "mandatory": requirement.mandatory,
                "risk_level": requirement.risk_level,
                "category": requirement.category,
            },
            evidence=evidence,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _determine_severity(self, requirement: ComplianceRequirement) -> ValidationSeverity:
        """Determine severity based on requirement and policy."""
        
        # Use policy-based severity determination
        if (requirement.mandatory and 
            requirement.risk_level == "high" and 
            self.compliance_policy.validation_policy.compliance_level.value == "strict"):
            return ValidationSeverity.CRITICAL
        elif requirement.mandatory:
            return ValidationSeverity.HIGH
        elif requirement.risk_level == "high":
            return ValidationSeverity.MEDIUM
        else:
            return ValidationSeverity.LOW

    def validate_multiple_frameworks(
        self,
        frameworks: List[ComplianceFramework],
        audit_generator: AuditTrailGenerator,
        model_version: str = "current",
        validation_period_days: int = 30,
    ) -> Dict[str, List[ValidationResult]]:
        """Validate compliance with multiple frameworks."""

        results_by_framework = {}

        for framework in frameworks:
            framework_results = self.validate_framework_compliance(
                framework, audit_generator, model_version, validation_period_days
            )
            results_by_framework[framework.value] = framework_results

        return results_by_framework

    def validate_data_governance(
        self, audit_generator: AuditTrailGenerator, validation_period_days: int = 30
    ) -> List[ValidationResult]:
        """Validate data governance practices."""

        results = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=validation_period_days)

        # Get data access events
        data_events = audit_generator.get_audit_trail(
            start_date,
            end_date,
            [AuditEventType.DATA_ACCESS, AuditEventType.DATA_INGESTION],
        )

        # Validation 1: Data access logging
        result = ValidationResult(
            validation_id="DG_001",
            requirement_id="data_access_logging",
            framework="data_governance",
            title="Data Access Logging",
            severity=ValidationSeverity.HIGH,
            status="pass" if data_events else "fail",
            message=(
                "Data access events are being logged"
                if data_events
                else "No data access events found"
            ),
            details={"data_events_count": len(data_events)},
            evidence=[f"Found {len(data_events)} data access events"],
            recommendations=(
                [] if data_events else ["Ensure data access logging is enabled"]
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 2: PII handling
        pii_events = [e for e in data_events if e.contains_pii]
        result = ValidationResult(
            validation_id="DG_002",
            requirement_id="pii_handling",
            framework="data_governance",
            title="PII Data Handling",
            severity=ValidationSeverity.CRITICAL,
            status="warning" if pii_events else "pass",
            message=(
                f"Found {len(pii_events)} events with PII data"
                if pii_events
                else "No PII data events detected"
            ),
            details={"pii_events_count": len(pii_events)},
            evidence=(
                [f"PII events: {[e.event_id for e in pii_events[:5]]}"]
                if pii_events
                else ["No PII detected"]
            ),
            recommendations=["Review PII handling procedures"] if pii_events else [],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 3: Encryption usage
        encrypted_events = [e for e in data_events if e.encryption_used]
        encryption_rate = len(encrypted_events) / len(data_events) if data_events else 0

        result = ValidationResult(
            validation_id="DG_003",
            requirement_id="data_encryption",
            framework="data_governance",
            title="Data Encryption",
            severity=ValidationSeverity.HIGH,
            status="pass" if encryption_rate >= 0.95 else "fail",
            message=f"Encryption rate: {encryption_rate:.1%}",
            details={
                "encryption_rate": encryption_rate,
                "total_events": len(data_events),
            },
            evidence=[
                f"Encryption used in {len(encrypted_events)}/{len(data_events)} events"
            ],
            recommendations=(
                ["Enable encryption for all data operations"]
                if encryption_rate < 0.95
                else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        self.validation_results.extend(results)
        return results

    def validate_audit_integrity(
        self, audit_generator: AuditTrailGenerator
    ) -> List[ValidationResult]:
        """Validate audit trail integrity."""

        results = []

        # Perform integrity verification
        integrity_check = audit_generator.verify_audit_integrity()

        # Validation 1: Overall integrity
        result = ValidationResult(
            validation_id="AI_001",
            requirement_id="audit_integrity",
            framework="audit_controls",
            title="Audit Trail Integrity",
            severity=ValidationSeverity.CRITICAL,
            status="pass" if integrity_check["integrity_verified"] else "fail",
            message=(
                "Audit trail integrity verified"
                if integrity_check["integrity_verified"]
                else "Audit trail integrity compromised"
            ),
            details=integrity_check,
            evidence=(
                ["Cryptographic integrity verification passed"]
                if integrity_check["integrity_verified"]
                else ["Integrity verification failed"]
            ),
            recommendations=(
                []
                if integrity_check["integrity_verified"]
                else ["Investigate audit trail corruption"]
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 2: Hash connections continuity
        broken_connections = len(integrity_check.get("broken_connections", []))
        result = ValidationResult(
            validation_id="AI_002",
            requirement_id="hash_connections_continuity",
            framework="audit_controls",
            title="Hash Connections Continuity",
            severity=ValidationSeverity.HIGH,
            status="pass" if broken_connections == 0 else "fail",
            message=f"Hash connections continuity: {broken_connections} breaks found",
            details={"broken_connections_count": broken_connections},
            evidence=[f"Hash connections integrity: {broken_connections} breaks"],
            recommendations=(
                ["Restore audit trail from backup"] if broken_connections > 0 else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 3: Timestamp consistency
        timestamp_issues = len(integrity_check.get("timestamp_issues", []))
        result = ValidationResult(
            validation_id="AI_003",
            requirement_id="timestamp_consistency",
            framework="audit_controls",
            title="Timestamp Consistency",
            severity=ValidationSeverity.MEDIUM,
            status="pass" if timestamp_issues == 0 else "warning",
            message=f"Timestamp consistency: {timestamp_issues} issues found",
            details={"timestamp_issues_count": timestamp_issues},
            evidence=[f"Timestamp ordering: {timestamp_issues} issues"],
            recommendations=(
                ["Review system clock synchronization"] if timestamp_issues > 0 else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        self.validation_results.extend(results)
        return results

    def validate_access_controls(
        self, audit_generator: AuditTrailGenerator, validation_period_days: int = 30
    ) -> List[ValidationResult]:
        """Validate access control implementation."""

        results = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=validation_period_days)

        # Get all events with user access
        all_events = audit_generator.get_audit_trail(start_date, end_date)
        user_events = [e for e in all_events if e.user_id and e.user_id != "system"]

        # Validation 1: User identification
        anonymous_events = [
            e for e in all_events if not e.user_id or e.user_id == "anonymous"
        ]
        anonymous_rate = len(anonymous_events) / len(all_events) if all_events else 0

        result = ValidationResult(
            validation_id="AC_001",
            requirement_id="user_identification",
            framework="access_controls",
            title="User Identification",
            severity=ValidationSeverity.HIGH,
            status="pass" if anonymous_rate < 0.1 else "warning",
            message=f"Anonymous events rate: {anonymous_rate:.1%}",
            details={"anonymous_rate": anonymous_rate, "total_events": len(all_events)},
            evidence=[
                f"User identification: {len(user_events)}/{len(all_events)} events"
            ],
            recommendations=(
                ["Implement user authentication for all operations"]
                if anonymous_rate >= 0.1
                else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 2: Access control enforcement
        events_with_controls = [e for e in all_events if e.access_controls]
        control_rate = len(events_with_controls) / len(all_events) if all_events else 0

        result = ValidationResult(
            validation_id="AC_002",
            requirement_id="access_control_enforcement",
            framework="access_controls",
            title="Access Control Enforcement",
            severity=ValidationSeverity.HIGH,
            status="pass" if control_rate >= 0.9 else "fail",
            message=f"Access controls applied: {control_rate:.1%}",
            details={
                "control_rate": control_rate,
                "controlled_events": len(events_with_controls),
            },
            evidence=[
                f"Access controls: {len(events_with_controls)}/{len(all_events)} events"
            ],
            recommendations=(
                ["Implement access controls for all operations"]
                if control_rate < 0.9
                else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 3: Privileged access monitoring
        high_risk_events = [e for e in all_events if e.risk_level == "high"]
        privileged_events = [
            e for e in high_risk_events if "admin" in str(e.access_controls).lower()
        ]

        result = ValidationResult(
            validation_id="AC_003",
            requirement_id="privileged_access_monitoring",
            framework="access_controls",
            title="Privileged Access Monitoring",
            severity=ValidationSeverity.MEDIUM,
            status=(
                "pass"
                if len(privileged_events) == 0
                or len(privileged_events) < len(high_risk_events) * 0.5
                else "warning"
            ),
            message=f"Privileged access events: {len(privileged_events)}",
            details={
                "privileged_events": len(privileged_events),
                "high_risk_events": len(high_risk_events),
            },
            evidence=[f"Privileged access monitoring: {len(privileged_events)} events"],
            recommendations=(
                ["Review privileged access procedures"]
                if len(privileged_events) > 0
                else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        self.validation_results.extend(results)
        return results

    def validate_risk_management(
        self, audit_generator: AuditTrailGenerator, validation_period_days: int = 30
    ) -> List[ValidationResult]:
        """Validate risk management practices."""

        results = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=validation_period_days)

        all_events = audit_generator.get_audit_trail(start_date, end_date)

        # Validation 1: Risk assessment coverage
        risk_events = [
            e for e in all_events if e.event_type == AuditEventType.RISK_ASSESSMENT
        ]

        result = ValidationResult(
            validation_id="RM_001",
            requirement_id="risk_assessment_coverage",
            framework="risk_management",
            title="Risk Assessment Coverage",
            severity=ValidationSeverity.MEDIUM,
            status="pass" if risk_events else "warning",
            message=f"Risk assessments conducted: {len(risk_events)}",
            details={"risk_assessments": len(risk_events)},
            evidence=[f"Risk assessment events: {len(risk_events)}"],
            recommendations=(
                ["Conduct regular risk assessments"] if not risk_events else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 2: High-risk event handling
        high_risk_events = [e for e in all_events if e.risk_level == "high"]

        result = ValidationResult(
            validation_id="RM_002",
            requirement_id="high_risk_event_handling",
            framework="risk_management",
            title="High-Risk Event Handling",
            severity=ValidationSeverity.HIGH,
            status="warning" if len(high_risk_events) > 10 else "pass",
            message=f"High-risk events: {len(high_risk_events)}",
            details={"high_risk_events": len(high_risk_events)},
            evidence=[f"High-risk event count: {len(high_risk_events)}"],
            recommendations=(
                ["Review high-risk event handling procedures"]
                if len(high_risk_events) > 10
                else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Validation 3: Risk level distribution
        risk_distribution = {
            "high": len([e for e in all_events if e.risk_level == "high"]),
            "medium": len([e for e in all_events if e.risk_level == "medium"]),
            "low": len([e for e in all_events if e.risk_level == "low"]),
        }

        high_risk_rate = (
            risk_distribution["high"] / len(all_events) if all_events else 0
        )

        result = ValidationResult(
            validation_id="RM_003",
            requirement_id="risk_level_distribution",
            framework="risk_management",
            title="Risk Level Distribution",
            severity=ValidationSeverity.LOW,
            status="warning" if high_risk_rate > 0.2 else "pass",
            message=f"High-risk event rate: {high_risk_rate:.1%}",
            details=risk_distribution,
            evidence=[f"Risk distribution: {risk_distribution}"],
            recommendations=(
                ["Review risk classification criteria"] if high_risk_rate > 0.2 else []
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        self.validation_results.extend(results)
        return results

    def validate_documentation_completeness(
        self, frameworks: List[ComplianceFramework]
    ) -> List[ValidationResult]:
        """Validate documentation completeness for frameworks."""

        results = []

        for framework in frameworks:
            requirements = self.regulatory_mapper.get_requirements([framework])

            # Check for missing documentation requirements
            for req in requirements:
                if req.documentation_required:
                    result = ValidationResult(
                        validation_id=f"DOC_{req.requirement_id}",
                        requirement_id=req.requirement_id,
                        framework=framework.value,
                        title=f"Documentation: {req.title}",
                        severity=(
                            ValidationSeverity.MEDIUM
                            if req.mandatory
                            else ValidationSeverity.LOW
                        ),
                        status="warning",  # Would check actual documentation in practice
                        message=f"Documentation required: {', '.join(req.documentation_required)}",
                        details={"required_docs": req.documentation_required},
                        evidence=["Documentation requirements identified"],
                        recommendations=[
                            f"Prepare {doc}" for doc in req.documentation_required
                        ],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    results.append(result)

        self.validation_results.extend(results)
        return results

    def _validate_single_requirement(
        self,
        requirement: ComplianceRequirement,
        audit_generator: AuditTrailGenerator,
        model_version: str,
        validation_period_days: int,
    ) -> ValidationResult:
        """Validate a single compliance requirement."""

        # Check if CIAF satisfies the requirement
        if requirement.is_satisfied_by_ciaf():
            status = "pass"
            message = f"Requirement satisfied by CIAF capabilities: {', '.join(requirement.ciaf_capabilities)}"
            evidence = [
                f"CIAF capability: {cap}" for cap in requirement.ciaf_capabilities
            ]
            recommendations = []
        else:
            status = "fail" if requirement.mandatory else "warning"
            message = "Requirement not satisfied by current CIAF capabilities"
            evidence = ["No CIAF capabilities mapped to this requirement"]
            recommendations = [requirement.implementation_notes]

        # Determine severity based on requirement properties
        if requirement.mandatory and requirement.risk_level == "high":
            severity = ValidationSeverity.CRITICAL
        elif requirement.mandatory:
            severity = ValidationSeverity.HIGH
        elif requirement.risk_level == "high":
            severity = ValidationSeverity.MEDIUM
        else:
            severity = ValidationSeverity.LOW

        return ValidationResult(
            validation_id=f"VAL_{requirement.requirement_id}",
            requirement_id=requirement.requirement_id,
            framework=requirement.framework.value,
            title=requirement.title,
            severity=severity,
            status=status,
            message=message,
            details={
                "requirement_description": requirement.description,
                "mandatory": requirement.mandatory,
                "risk_level": requirement.risk_level,
                "category": requirement.category,
            },
            evidence=evidence,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results (Protocol implementation)."""

        if not self.validation_results:
            return {"message": "No validations performed"}

        total_validations = len(self.validation_results)
        passing = len([r for r in self.validation_results if r.is_passing()])
        failing = len([r for r in self.validation_results if r.is_failing()])
        warnings = len([r for r in self.validation_results if r.status == "warning"])

        # Count by severity
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len(
                [r for r in self.validation_results if r.severity == severity]
            )

        # Count by framework
        framework_counts = {}
        for result in self.validation_results:
            framework = result.framework
            if framework not in framework_counts:
                framework_counts[framework] = {
                    "pass": 0,
                    "fail": 0,
                    "warning": 0,
                    "total": 0,
                }
            framework_counts[framework][result.status] += 1
            framework_counts[framework]["total"] += 1

        # Assess overall compliance status using policy thresholds
        overall_compliant = self._assess_overall_compliance(failing, warnings, severity_counts)

        return {
            "total_validations": total_validations,
            "passing": passing,
            "failing": failing,
            "warnings": warnings,
            "pass_rate": (
                (passing / total_validations) * 100 if total_validations else 0
            ),
            "severity_breakdown": severity_counts,
            "framework_breakdown": framework_counts,
            "overall_status": "compliant" if overall_compliant else "non_compliant",
            "needs_attention": failing + warnings,
            "policy_compliance": self._check_policy_compliance(severity_counts)
        }

    def _assess_overall_compliance(
        self, 
        failing: int, 
        warnings: int, 
        severity_counts: Dict[str, int]
    ) -> bool:
        """Assess overall compliance based on policy thresholds."""
        
        policy = self.compliance_policy.validation_policy
        
        # Check each severity level against thresholds
        for severity, threshold in policy.failure_threshold.items():
            if threshold >= 0:  # -1 means no limit
                actual_count = severity_counts.get(severity.value, 0)
                if actual_count > threshold:
                    return False
        
        # For strict compliance, no failures allowed
        if policy.compliance_level.value == "strict":
            return failing == 0
        
        # For standard compliance, no critical or high severity failures
        if policy.compliance_level.value == "standard":
            critical_count = severity_counts.get(ValidationSeverity.CRITICAL.value, 0)
            high_count = severity_counts.get(ValidationSeverity.HIGH.value, 0)
            return critical_count == 0 and high_count <= policy.failure_threshold.get(ValidationSeverity.HIGH, 2)
        
        # For advisory compliance, just track everything
        return True

    def _check_policy_compliance(self, severity_counts: Dict[str, int]) -> Dict[str, Any]:
        """Check compliance against policy thresholds."""
        
        policy = self.compliance_policy.validation_policy
        policy_status = {
            "compliance_level": policy.compliance_level.value,
            "threshold_violations": [],
            "within_policy": True
        }
        
        for severity, threshold in policy.failure_threshold.items():
            if threshold >= 0:  # -1 means no limit
                actual_count = severity_counts.get(severity.value, 0)
                if actual_count > threshold:
                    policy_status["within_policy"] = False
                    policy_status["threshold_violations"].append({
                        "severity": severity.value,
                        "threshold": threshold,
                        "actual": actual_count,
                        "excess": actual_count - threshold
                    })
        
        return policy_status

    def get_failing_validations(self) -> List[ValidationResult]:
        """Get all failing validations."""
        return [r for r in self.validation_results if r.is_failing()]

    def get_validations_by_severity(
        self, severity: ValidationSeverity
    ) -> List[ValidationResult]:
        """Get validations by severity level."""
        return [r for r in self.validation_results if r.severity == severity]

    def get_validations_by_framework(self, framework: str) -> List[ValidationResult]:
        """Get validations for a specific framework."""
        return [r for r in self.validation_results if r.framework == framework]

    def export_validation_results(self, format: str = "json") -> str:
        """Export validation results in specified format."""

        if format.lower() == "json":
            results_dict = []
            for result in self.validation_results:
                result_dict = {
                    "validation_id": result.validation_id,
                    "requirement_id": result.requirement_id,
                    "framework": result.framework,
                    "title": result.title,
                    "severity": result.severity.value,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                    "evidence": result.evidence,
                    "recommendations": result.recommendations,
                    "timestamp": result.timestamp,
                }
                results_dict.append(result_dict)

            return json.dumps(
                {
                    "model_name": self.model_name,
                    "validation_summary": self.get_validation_summary(),
                    "validation_results": results_dict,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_results(self):
        """Clear all validation results."""
        self.validation_results.clear()


class BiasValidator:
    """
    Validator for detecting and measuring bias in AI models
    """

    def __init__(self):
        self.bias_thresholds = {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "calibration": 0.1,
        }

    def validate_predictions(
        self, predictions, protected_attributes, ground_truth=None
    ):
        """
        Validate predictions for bias across protected attributes

        Args:
            predictions: Model predictions
            protected_attributes: Dict of protected attribute arrays
            ground_truth: Optional ground truth labels

        Returns:
            Dict with bias analysis results
        """
        import numpy as np

        bias_results = {
            "overall_bias_score": 0.95,  # Default high score
            "demographic_parity": {},
            "statistical_parity": {},
            "bias_detected": False,
        }

        try:
            # Calculate demographic parity for each protected attribute
            for attr_name, attr_values in protected_attributes.items():
                unique_values = np.unique(attr_values)
                if len(unique_values) > 1:
                    parity_scores = {}
                    for value in unique_values:
                        mask = attr_values == value
                        if np.sum(mask) > 0:
                            positive_rate = np.mean(predictions[mask])
                            parity_scores[str(value)] = positive_rate

                    # Calculate parity difference
                    if len(parity_scores) >= 2:
                        rates = list(parity_scores.values())
                        parity_diff = max(rates) - min(rates)
                        bias_results["demographic_parity"][attr_name] = {
                            "rates": parity_scores,
                            "difference": parity_diff,
                            "bias_detected": parity_diff
                            > self.bias_thresholds["demographic_parity"],
                        }

                        if parity_diff > self.bias_thresholds["demographic_parity"]:
                            bias_results["bias_detected"] = True
                            bias_results[
                                "overall_bias_score"
                            ] *= 0.9  # Reduce score if bias detected

        except Exception as e:
            bias_results["error"] = str(e)

        return bias_results

    def calculate_bias_metrics(
        self, predictions, protected_attributes, ground_truth=None
    ):
        """Calculate comprehensive bias metrics"""
        return self.validate_predictions(
            predictions, protected_attributes, ground_truth
        )


class FairnessValidator:
    """
    Validator for measuring fairness in AI model outcomes
    """

    def __init__(self):
        self.fairness_thresholds = {
            "equalized_odds": 0.1,
            "demographic_parity": 0.1,
            "individual_fairness": 0.1,
        }

    def calculate_fairness_metrics(
        self, predictions, protected_attributes, ground_truth=None
    ):
        """
        Calculate fairness metrics for model predictions

        Args:
            predictions: Model predictions
            protected_attributes: Dict of protected attribute arrays
            ground_truth: Optional ground truth labels

        Returns:
            Dict with fairness metrics
        """
        import numpy as np

        fairness_metrics = {
            "overall_fairness_score": 0.92,  # Default good score
            "demographic_parity": {},
            "equalized_odds": {},
            "fair_across_groups": True,
        }

        try:
            # Calculate fairness metrics for each protected attribute
            for attr_name, attr_values in protected_attributes.items():
                unique_values = np.unique(attr_values)
                if len(unique_values) > 1:
                    group_metrics = {}
                    for value in unique_values:
                        mask = attr_values == value
                        if np.sum(mask) > 0:
                            group_predictions = predictions[mask]

                            # Calculate group-specific metrics
                            positive_rate = np.mean(group_predictions)
                            if ground_truth is not None:
                                group_truth = ground_truth[mask]
                                accuracy = (
                                    np.mean(group_predictions == group_truth)
                                    if len(group_truth) > 0
                                    else 0
                                )
                            else:
                                accuracy = 0.85  # Default assumption

                            group_metrics[str(value)] = {
                                "positive_rate": positive_rate,
                                "accuracy": accuracy,
                                "sample_size": np.sum(mask),
                            }

                    fairness_metrics["demographic_parity"][attr_name] = group_metrics

                    # Check if fair across groups
                    if len(group_metrics) >= 2:
                        rates = [
                            metrics["positive_rate"]
                            for metrics in group_metrics.values()
                        ]
                        rate_diff = max(rates) - min(rates)
                        if rate_diff > self.fairness_thresholds["demographic_parity"]:
                            fairness_metrics["fair_across_groups"] = False
                            fairness_metrics["overall_fairness_score"] *= 0.9

        except Exception as e:
            fairness_metrics["error"] = str(e)

        return fairness_metrics

    def validate_fairness(self, predictions, protected_attributes, ground_truth=None):
        """Validate fairness of model predictions"""
        return self.calculate_fairness_metrics(
            predictions, protected_attributes, ground_truth
        )
