"""
Policy enforcement and risk assessment for CIAF core operations.

Implements PolicyEnforcer for high-risk domain detection, compliance
checking, and audit policy enforcement throughout the CIAF pipeline.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .canonicalization import Policy
from .enums import RecordType


class RiskLevel(str, Enum):
    """Risk levels for policy enforcement."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceResult(str, Enum):
    """Compliance check results."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    rule_id: str
    severity: RiskLevel
    description: str
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class RiskAssessment:
    """Risk assessment result for an operation."""
    risk_level: RiskLevel
    risk_factors: List[str]
    violations: List[PolicyViolation]
    compliance_result: ComplianceResult
    recommendations: List[str]


class PolicyRule:
    """Base class for policy rules."""
    
    def __init__(self, rule_id: str, description: str, severity: RiskLevel):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity
    
    def evaluate(self, metadata: Dict[str, Any], policy: Policy) -> Optional[PolicyViolation]:
        """Evaluate rule against metadata and policy."""
        raise NotImplementedError


class HighRiskDomainRule(PolicyRule):
    """Rule to detect high-risk domains."""
    
    def __init__(self):
        super().__init__(
            "HIGH_RISK_DOMAIN",
            "Detects operations in high-risk domains",
            RiskLevel.HIGH
        )
    
    def evaluate(self, metadata: Dict[str, Any], policy: Policy) -> Optional[PolicyViolation]:
        if policy.is_high_risk():
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                description=f"Operation involves high-risk domains: {policy.domain_labels}",
                metadata={
                    "domains": policy.domain_labels,
                    "high_risk_domains": policy.high_risk_domains
                },
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        return None


class PiiDetectionRule(PolicyRule):
    """Rule to detect potential PII in metadata."""
    
    def __init__(self):
        super().__init__(
            "PII_DETECTION",
            "Detects potential personally identifiable information",
            RiskLevel.MEDIUM
        )
        
        # Common PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def evaluate(self, metadata: Dict[str, Any], policy: Policy) -> Optional[PolicyViolation]:
        pii_found = []
        
        # Check all string values in metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                for pii_type, pattern in self.pii_patterns.items():
                    if re.search(pattern, value, re.IGNORECASE):
                        pii_found.append(f"{pii_type} in {key}")
        
        if pii_found:
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                description=f"Potential PII detected: {', '.join(pii_found)}",
                metadata={
                    "pii_types": pii_found,
                    "affected_fields": list(metadata.keys())
                },
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        return None


class TimestampValidationRule(PolicyRule):
    """Rule to validate timestamp formats and recency."""
    
    def __init__(self):
        super().__init__(
            "TIMESTAMP_VALIDATION",
            "Validates timestamp formats and recency",
            RiskLevel.LOW
        )
    
    def evaluate(self, metadata: Dict[str, Any], policy: Policy) -> Optional[PolicyViolation]:
        timestamp_str = metadata.get('timestamp')
        if not timestamp_str:
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=RiskLevel.MEDIUM,
                description="Missing required timestamp field",
                metadata={"missing_field": "timestamp"},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        try:
            # Validate ISO format
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Check if timestamp is in the future (suspicious)
            now = datetime.now(timezone.utc)
            if timestamp > now:
                return PolicyViolation(
                    rule_id=self.rule_id,
                    severity=RiskLevel.MEDIUM,
                    description="Timestamp is in the future",
                    metadata={
                        "timestamp": timestamp_str,
                        "current_time": now.isoformat()
                    },
                    timestamp=now.isoformat()
                )
        
        except (ValueError, TypeError):
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=RiskLevel.MEDIUM,
                description="Invalid timestamp format",
                metadata={"invalid_timestamp": timestamp_str},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        return None


class RequiredFieldsRule(PolicyRule):
    """Rule to enforce required fields based on record type."""
    
    def __init__(self):
        super().__init__(
            "REQUIRED_FIELDS",
            "Validates presence of required fields",
            RiskLevel.HIGH
        )
        
        # Import here to avoid circular dependency
        from .canonicalization import REQUIRED_FIELDS
        self.required_fields = REQUIRED_FIELDS
    
    def evaluate(self, metadata: Dict[str, Any], policy: Policy) -> Optional[PolicyViolation]:
        record_type_str = metadata.get('record_type')
        if not record_type_str:
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                description="Missing record_type field",
                metadata={"missing_field": "record_type"},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        try:
            record_type = RecordType(record_type_str)
        except ValueError:
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                description=f"Invalid record_type: {record_type_str}",
                metadata={"invalid_record_type": record_type_str},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        required = self.required_fields.get(record_type, [])
        missing = [field for field in required if field not in metadata]
        
        if missing:
            return PolicyViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                description=f"Missing required fields: {missing}",
                metadata={
                    "missing_fields": missing,
                    "record_type": record_type_str
                },
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        return None


class PolicyEnforcer:
    """
    Central policy enforcement engine for CIAF operations.
    
    Evaluates operations against policy rules and provides risk assessment
    and compliance checking capabilities.
    """
    
    def __init__(self, additional_rules: Optional[List[PolicyRule]] = None):
        """
        Initialize policy enforcer with default and additional rules.
        
        Args:
            additional_rules: Optional list of additional policy rules
        """
        # Default rule set
        self.rules: List[PolicyRule] = [
            HighRiskDomainRule(),
            PiiDetectionRule(),
            TimestampValidationRule(),
            RequiredFieldsRule()
        ]
        
        # Add any additional rules
        if additional_rules:
            self.rules.extend(additional_rules)
        
        # Track enforcement statistics
        self.enforcement_stats = {
            'total_assessments': 0,
            'violations_found': 0,
            'high_risk_operations': 0,
            'non_compliant_operations': 0
        }
    
    def assess_risk(self, metadata: Dict[str, Any], policy: Policy) -> RiskAssessment:
        """
        Perform comprehensive risk assessment on an operation.
        
        Args:
            metadata: Operation metadata to assess
            policy: Policy configuration
            
        Returns:
            RiskAssessment with violations and recommendations
        """
        self.enforcement_stats['total_assessments'] += 1
        
        violations = []
        risk_factors = []
        
        # Evaluate all rules
        for rule in self.rules:
            violation = rule.evaluate(metadata, policy)
            if violation:
                violations.append(violation)
                risk_factors.append(violation.description)
        
        # Determine overall risk level
        if violations:
            self.enforcement_stats['violations_found'] += 1
            max_severity = max(v.severity for v in violations)
            risk_level = max_severity
        else:
            risk_level = RiskLevel.LOW
        
        # Determine compliance result
        critical_violations = [v for v in violations if v.severity == RiskLevel.CRITICAL]
        high_violations = [v for v in violations if v.severity == RiskLevel.HIGH]
        
        if critical_violations:
            compliance_result = ComplianceResult.NON_COMPLIANT
            self.enforcement_stats['non_compliant_operations'] += 1
        elif high_violations:
            compliance_result = ComplianceResult.REQUIRES_REVIEW
        else:
            compliance_result = ComplianceResult.COMPLIANT
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self.enforcement_stats['high_risk_operations'] += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, policy)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_factors=risk_factors,
            violations=violations,
            compliance_result=compliance_result,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, violations: List[PolicyViolation], policy: Policy) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        violation_types = {v.rule_id for v in violations}
        
        if 'HIGH_RISK_DOMAIN' in violation_types:
            recommendations.append("Consider additional monitoring and approval processes for high-risk domains")
            recommendations.append("Implement enhanced logging and audit trails")
        
        if 'PII_DETECTION' in violation_types:
            recommendations.append("Review and potentially redact or encrypt detected PII")
            recommendations.append("Ensure GDPR/privacy compliance measures are in place")
        
        if 'TIMESTAMP_VALIDATION' in violation_types:
            recommendations.append("Verify system clock synchronization")
            recommendations.append("Implement timestamp validation at data ingestion")
        
        if 'REQUIRED_FIELDS' in violation_types:
            recommendations.append("Implement input validation to ensure required fields")
            recommendations.append("Review data collection processes")
        
        if policy.external_timestamping:
            recommendations.append("Ensure external timestamping service is operational")
        
        return recommendations
    
    def is_operation_allowed(self, metadata: Dict[str, Any], policy: Policy) -> bool:
        """
        Quick check if operation should be allowed based on policy.
        
        Args:
            metadata: Operation metadata
            policy: Policy configuration
            
        Returns:
            True if operation is allowed, False if blocked
        """
        assessment = self.assess_risk(metadata, policy)
        return assessment.compliance_result != ComplianceResult.NON_COMPLIANT
    
    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        return self.enforcement_stats.copy()
    
    def add_rule(self, rule: PolicyRule):
        """Add a new policy rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a policy rule by ID."""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]


# Factory functions for common policy configurations

def create_healthcare_policy_enforcer() -> PolicyEnforcer:
    """Create policy enforcer with healthcare-specific rules."""
    additional_rules = [
        # Add healthcare-specific rules here
    ]
    return PolicyEnforcer(additional_rules)


def create_financial_policy_enforcer() -> PolicyEnforcer:
    """Create policy enforcer with financial services-specific rules."""
    additional_rules = [
        # Add financial-specific rules here
    ]
    return PolicyEnforcer(additional_rules)


def create_gdpr_policy_enforcer() -> PolicyEnforcer:
    """Create policy enforcer with GDPR-specific rules."""
    additional_rules = [
        # Add GDPR-specific rules here
    ]
    return PolicyEnforcer(additional_rules)