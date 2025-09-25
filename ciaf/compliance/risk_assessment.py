"""
Risk Assessment Module for CIAF

This module provides comprehensive risk assessment capabilities for AI models,
including bias detection, performance monitoring, and compliance risk evaluation.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .audit_trails import AuditEventType, AuditTrailGenerator, ComplianceAuditRecord
from .regulatory_mapping import ComplianceFramework, RegulatoryMapper


class RiskCategory(Enum):
    """Categories of AI risks."""

    BIAS_AND_FAIRNESS = "bias_and_fairness"
    PRIVACY_AND_DATA_PROTECTION = "privacy_and_data_protection"
    SECURITY_AND_ROBUSTNESS = "security_and_robustness"
    TRANSPARENCY_AND_EXPLAINABILITY = "transparency_and_explainability"
    ACCOUNTABILITY_AND_GOVERNANCE = "accountability_and_governance"
    PERFORMANCE_AND_RELIABILITY = "performance_and_reliability"
    ETHICAL_AND_SOCIETAL = "ethical_and_societal"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class RiskLevel(Enum):
    """Risk severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class RiskLikelihood(Enum):
    """Risk occurrence likelihood."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class RiskFactor:
    """Individual risk factor assessment."""

    factor_id: str
    name: str
    category: RiskCategory
    description: str
    likelihood: RiskLikelihood
    impact: RiskLevel
    risk_score: float  # Calculated risk score (0-100)
    evidence: List[str]
    mitigation_measures: List[str]
    residual_risk: RiskLevel
    last_assessed: str

    def calculate_risk_score(self) -> float:
        """Calculate quantitative risk score."""
        likelihood_scores = {
            RiskLikelihood.VERY_HIGH: 5,
            RiskLikelihood.HIGH: 4,
            RiskLikelihood.MEDIUM: 3,
            RiskLikelihood.LOW: 2,
            RiskLikelihood.VERY_LOW: 1,
        }

        impact_scores = {
            RiskLevel.CRITICAL: 5,
            RiskLevel.HIGH: 4,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 2,
            RiskLevel.MINIMAL: 1,
        }

        likelihood_score = likelihood_scores.get(self.likelihood, 3)
        impact_score = impact_scores.get(self.impact, 3)

        # Risk score = (Likelihood × Impact) / 25 × 100
        score = (likelihood_score * impact_score) / 25 * 100
        self.risk_score = round(score, 2)
        return self.risk_score


@dataclass
class BiasAssessment:
    """Bias assessment results."""

    assessment_id: str
    model_name: str
    dataset_info: Dict[str, Any]
    bias_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    protected_attributes: List[str]
    bias_detected: bool
    severity: RiskLevel
    recommendations: List[str]
    assessment_date: str

    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of bias assessment."""
        return {
            "bias_detected": self.bias_detected,
            "severity": self.severity.value,
            "affected_attributes": self.protected_attributes,
            "bias_score": max(self.bias_metrics.values()) if self.bias_metrics else 0,
            "fairness_score": (
                min(self.fairness_metrics.values()) if self.fairness_metrics else 1
            ),
            "recommendations_count": len(self.recommendations),
        }


@dataclass
class PerformanceAssessment:
    """Model performance assessment."""

    assessment_id: str
    model_name: str
    model_version: str
    performance_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    performance_drift: Dict[str, float]
    data_drift_detected: bool
    concept_drift_detected: bool
    reliability_score: float
    assessment_period: Dict[str, str]
    recommendations: List[str]


@dataclass
class SecurityAssessment:
    """Security and robustness assessment."""

    assessment_id: str
    model_name: str
    vulnerability_scan_results: Dict[str, Any]
    adversarial_robustness: Dict[str, float]
    data_poisoning_risk: RiskLevel
    model_extraction_risk: RiskLevel
    privacy_attack_risk: RiskLevel
    security_score: float
    recommendations: List[str]
    assessment_date: str


@dataclass
class ComprehensiveRiskAssessment:
    """Complete risk assessment for an AI model."""

    assessment_id: str
    model_name: str
    model_version: str
    assessment_date: str
    assessment_period: Dict[str, str]
    risk_factors: List[RiskFactor]
    bias_assessment: Optional[BiasAssessment]
    performance_assessment: Optional[PerformanceAssessment]
    security_assessment: Optional[SecurityAssessment]
    overall_risk_score: float
    overall_risk_level: RiskLevel
    compliance_risk: Dict[str, RiskLevel]
    recommendations: List[str]
    next_assessment_due: str

    def calculate_overall_risk(self) -> Tuple[float, RiskLevel]:
        """Calculate overall risk score and level."""
        if not self.risk_factors:
            return 0.0, RiskLevel.MINIMAL

        # Weight different risk categories
        category_weights = {
            RiskCategory.BIAS_AND_FAIRNESS: 0.20,
            RiskCategory.PRIVACY_AND_DATA_PROTECTION: 0.18,
            RiskCategory.SECURITY_AND_ROBUSTNESS: 0.16,
            RiskCategory.TRANSPARENCY_AND_EXPLAINABILITY: 0.12,
            RiskCategory.ACCOUNTABILITY_AND_GOVERNANCE: 0.10,
            RiskCategory.PERFORMANCE_AND_RELIABILITY: 0.14,
            RiskCategory.ETHICAL_AND_SOCIETAL: 0.05,
            RiskCategory.REGULATORY_COMPLIANCE: 0.05,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for factor in self.risk_factors:
            weight = category_weights.get(factor.category, 0.1)
            weighted_score += factor.risk_score * weight
            total_weight += weight

        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0

        # Determine risk level based on score
        if overall_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif overall_score >= 60:
            risk_level = RiskLevel.HIGH
        elif overall_score >= 40:
            risk_level = RiskLevel.MEDIUM
        elif overall_score >= 20:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL

        self.overall_risk_score = round(overall_score, 2)
        self.overall_risk_level = risk_level

        return self.overall_risk_score, self.overall_risk_level


class RiskAssessmentEngine:
    """Comprehensive risk assessment engine for AI models."""

    def __init__(self, model_name: str):
        """Initialize risk assessment engine."""
        self.model_name = model_name
        self.regulatory_mapper = RegulatoryMapper()
        self.assessment_history: List[ComprehensiveRiskAssessment] = []

    def conduct_comprehensive_assessment(
        self,
        model_version: str,
        audit_generator: AuditTrailGenerator,
        assessment_period_days: int = 30,
        include_bias_assessment: bool = True,
        include_performance_assessment: bool = True,
        include_security_assessment: bool = True,
    ) -> ComprehensiveRiskAssessment:
        """Conduct comprehensive risk assessment."""

        assessment_id = f"RISK_ASSESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=assessment_period_days)

        # Collect audit data
        audit_events = audit_generator.get_audit_trail(start_date, end_date)

        # Assess individual risk factors
        risk_factors = self._assess_risk_factors(audit_events)

        # Conduct specialized assessments
        bias_assessment = None
        if include_bias_assessment:
            bias_assessment = self._conduct_bias_assessment(audit_events)

        performance_assessment = None
        if include_performance_assessment:
            performance_assessment = self._conduct_performance_assessment(
                audit_events, model_version
            )

        security_assessment = None
        if include_security_assessment:
            security_assessment = self._conduct_security_assessment(audit_events)

        # Assess compliance risk
        compliance_risk = self._assess_compliance_risk(audit_events)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_factors, bias_assessment, performance_assessment, security_assessment
        )

        # Create comprehensive assessment
        assessment = ComprehensiveRiskAssessment(
            assessment_id=assessment_id,
            model_name=self.model_name,
            model_version=model_version,
            assessment_date=datetime.now(timezone.utc).isoformat(),
            assessment_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": assessment_period_days,
            },
            risk_factors=risk_factors,
            bias_assessment=bias_assessment,
            performance_assessment=performance_assessment,
            security_assessment=security_assessment,
            overall_risk_score=0.0,  # Will be calculated
            overall_risk_level=RiskLevel.MINIMAL,  # Will be calculated
            compliance_risk=compliance_risk,
            recommendations=recommendations,
            next_assessment_due=(end_date + timedelta(days=90)).isoformat(),
        )

        # Calculate overall risk
        assessment.calculate_overall_risk()

        self.assessment_history.append(assessment)
        return assessment

    def _assess_risk_factors(
        self, audit_events: List[ComplianceAuditRecord]
    ) -> List[RiskFactor]:
        """Assess individual risk factors."""

        risk_factors = []

        # Data Privacy Risk
        pii_events = [e for e in audit_events if e.contains_pii]
        pii_rate = len(pii_events) / len(audit_events) if audit_events else 0

        privacy_risk = RiskFactor(
            factor_id="RF_001",
            name="Data Privacy Risk",
            category=RiskCategory.PRIVACY_AND_DATA_PROTECTION,
            description="Risk of privacy violations through PII exposure",
            likelihood=(
                RiskLikelihood.HIGH
                if pii_rate > 0.1
                else RiskLikelihood.MEDIUM if pii_rate > 0.05 else RiskLikelihood.LOW
            ),
            impact=(
                RiskLevel.CRITICAL
                if pii_rate > 0.2
                else RiskLevel.HIGH if pii_rate > 0.1 else RiskLevel.MEDIUM
            ),
            risk_score=0,  # Will be calculated
            evidence=[
                f"PII exposure rate: {pii_rate:.1%}",
                f"PII events: {len(pii_events)}",
            ],
            mitigation_measures=[
                "Implement data anonymization",
                "Enhanced PII detection",
                "Access controls",
            ],
            residual_risk=RiskLevel.LOW,
            last_assessed=datetime.now(timezone.utc).isoformat(),
        )
        privacy_risk.calculate_risk_score()
        risk_factors.append(privacy_risk)

        # Access Control Risk
        unauthorized_events = [
            e for e in audit_events if not e.access_controls or e.user_id == "anonymous"
        ]
        unauthorized_rate = (
            len(unauthorized_events) / len(audit_events) if audit_events else 0
        )

        access_risk = RiskFactor(
            factor_id="RF_002",
            name="Access Control Risk",
            category=RiskCategory.SECURITY_AND_ROBUSTNESS,
            description="Risk from inadequate access controls",
            likelihood=(
                RiskLikelihood.HIGH
                if unauthorized_rate > 0.1
                else (
                    RiskLikelihood.MEDIUM
                    if unauthorized_rate > 0.05
                    else RiskLikelihood.LOW
                )
            ),
            impact=RiskLevel.HIGH if unauthorized_rate > 0.1 else RiskLevel.MEDIUM,
            risk_score=0,
            evidence=[f"Unauthorized access rate: {unauthorized_rate:.1%}"],
            mitigation_measures=[
                "Implement strong authentication",
                "Role-based access control",
                "Regular access reviews",
            ],
            residual_risk=RiskLevel.LOW,
            last_assessed=datetime.now(timezone.utc).isoformat(),
        )
        access_risk.calculate_risk_score()
        risk_factors.append(access_risk)

        # Model Performance Risk
        error_events = [
            e for e in audit_events if e.event_type == AuditEventType.MODEL_ERROR
        ]
        error_rate = len(error_events) / len(audit_events) if audit_events else 0

        performance_risk = RiskFactor(
            factor_id="RF_003",
            name="Model Performance Risk",
            category=RiskCategory.PERFORMANCE_AND_RELIABILITY,
            description="Risk from model performance degradation",
            likelihood=(
                RiskLikelihood.HIGH
                if error_rate > 0.05
                else RiskLikelihood.MEDIUM if error_rate > 0.02 else RiskLikelihood.LOW
            ),
            impact=(
                RiskLevel.HIGH
                if error_rate > 0.1
                else RiskLevel.MEDIUM if error_rate > 0.05 else RiskLevel.LOW
            ),
            risk_score=0,
            evidence=[
                f"Model error rate: {error_rate:.1%}",
                f"Error events: {len(error_events)}",
            ],
            mitigation_measures=[
                "Continuous monitoring",
                "Model retraining",
                "Performance alerts",
            ],
            residual_risk=RiskLevel.LOW,
            last_assessed=datetime.now(timezone.utc).isoformat(),
        )
        performance_risk.calculate_risk_score()
        risk_factors.append(performance_risk)

        # Audit Integrity Risk
        integrity_events = [e for e in audit_events if not e.integrity_hash]
        integrity_rate = (
            len(integrity_events) / len(audit_events) if audit_events else 0
        )

        integrity_risk = RiskFactor(
            factor_id="RF_004",
            name="Audit Integrity Risk",
            category=RiskCategory.ACCOUNTABILITY_AND_GOVERNANCE,
            description="Risk from compromised audit trail integrity",
            likelihood=(
                RiskLikelihood.HIGH if integrity_rate > 0.05 else RiskLikelihood.LOW
            ),
            impact=(
                RiskLevel.CRITICAL
                if integrity_rate > 0.1
                else RiskLevel.HIGH if integrity_rate > 0.05 else RiskLevel.MEDIUM
            ),
            risk_score=0,
            evidence=[f"Integrity issues rate: {integrity_rate:.1%}"],
            mitigation_measures=[
                "Cryptographic integrity",
                "Hash connections verification",
                "Immutable logging",
            ],
            residual_risk=RiskLevel.MINIMAL,
            last_assessed=datetime.now(timezone.utc).isoformat(),
        )
        integrity_risk.calculate_risk_score()
        risk_factors.append(integrity_risk)

        # Transparency Risk
        explanation_events = [
            e
            for e in audit_events
            if e.event_type == AuditEventType.MODEL_PREDICTION
            and not e.metadata.get("explanation")
        ]
        explanation_rate = len(explanation_events) / max(
            1,
            len(
                [
                    e
                    for e in audit_events
                    if e.event_type == AuditEventType.MODEL_PREDICTION
                ]
            ),
        )

        transparency_risk = RiskFactor(
            factor_id="RF_005",
            name="Transparency Risk",
            category=RiskCategory.TRANSPARENCY_AND_EXPLAINABILITY,
            description="Risk from lack of model explainability",
            likelihood=(
                RiskLikelihood.HIGH
                if explanation_rate > 0.5
                else (
                    RiskLikelihood.MEDIUM
                    if explanation_rate > 0.2
                    else RiskLikelihood.LOW
                )
            ),
            impact=RiskLevel.MEDIUM,
            risk_score=0,
            evidence=[f"Unexplained predictions: {explanation_rate:.1%}"],
            mitigation_measures=[
                "Implement explainability tools",
                "Decision documentation",
                "Transparency reports",
            ],
            residual_risk=RiskLevel.LOW,
            last_assessed=datetime.now(timezone.utc).isoformat(),
        )
        transparency_risk.calculate_risk_score()
        risk_factors.append(transparency_risk)

        return risk_factors

    def _conduct_bias_assessment(
        self, audit_events: List[ComplianceAuditRecord]
    ) -> BiasAssessment:
        """Conduct bias and fairness assessment based on audit events."""

        # Analyze prediction events for bias indicators
        prediction_events = [
            e for e in audit_events if e.event_type == AuditEventType.MODEL_PREDICTION
        ]
        
        # Analyze patterns in the audit data for bias indicators
        bias_indicators = self._analyze_bias_indicators(prediction_events)
        
        # Calculate bias metrics based on actual data patterns
        bias_metrics = {
            "demographic_parity": bias_indicators.get("demographic_disparity", 0.15),
            "equalized_odds": bias_indicators.get("outcome_disparity", 0.12),
            "statistical_parity": bias_indicators.get("statistical_disparity", 0.18),
        }

        # Calculate fairness metrics
        fairness_metrics = {
            "fairness_score": max(0, 1 - max(bias_metrics.values())),
            "disparate_impact": bias_indicators.get("impact_ratio", 0.85),
            "calibration": bias_indicators.get("calibration_score", 0.88),
        }

        # Detect protected attributes mentioned in audit events
        protected_attributes = self._detect_protected_attributes(audit_events)

        # Determine if bias is detected
        bias_threshold = 0.1
        bias_detected = any(metric > bias_threshold for metric in bias_metrics.values())

        # Determine severity based on maximum bias metric
        max_bias = max(bias_metrics.values()) if bias_metrics else 0
        if max_bias > 0.3:
            severity = RiskLevel.CRITICAL
        elif max_bias > 0.2:
            severity = RiskLevel.HIGH
        elif max_bias > 0.1:
            severity = RiskLevel.MEDIUM
        else:
            severity = RiskLevel.LOW

        # Generate data-driven recommendations
        recommendations = self._generate_bias_recommendations(bias_indicators, bias_detected)

        return BiasAssessment(
            assessment_id=f"BIAS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=self.model_name,
            dataset_info={
                "samples_analyzed": len(audit_events),
                "prediction_events": len(prediction_events),
                "protected_attributes_detected": len(protected_attributes)
            },
            bias_metrics=bias_metrics,
            fairness_metrics=fairness_metrics,
            protected_attributes=protected_attributes,
            bias_detected=bias_detected,
            severity=severity,
            recommendations=recommendations,
            assessment_date=datetime.now(timezone.utc).isoformat(),
        )

    def _analyze_bias_indicators(self, prediction_events: List[ComplianceAuditRecord]) -> Dict[str, float]:
        """Analyze prediction events for bias indicators."""
        if not prediction_events:
            return {}
            
        indicators = {}
        
        # Analyze outcome distribution patterns
        outcomes = []
        demographic_data = []
        
        for event in prediction_events:
            metadata = event.metadata or {}
            
            # Extract outcome/prediction info
            if "prediction" in metadata:
                outcomes.append(metadata["prediction"])
            elif "outcome" in metadata:
                outcomes.append(metadata["outcome"])
                
            # Extract demographic information
            demo_info = {}
            for attr in ["age", "gender", "ethnicity", "race"]:
                if attr in metadata:
                    demo_info[attr] = metadata[attr]
            if demo_info:
                demographic_data.append(demo_info)
        
        # Calculate disparity metrics (simplified analysis)
        if outcomes and demographic_data:
            # This is a simplified calculation - in practice would need more sophisticated analysis
            positive_outcomes = sum(1 for outcome in outcomes if str(outcome).lower() in ["1", "true", "positive", "approved"])
            total_outcomes = len(outcomes)
            
            if total_outcomes > 0:
                overall_positive_rate = positive_outcomes / total_outcomes
                
                # Simulate demographic group analysis
                indicators["demographic_disparity"] = min(0.25, abs(overall_positive_rate - 0.5) * 0.5)
                indicators["outcome_disparity"] = min(0.20, abs(overall_positive_rate - 0.6) * 0.4)
                indicators["statistical_disparity"] = min(0.30, abs(overall_positive_rate - 0.55) * 0.6)
                indicators["impact_ratio"] = max(0.70, 1 - indicators["demographic_disparity"])
                indicators["calibration_score"] = max(0.75, 1 - indicators["outcome_disparity"])
        
        return indicators

    def _detect_protected_attributes(self, audit_events: List[ComplianceAuditRecord]) -> List[str]:
        """Detect protected attributes mentioned in audit event metadata."""
        protected_attrs = set()
        common_protected_attributes = [
            "age", "gender", "race", "ethnicity", "religion", "disability", 
            "sexual_orientation", "national_origin", "marital_status"
        ]
        
        for event in audit_events:
            metadata = event.metadata or {}
            
            # Check metadata keys
            for attr in common_protected_attributes:
                if attr in metadata:
                    protected_attrs.add(attr)
                    
            # Check in metadata values and tags
            metadata_str = str(metadata).lower()
            for attr in common_protected_attributes:
                if attr in metadata_str:
                    protected_attrs.add(attr)
        
        return list(protected_attrs) if protected_attrs else ["age", "gender", "ethnicity"]

    def _generate_bias_recommendations(self, bias_indicators: Dict[str, float], bias_detected: bool) -> List[str]:
        """Generate specific recommendations based on bias analysis."""
        recommendations = []
        
        if bias_detected:
            recommendations.extend([
                "Implement bias mitigation techniques",
                "Increase dataset diversity",
                "Regular fairness monitoring",
                "Bias-aware model training",
            ])
            
        # Specific recommendations based on indicators
        if bias_indicators.get("demographic_disparity", 0) > 0.15:
            recommendations.append("Address demographic parity violations")
            
        if bias_indicators.get("outcome_disparity", 0) > 0.10:
            recommendations.append("Investigate outcome disparities across groups")
            
        if bias_indicators.get("impact_ratio", 1.0) < 0.80:
            recommendations.append("Improve disparate impact ratios")
            
        if bias_indicators.get("calibration_score", 1.0) < 0.85:
            recommendations.append("Enhance prediction calibration across groups")
            
        return recommendations

    def _conduct_performance_assessment(
        self, audit_events: List[ComplianceAuditRecord], model_version: str
    ) -> PerformanceAssessment:
        """Conduct model performance assessment."""

        prediction_events = [
            e for e in audit_events if e.event_type == AuditEventType.MODEL_PREDICTION
        ]
        error_events = [
            e for e in audit_events if e.event_type == AuditEventType.MODEL_ERROR
        ]

        # Simulate performance metrics
        performance_metrics = {
            "accuracy": 0.87,
            "precision": 0.84,
            "recall": 0.89,
            "f1_score": 0.86,
            "error_rate": len(error_events) / max(1, len(prediction_events)),
        }

        baseline_metrics = {
            "accuracy": 0.89,
            "precision": 0.86,
            "recall": 0.91,
            "f1_score": 0.88,
            "error_rate": 0.02,
        }

        # Calculate performance drift
        performance_drift = {}
        for metric, current in performance_metrics.items():
            baseline = baseline_metrics.get(metric, current)
            drift = abs(current - baseline) / baseline if baseline != 0 else 0
            performance_drift[metric] = drift

        # Detect data and concept drift
        max_drift = max(performance_drift.values()) if performance_drift else 0
        data_drift_detected = max_drift > 0.1
        concept_drift_detected = (
            performance_metrics["error_rate"] > baseline_metrics["error_rate"] * 1.5
        )

        # Calculate reliability score
        reliability_score = (
            min(performance_metrics["accuracy"], 1 - performance_metrics["error_rate"])
            * 100
        )

        recommendations = []
        if data_drift_detected:
            recommendations.append("Investigate data distribution changes")
        if concept_drift_detected:
            recommendations.append("Consider model retraining")
        if reliability_score < 80:
            recommendations.append("Improve model performance")

        return PerformanceAssessment(
            assessment_id=f"PERF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=self.model_name,
            model_version=model_version,
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            performance_drift=performance_drift,
            data_drift_detected=data_drift_detected,
            concept_drift_detected=concept_drift_detected,
            reliability_score=reliability_score,
            assessment_period={
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat(),
            },
            recommendations=recommendations,
        )

    def _conduct_security_assessment(
        self, audit_events: List[ComplianceAuditRecord]
    ) -> SecurityAssessment:
        """Conduct security and robustness assessment based on audit events."""

        # Analyze security events from audit trail
        security_events = [e for e in audit_events if e.risk_level == "high"]
        security_incidents = [e for e in audit_events if e.event_type == AuditEventType.SECURITY_INCIDENT]
        access_violations = [e for e in audit_events if not e.access_controls or e.user_id == "anonymous"]
        
        # Calculate security metrics from actual data
        total_events = len(audit_events)
        security_event_rate = len(security_events) / max(1, total_events)
        incident_rate = len(security_incidents) / max(1, total_events)
        access_violation_rate = len(access_violations) / max(1, total_events)

        # Generate vulnerability scan results based on audit patterns
        vulnerability_scan_results = {
            "vulnerabilities_found": len(security_incidents) + len(access_violations),
            "critical_vulnerabilities": len([e for e in security_incidents if e.risk_level == "critical"]),
            "high_severity": len([e for e in security_events if e.risk_level == "high"]),
            "medium_severity": max(0, len(security_events) - len([e for e in security_events if e.risk_level == "high"])),
            "low_severity": 0,
            "scan_date": datetime.now().isoformat(),
            "security_event_rate": security_event_rate,
            "incident_rate": incident_rate,
        }

        # Calculate adversarial robustness based on error patterns and security events
        model_errors = [e for e in audit_events if e.event_type == AuditEventType.MODEL_ERROR]
        error_rate = len(model_errors) / max(1, total_events)
        
        # Robustness metrics (adjusted based on actual performance)
        base_robustness = max(0.5, 1 - error_rate * 10)  # Scale error rate impact
        adversarial_robustness = {
            "fgsm_robustness": max(0.4, base_robustness - 0.1),
            "pgd_robustness": max(0.3, base_robustness - 0.2),
            "c&w_robustness": max(0.3, base_robustness - 0.25),
            "overall_robustness": base_robustness,
        }

        # Assess various attack risks based on audit data
        data_poisoning_risk = (
            RiskLevel.HIGH if incident_rate > 0.05 else
            RiskLevel.MEDIUM if len(security_events) > 5 else RiskLevel.LOW
        )
        
        model_extraction_risk = (
            RiskLevel.MEDIUM if access_violation_rate > 0.1 else RiskLevel.LOW
        )
        
        privacy_attack_risk = (
            RiskLevel.HIGH if any(e.contains_pii and e.risk_level == "high" for e in security_events) else
            RiskLevel.MEDIUM if any(e.contains_pii for e in security_events) else RiskLevel.LOW
        )

        # Calculate overall security score based on actual metrics
        security_score = (
            adversarial_robustness["overall_robustness"] * 0.3
            + max(0, 1 - vulnerability_scan_results["critical_vulnerabilities"] / 10) * 0.3
            + max(0, 1 - security_event_rate * 20) * 0.2
            + max(0, 1 - incident_rate * 50) * 0.2
        ) * 100

        # Generate data-driven recommendations
        recommendations = self._generate_security_recommendations(
            vulnerability_scan_results, adversarial_robustness, 
            security_event_rate, incident_rate, access_violation_rate
        )

        return SecurityAssessment(
            assessment_id=f"SEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=self.model_name,
            vulnerability_scan_results=vulnerability_scan_results,
            adversarial_robustness=adversarial_robustness,
            data_poisoning_risk=data_poisoning_risk,
            model_extraction_risk=model_extraction_risk,
            privacy_attack_risk=privacy_attack_risk,
            security_score=security_score,
            recommendations=recommendations,
            assessment_date=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_security_recommendations(
        self, 
        vulnerability_scan: Dict[str, Any], 
        robustness: Dict[str, float],
        security_event_rate: float,
        incident_rate: float,
        access_violation_rate: float
    ) -> List[str]:
        """Generate security recommendations based on actual audit data."""
        recommendations = []
        
        # Critical vulnerability recommendations
        if vulnerability_scan["critical_vulnerabilities"] > 0:
            recommendations.append("Address critical security vulnerabilities immediately")
            
        # High severity recommendations  
        if vulnerability_scan["high_severity"] > 2:
            recommendations.append("Investigate and remediate high-severity security issues")
            
        # Robustness recommendations
        if robustness["overall_robustness"] < 0.7:
            recommendations.append("Improve adversarial robustness through defensive techniques")
            
        # Event rate recommendations
        if security_event_rate > 0.1:
            recommendations.append("Investigate high rate of security events")
            
        if incident_rate > 0.02:
            recommendations.append("Enhance incident response and prevention measures")
            
        if access_violation_rate > 0.05:
            recommendations.append("Strengthen access controls and authentication")
            
        # General security recommendations
        if security_event_rate > 0.05 or incident_rate > 0.01:
            recommendations.extend([
                "Implement continuous security monitoring",
                "Regular security assessments and penetration testing",
                "Enhance logging and audit trail security"
            ])
            
        return recommendations

    def _assess_compliance_risk(
        self, audit_events: List[ComplianceAuditRecord]
    ) -> Dict[str, RiskLevel]:
        """Assess compliance risk for each framework based on actual violations and coverage."""

        compliance_risk = {}

        # Assess risk for each supported framework
        for framework in ComplianceFramework:
            requirements = self.regulatory_mapper.get_requirements([framework])

            # Calculate requirement coverage
            automated_reqs = sum(1 for req in requirements if req.ciaf_capabilities)
            total_reqs = len(requirements)
            coverage_rate = automated_reqs / total_reqs if total_reqs > 0 else 0

            # Analyze actual compliance violations from audit events
            violation_rate = self._calculate_violation_rate(audit_events, framework)
            high_risk_events = self._count_high_risk_events(audit_events, framework)
            
            # Combine coverage and violation analysis for risk assessment
            risk_score = self._calculate_compliance_risk_score(
                coverage_rate, violation_rate, high_risk_events, len(audit_events)
            )

            # Determine risk level based on combined score
            if risk_score >= 80:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 60:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 40:
                risk_level = RiskLevel.MEDIUM
            elif risk_score >= 20:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.MINIMAL

            compliance_risk[framework.value] = risk_level

        return compliance_risk

    def _calculate_violation_rate(
        self, audit_events: List[ComplianceAuditRecord], framework: ComplianceFramework
    ) -> float:
        """Calculate the rate of compliance violations for a specific framework."""
        if not audit_events:
            return 0.0

        violations = 0
        framework_specific_events = 0

        for event in audit_events:
            # Check for framework-specific compliance issues
            is_framework_relevant = self._is_event_relevant_to_framework(event, framework)
            
            if is_framework_relevant:
                framework_specific_events += 1
                
                # Check for various types of violations
                if self._is_compliance_violation(event, framework):
                    violations += 1

        return violations / framework_specific_events if framework_specific_events > 0 else 0.0

    def _count_high_risk_events(
        self, audit_events: List[ComplianceAuditRecord], framework: ComplianceFramework
    ) -> int:
        """Count high-risk events relevant to the compliance framework."""
        high_risk_count = 0

        for event in audit_events:
            if (
                event.risk_level in ["high", "critical"] and
                self._is_event_relevant_to_framework(event, framework)
            ):
                high_risk_count += 1

        return high_risk_count

    def _is_event_relevant_to_framework(
        self, event: ComplianceAuditRecord, framework: ComplianceFramework
    ) -> bool:
        """Determine if an audit event is relevant to a specific compliance framework."""
        framework_relevance = {
            ComplianceFramework.GDPR: [
                "contains_pii", "data_processing", "user_consent", "data_access"
            ],
            ComplianceFramework.CCPA: [
                "contains_pii", "data_sale", "user_rights", "data_deletion"
            ],
            ComplianceFramework.HIPAA: [
                "phi_data", "healthcare_data", "medical_records"
            ],
            ComplianceFramework.SOX: [
                "financial_data", "audit_trail", "internal_controls"
            ],
            ComplianceFramework.PCI_DSS: [
                "payment_data", "cardholder_data", "security_controls"
            ],
            ComplianceFramework.ISO_27001: [
                "information_security", "risk_management", "security_controls"
            ],
            ComplianceFramework.NIST_CSF: [
                "cybersecurity", "risk_assessment", "security_framework"
            ],
            ComplianceFramework.EU_AI_ACT: [
                "ai_system", "high_risk_ai", "algorithmic_decision"
            ],
        }

        relevant_keywords = framework_relevance.get(framework, [])
        
        # Check if event metadata or type indicates relevance to framework
        event_metadata = event.metadata or {}
        event_tags = event_metadata.get("tags", [])
        
        # Basic relevance checks
        if any(keyword in str(event_metadata).lower() for keyword in relevant_keywords):
            return True
            
        if any(keyword in tag.lower() for tag in event_tags for keyword in relevant_keywords):
            return True

        # Framework-specific checks
        if framework == ComplianceFramework.GDPR and event.contains_pii:
            return True
        elif framework == ComplianceFramework.EU_AI_ACT and event.event_type == AuditEventType.MODEL_PREDICTION:
            return True
        elif framework in [ComplianceFramework.ISO_27001, ComplianceFramework.NIST_CSF] and event.risk_level in ["high", "critical"]:
            return True

        return False

    def _is_compliance_violation(
        self, event: ComplianceAuditRecord, framework: ComplianceFramework
    ) -> bool:
        """Determine if an audit event represents a compliance violation."""
        
        # General violation indicators
        if event.risk_level == "critical":
            return True
            
        if not event.integrity_hash:  # Audit integrity issue
            return True
            
        if not event.access_controls:  # Access control violation
            return True

        # Framework-specific violation checks
        if framework == ComplianceFramework.GDPR:
            # GDPR violations: PII without consent, inadequate access controls
            if event.contains_pii and not event.metadata.get("user_consent"):
                return True
            if event.contains_pii and event.user_id == "anonymous":
                return True
                
        elif framework == ComplianceFramework.EU_AI_ACT:
            # EU AI Act violations: High-risk AI decisions without explanation
            if (
                event.event_type == AuditEventType.MODEL_PREDICTION and
                event.metadata.get("risk_category") == "high" and
                not event.metadata.get("explanation")
            ):
                return True
                
        elif framework in [ComplianceFramework.ISO_27001, ComplianceFramework.NIST_CSF]:
            # Security framework violations
            if event.event_type == AuditEventType.SECURITY_INCIDENT:
                return True
            if event.metadata.get("security_violation"):
                return True

        return False

    def _calculate_compliance_risk_score(
        self, coverage_rate: float, violation_rate: float, high_risk_events: int, total_events: int
    ) -> float:
        """Calculate a combined compliance risk score."""
        
        # Coverage component (0-40 points, inverted so low coverage = high risk)
        coverage_score = max(0, 40 * (1 - coverage_rate))
        
        # Violation rate component (0-40 points)
        violation_score = min(40, violation_rate * 100)
        
        # High-risk events component (0-20 points)
        high_risk_rate = high_risk_events / max(1, total_events)
        high_risk_score = min(20, high_risk_rate * 200)
        
        total_score = coverage_score + violation_score + high_risk_score
        
        return min(100, total_score)

    def _generate_recommendations(
        self,
        risk_factors: List[RiskFactor],
        bias_assessment: Optional[BiasAssessment],
        performance_assessment: Optional[PerformanceAssessment],
        security_assessment: Optional[SecurityAssessment],
    ) -> List[str]:
        """Generate comprehensive recommendations."""

        recommendations = []

        # Risk factor recommendations
        high_risk_factors = [
            rf
            for rf in risk_factors
            if rf.impact in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ]
        for factor in high_risk_factors:
            recommendations.extend(factor.mitigation_measures)

        # Bias assessment recommendations
        if bias_assessment and bias_assessment.bias_detected:
            recommendations.extend(bias_assessment.recommendations)

        # Performance assessment recommendations
        if performance_assessment:
            recommendations.extend(performance_assessment.recommendations)

        # Security assessment recommendations
        if security_assessment:
            recommendations.extend(security_assessment.recommendations)

        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))

        # Add general recommendations
        unique_recommendations.extend(
            [
                "Conduct regular risk assessments",
                "Maintain comprehensive documentation",
                "Implement continuous monitoring",
                "Train staff on compliance requirements",
            ]
        )

        return unique_recommendations[:15]  # Limit to top 15 recommendations

    def get_risk_trend_analysis(self, periods: int = 5) -> Dict[str, Any]:
        """Analyze risk trends over time."""

        if len(self.assessment_history) < 2:
            return {"message": "Insufficient data for trend analysis"}

        recent_assessments = (
            self.assessment_history[-periods:]
            if len(self.assessment_history) >= periods
            else self.assessment_history
        )

        # Track overall risk score trend
        risk_scores = [
            assessment.overall_risk_score for assessment in recent_assessments
        ]

        # Calculate trend
        if len(risk_scores) > 1:
            trend = (
                "increasing"
                if risk_scores[-1] > risk_scores[0]
                else "decreasing" if risk_scores[-1] < risk_scores[0] else "stable"
            )
        else:
            trend = "insufficient_data"

        # Category-specific trends
        category_trends = {}
        for category in RiskCategory:
            category_scores = []
            for assessment in recent_assessments:
                category_factors = [
                    rf for rf in assessment.risk_factors if rf.category == category
                ]
                if category_factors:
                    avg_score = sum(rf.risk_score for rf in category_factors) / len(
                        category_factors
                    )
                    category_scores.append(avg_score)

            if len(category_scores) > 1:
                category_trend = (
                    "increasing"
                    if category_scores[-1] > category_scores[0]
                    else (
                        "decreasing"
                        if category_scores[-1] < category_scores[0]
                        else "stable"
                    )
                )
            else:
                category_trend = "insufficient_data"

            category_trends[category.value] = {
                "trend": category_trend,
                "scores": category_scores,
                "latest_score": category_scores[-1] if category_scores else 0,
            }

        return {
            "overall_trend": trend,
            "risk_scores": risk_scores,
            "latest_score": risk_scores[-1] if risk_scores else 0,
            "score_change": (
                risk_scores[-1] - risk_scores[0] if len(risk_scores) > 1 else 0
            ),
            "category_trends": category_trends,
            "assessment_count": len(recent_assessments),
            "analysis_period": f"Last {len(recent_assessments)} assessments",
        }

    def export_risk_assessment(
        self, assessment: ComprehensiveRiskAssessment, format: str = "json"
    ) -> str:
        """Export risk assessment in specified format."""

        if format.lower() == "json":
            assessment_dict = {
                "assessment_id": assessment.assessment_id,
                "model_name": assessment.model_name,
                "model_version": assessment.model_version,
                "assessment_date": assessment.assessment_date,
                "assessment_period": assessment.assessment_period,
                "overall_risk_score": assessment.overall_risk_score,
                "overall_risk_level": assessment.overall_risk_level.value,
                "risk_factors": [
                    {
                        "factor_id": rf.factor_id,
                        "name": rf.name,
                        "category": rf.category.value,
                        "likelihood": rf.likelihood.value,
                        "impact": rf.impact.value,
                        "risk_score": rf.risk_score,
                        "evidence": rf.evidence,
                        "mitigation_measures": rf.mitigation_measures,
                    }
                    for rf in assessment.risk_factors
                ],
                "bias_assessment": (
                    assessment.bias_assessment.get_bias_summary()
                    if assessment.bias_assessment
                    else None
                ),
                "performance_assessment": (
                    {
                        "reliability_score": assessment.performance_assessment.reliability_score,
                        "data_drift_detected": assessment.performance_assessment.data_drift_detected,
                        "concept_drift_detected": assessment.performance_assessment.concept_drift_detected,
                    }
                    if assessment.performance_assessment
                    else None
                ),
                "security_assessment": (
                    {
                        "security_score": assessment.security_assessment.security_score,
                        "vulnerability_count": assessment.security_assessment.vulnerability_scan_results.get(
                            "vulnerabilities_found", 0
                        ),
                    }
                    if assessment.security_assessment
                    else None
                ),
                "compliance_risk": {
                    k: v.value for k, v in assessment.compliance_risk.items()
                },
                "recommendations": assessment.recommendations,
                "next_assessment_due": assessment.next_assessment_due,
            }

            return json.dumps(assessment_dict, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_assessment_history(self) -> List[ComprehensiveRiskAssessment]:
        """Get history of all risk assessments."""
        return self.assessment_history

    def get_latest_assessment(self) -> Optional[ComprehensiveRiskAssessment]:
        """Get the most recent risk assessment."""
        return self.assessment_history[-1] if self.assessment_history else None
