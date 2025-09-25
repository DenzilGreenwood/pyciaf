"""
Human Oversight Module for CIAF - EU AI Act Article 14 Compliance

This module provides comprehensive human oversight capabilities for AI systems,
including reviewer workflows, override mechanisms, and alerting systems to
comply with EU AI Act Article 14 requirements.

Created: 2025-09-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union

from ..inference import InferenceReceipt


class OversightLevel(Enum):
    """Levels of human oversight required."""
    NONE = "none"
    MONITORING = "monitoring"
    REVIEW_ON_ALERT = "review_on_alert"
    CONTINUOUS_REVIEW = "continuous_review"
    MANUAL_APPROVAL = "manual_approval"


class AlertSeverity(Enum):
    """Severity levels for oversight alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OversightDecision(Enum):
    """Possible oversight decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    MODIFY = "modify"
    DEFER = "defer"


class InterventionType(Enum):
    """Types of human intervention."""
    PREVENTION = "prevention"  # Stop action before execution
    CORRECTION = "correction"  # Modify ongoing action
    TERMINATION = "termination"  # Stop current action
    OVERRIDE = "override"  # Override system decision


@dataclass
class OversightAlert:
    """Alert requiring human oversight."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    model_name: str
    inference_id: Optional[str]
    description: str
    context: Dict[str, Any]
    created_timestamp: str
    threshold_triggered: str
    risk_factors: List[str]
    recommended_action: str
    auto_escalate_after: Optional[int] = None  # minutes
    requires_immediate_attention: bool = False
    
    def __post_init__(self):
        if self.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self.requires_immediate_attention = True


@dataclass
class OversightReview:
    """Human review record."""
    review_id: str
    alert_id: str
    reviewer_id: str
    reviewer_role: str
    decision: OversightDecision
    rationale: str
    intervention_type: Optional[InterventionType]
    review_timestamp: str
    response_time_seconds: float
    additional_notes: str = ""
    confidence_level: float = 1.0  # 0.0 - 1.0
    escalation_needed: bool = False
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Convert review to audit record format."""
        return {
            "event_type": "human_oversight_review",
            "review_id": self.review_id,
            "alert_id": self.alert_id,
            "reviewer": self.reviewer_id,
            "decision": self.decision.value,
            "timestamp": self.review_timestamp,
            "response_time": self.response_time_seconds,
            "intervention": self.intervention_type.value if self.intervention_type else None
        }


class HumanOversightEngine:
    """Core engine for human oversight management."""
    
    def __init__(self, model_name: str, oversight_level: OversightLevel = OversightLevel.REVIEW_ON_ALERT):
        self.model_name = model_name
        self.oversight_level = oversight_level
        self.active_alerts: Dict[str, OversightAlert] = {}
        self.review_history: Dict[str, OversightReview] = {}
        self.alert_handlers: Dict[str, Callable] = {}
        self.escalation_handlers: Dict[AlertSeverity, Callable] = {}
        self.oversight_thresholds = self._initialize_default_thresholds()
        self.metrics = {
            "total_alerts": 0,
            "total_reviews": 0,
            "average_response_time": 0.0,
            "escalation_rate": 0.0,
            "intervention_rate": 0.0
        }
    
    def _initialize_default_thresholds(self) -> Dict[str, Any]:
        """Initialize default oversight thresholds."""
        return {
            "confidence_threshold": 0.7,  # Below this triggers review
            "uncertainty_threshold": 0.3,  # Above this triggers review
            "bias_threshold": 0.1,  # Bias score above triggers review
            "drift_threshold": 0.15,  # Model drift above triggers review
            "volume_threshold": 100,  # Requests per minute threshold
            "error_rate_threshold": 0.05,  # Error rate threshold
            "anomaly_score_threshold": 0.8,  # Anomaly detection threshold
            "risk_score_threshold": 0.6,  # Risk assessment threshold
        }
    
    def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register custom alert handler."""
        self.alert_handlers[alert_type] = handler
    
    def register_escalation_handler(self, severity: AlertSeverity, handler: Callable):
        """Register escalation handler for specific severity."""
        self.escalation_handlers[severity] = handler
    
    def evaluate_inference_for_oversight(
        self, 
        inference_receipt: InferenceReceipt,
        prediction_confidence: float = 1.0,
        uncertainty_score: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[OversightAlert]:
        """Evaluate if inference requires human oversight."""
        
        context = context or {}
        risk_factors = []
        alert_needed = False
        alert_type = "routine_inference"
        severity = AlertSeverity.LOW
        
        # Check confidence threshold
        if prediction_confidence < self.oversight_thresholds["confidence_threshold"]:
            risk_factors.append(f"Low prediction confidence: {prediction_confidence:.3f}")
            alert_needed = True
            alert_type = "low_confidence"
            severity = AlertSeverity.MEDIUM
        
        # Check uncertainty threshold
        if uncertainty_score > self.oversight_thresholds["uncertainty_threshold"]:
            risk_factors.append(f"High uncertainty: {uncertainty_score:.3f}")
            alert_needed = True
            alert_type = "high_uncertainty"
            severity = max(severity, AlertSeverity.MEDIUM)
        
        # Check for high-risk domains
        if context.get("domain") in ["healthcare", "finance", "legal", "safety_critical"]:
            risk_factors.append("High-risk domain detected")
            alert_needed = True
            severity = max(severity, AlertSeverity.HIGH)
        
        # Check for sensitive data indicators
        if context.get("contains_pii", False) or context.get("sensitive_data", False):
            risk_factors.append("Sensitive data present")
            alert_needed = True
            severity = max(severity, AlertSeverity.HIGH)
        
        # Check oversight level requirements
        if self.oversight_level == OversightLevel.MANUAL_APPROVAL:
            alert_needed = True
            alert_type = "manual_approval_required"
        elif self.oversight_level == OversightLevel.CONTINUOUS_REVIEW:
            alert_needed = True
        
        if alert_needed:
            return self._create_oversight_alert(
                alert_type=alert_type,
                severity=severity,
                inference_id=inference_receipt.receipt_id,
                description=f"Inference requires human oversight: {', '.join(risk_factors)}",
                context=context,
                risk_factors=risk_factors,
                prediction_confidence=prediction_confidence,
                uncertainty_score=uncertainty_score
            )
        
        return None
    
    def _create_oversight_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        inference_id: Optional[str] = None,
        description: str = "",
        context: Optional[Dict[str, Any]] = None,
        risk_factors: Optional[List[str]] = None,
        **kwargs
    ) -> OversightAlert:
        """Create oversight alert."""
        
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(alert_type, severity, context)
        
        # Set auto-escalation timing
        auto_escalate_minutes = None
        if severity == AlertSeverity.HIGH:
            auto_escalate_minutes = 15
        elif severity == AlertSeverity.CRITICAL:
            auto_escalate_minutes = 5
        
        alert = OversightAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            model_name=self.model_name,
            inference_id=inference_id,
            description=description,
            context=context or {},
            created_timestamp=datetime.now(timezone.utc).isoformat(),
            threshold_triggered=f"{alert_type}_threshold",
            risk_factors=risk_factors or [],
            recommended_action=recommended_action,
            auto_escalate_after=auto_escalate_minutes
        )
        
        self.active_alerts[alert_id] = alert
        self.metrics["total_alerts"] += 1
        
        # Trigger alert handlers
        if alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert_type](alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
        
        # Handle immediate escalation for critical alerts
        if severity == AlertSeverity.CRITICAL:
            self._escalate_alert(alert)
        
        return alert
    
    def _determine_recommended_action(
        self, 
        alert_type: str, 
        severity: AlertSeverity, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Determine recommended action for alert."""
        
        if severity == AlertSeverity.CRITICAL:
            return "IMMEDIATE_INTERVENTION_REQUIRED"
        elif severity == AlertSeverity.HIGH:
            if alert_type == "low_confidence":
                return "REVIEW_PREDICTION_AND_CONSIDER_REJECTION"
            elif alert_type == "high_uncertainty":
                return "REQUEST_ADDITIONAL_INFORMATION_OR_DEFER"
            else:
                return "URGENT_REVIEW_REQUIRED"
        elif severity == AlertSeverity.MEDIUM:
            return "REVIEW_WITHIN_30_MINUTES"
        else:
            return "REVIEW_WHEN_CONVENIENT"
    
    def _escalate_alert(self, alert: OversightAlert):
        """Escalate alert to appropriate handlers."""
        if alert.severity in self.escalation_handlers:
            try:
                self.escalation_handlers[alert.severity](alert)
            except Exception as e:
                print(f"Escalation handler error: {e}")
        
        # Default escalation behavior
        if alert.severity == AlertSeverity.CRITICAL:
            print(f"🚨 CRITICAL ALERT: {alert.alert_id} - {alert.description}")
            print(f"Model: {alert.model_name}")
            print(f"Recommended Action: {alert.recommended_action}")
    
    def submit_review(
        self,
        alert_id: str,
        reviewer_id: str,
        reviewer_role: str,
        decision: OversightDecision,
        rationale: str,
        intervention_type: Optional[InterventionType] = None,
        confidence_level: float = 1.0,
        additional_notes: str = ""
    ) -> OversightReview:
        """Submit human review for an alert."""
        
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        review_timestamp = datetime.now(timezone.utc)
        
        # Calculate response time
        alert_created = datetime.fromisoformat(alert.created_timestamp.replace('Z', '+00:00'))
        response_time = (review_timestamp - alert_created).total_seconds()
        
        review = OversightReview(
            review_id=f"review_{uuid.uuid4().hex[:8]}",
            alert_id=alert_id,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            decision=decision,
            rationale=rationale,
            intervention_type=intervention_type,
            review_timestamp=review_timestamp.isoformat(),
            response_time_seconds=response_time,
            additional_notes=additional_notes,
            confidence_level=confidence_level,
            escalation_needed=(decision == OversightDecision.ESCALATE)
        )
        
        self.review_history[review.review_id] = review
        self.metrics["total_reviews"] += 1
        
        # Update average response time
        total_response_time = (self.metrics["average_response_time"] * (self.metrics["total_reviews"] - 1) + response_time)
        self.metrics["average_response_time"] = total_response_time / self.metrics["total_reviews"]
        
        # Remove from active alerts if resolved
        if decision in [OversightDecision.APPROVE, OversightDecision.REJECT]:
            del self.active_alerts[alert_id]
        
        # Handle escalation
        if decision == OversightDecision.ESCALATE:
            self._handle_escalation(alert, review)
        
        return review
    
    def _handle_escalation(self, alert: OversightAlert, review: OversightReview):
        """Handle escalated reviews."""
        escalated_alert = OversightAlert(
            alert_id=f"escalated_{alert.alert_id}",
            alert_type=f"escalated_{alert.alert_type}",
            severity=AlertSeverity.HIGH if alert.severity != AlertSeverity.CRITICAL else AlertSeverity.CRITICAL,
            model_name=alert.model_name,
            inference_id=alert.inference_id,
            description=f"ESCALATED: {alert.description}. Escalation reason: {review.rationale}",
            context={**alert.context, "escalated_from": alert.alert_id, "escalation_reason": review.rationale},
            created_timestamp=datetime.now(timezone.utc).isoformat(),
            threshold_triggered="escalation_threshold",
            risk_factors=alert.risk_factors + ["Manual escalation"],
            recommended_action="SENIOR_REVIEW_REQUIRED",
            auto_escalate_after=30  # Auto-escalate after 30 minutes
        )
        
        self.active_alerts[escalated_alert.alert_id] = escalated_alert
        self._escalate_alert(escalated_alert)
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[OversightAlert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        return sorted(alerts, key=lambda x: x.created_timestamp, reverse=True)
    
    def get_review_history(self, limit: int = 100) -> List[OversightReview]:
        """Get recent review history."""
        reviews = list(self.review_history.values())
        return sorted(reviews, key=lambda x: x.review_timestamp, reverse=True)[:limit]
    
    def get_oversight_metrics(self) -> Dict[str, Any]:
        """Get oversight performance metrics."""
        active_count = len(self.active_alerts)
        high_priority_count = len([a for a in self.active_alerts.values() if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]])
        
        # Calculate rates
        total_reviews = max(self.metrics["total_reviews"], 1)
        escalations = len([r for r in self.review_history.values() if r.escalation_needed])
        interventions = len([r for r in self.review_history.values() if r.intervention_type is not None])
        
        return {
            **self.metrics,
            "active_alerts": active_count,
            "high_priority_alerts": high_priority_count,
            "escalation_rate": escalations / total_reviews,
            "intervention_rate": interventions / total_reviews,
            "avg_response_time_minutes": self.metrics["average_response_time"] / 60,
            "oversight_level": self.oversight_level.value,
            "thresholds": self.oversight_thresholds
        }
    
    def configure_thresholds(self, new_thresholds: Dict[str, float]):
        """Update oversight thresholds."""
        self.oversight_thresholds.update(new_thresholds)
    
    def export_oversight_data(self, format: str = "json", include_context: bool = False) -> str:
        """Export oversight data for compliance reporting."""
        
        data = {
            "model_name": self.model_name,
            "oversight_level": self.oversight_level.value,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.get_oversight_metrics(),
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "recent_reviews": [asdict(review) for review in self.get_review_history(50)]
        }
        
        if not include_context:
            # Remove potentially sensitive context data
            for alert in data["active_alerts"]:
                alert.pop("context", None)
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class OversightIntegration:
    """Integration layer for connecting oversight to CIAF framework."""
    
    def __init__(self, framework_instance):
        self.framework = framework_instance
        self.oversight_engines: Dict[str, HumanOversightEngine] = {}
    
    def enable_oversight(self, model_name: str, oversight_level: OversightLevel = OversightLevel.REVIEW_ON_ALERT):
        """Enable human oversight for a model."""
        engine = HumanOversightEngine(model_name, oversight_level)
        self.oversight_engines[model_name] = engine
        return engine
    
    def check_inference_oversight(self, model_name: str, inference_receipt, **kwargs) -> Optional[OversightAlert]:
        """Check if inference requires oversight."""
        if model_name in self.oversight_engines:
            return self.oversight_engines[model_name].evaluate_inference_for_oversight(
                inference_receipt, **kwargs
            )
        return None
    
    def get_oversight_engine(self, model_name: str) -> Optional[HumanOversightEngine]:
        """Get oversight engine for model."""
        return self.oversight_engines.get(model_name)


# Demo and example usage
def demo_human_oversight():
    """Demonstrate human oversight capabilities."""
    
    print("🤖 CIAF Human Oversight Demo")
    print("=" * 40)
    
    # Create oversight engine
    engine = HumanOversightEngine("credit_risk_model", OversightLevel.REVIEW_ON_ALERT)
    
    # Configure thresholds for financial domain
    engine.configure_thresholds({
        "confidence_threshold": 0.85,  # Higher threshold for financial decisions
        "uncertainty_threshold": 0.2,
        "bias_threshold": 0.05,
        "risk_score_threshold": 0.4
    })
    
    # Set up alert handlers
    def critical_alert_handler(alert: OversightAlert):
        print(f"🚨 CRITICAL ALERT HANDLER TRIGGERED: {alert.alert_id}")
        print(f"   Description: {alert.description}")
        print(f"   Immediate attention required!")
    
    engine.register_escalation_handler(AlertSeverity.CRITICAL, critical_alert_handler)
    
    print("\n1. Creating realistic inference scenarios...")
    
    # Scenario 1: Low confidence prediction
    # Create a more realistic receipt-like object
    class MockInferenceReceipt:
        def __init__(self, receipt_id: str, model_name: str):
            self.receipt_id = receipt_id
            self.model_hash = f"sha256_{receipt_id}"
            self.model_name = model_name
            self.timestamp = datetime.now(timezone.utc).isoformat()
            self.input_hash = f"input_hash_{receipt_id}"
            self.prediction = {"class": "approved", "probability": 0.7}
    
    mock_receipt_1 = MockInferenceReceipt('test_receipt_001', 'credit_risk_model')
    
    alert_1 = engine.evaluate_inference_for_oversight(
        mock_receipt_1,
        prediction_confidence=0.7,  # Below threshold
        uncertainty_score=0.1,
        context={"domain": "finance", "loan_amount": 50000, "applicant_score": 720}
    )
    
    if alert_1:
        print(f"   Alert created: {alert_1.alert_id} ({alert_1.severity.value})")
    
    # Scenario 2: High-risk domain with sensitive data
    mock_receipt_2 = MockInferenceReceipt('test_receipt_002', 'medical_diagnosis_model')
    
    alert_2 = engine.evaluate_inference_for_oversight(
        mock_receipt_2,
        prediction_confidence=0.9,
        uncertainty_score=0.25,  # Above threshold
        context={"domain": "healthcare", "contains_pii": True, "patient_age": 65}
    )
    
    if alert_2:
        print(f"   Alert created: {alert_2.alert_id} ({alert_2.severity.value})")
    
    print("\n2. Human reviewer actions...")
    
    # Submit reviews
    if alert_1:
        review_1 = engine.submit_review(
            alert_id=alert_1.alert_id,
            reviewer_id="analyst_001",
            reviewer_role="Senior Risk Analyst",
            decision=OversightDecision.APPROVE,
            rationale="Confidence level acceptable for this risk category",
            confidence_level=0.9
        )
        print(f"   Review submitted: {review_1.review_id} - Decision: {review_1.decision.value}")
    
    if alert_2:
        review_2 = engine.submit_review(
            alert_id=alert_2.alert_id,
            reviewer_id="supervisor_001", 
            reviewer_role="Compliance Supervisor",
            decision=OversightDecision.ESCALATE,
            rationale="High uncertainty with PII requires senior review",
            intervention_type=InterventionType.PREVENTION,
            confidence_level=0.8
        )
        print(f"   Review submitted: {review_2.review_id} - Decision: {review_2.decision.value}")
    
    print("\n3. Oversight metrics...")
    metrics = engine.get_oversight_metrics()
    print(f"   Total alerts: {metrics['total_alerts']}")
    print(f"   Total reviews: {metrics['total_reviews']}")
    print(f"   Active alerts: {metrics['active_alerts']}")
    print(f"   Average response time: {metrics['avg_response_time_minutes']:.2f} minutes")
    print(f"   Escalation rate: {metrics['escalation_rate']:.2%}")
    
    print("\n4. Export compliance data...")
    export_data = engine.export_oversight_data(include_context=False)
    print("   Oversight data exported (truncated):")
    print(f"   {export_data[:200]}...")
    
    print("\n✅ Human oversight demo completed successfully!")


if __name__ == "__main__":
    demo_human_oversight()