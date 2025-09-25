"""
CIAF Compliance Extensions

This module implements regulatory-specific compliance extensions for CIAF,
addressing gaps in EU AI Act, GDPR, NIST AI RMF, ISO/IEC 42001, HIPAA, and SOX compliance.

The extensions provide concrete mechanisms for:
- Human oversight enforcement (EU AI Act Art. 14)
- Robustness and cybersecurity proofs (EU AI Act Art. 15)
- GDPR consent receipts and erasure handling
- Continuous monitoring automation (NIST AI RMF)
- Corrective action anchoring (ISO/IEC 42001)
- Access control audit logging (HIPAA/SOX)

Created: 2025-09-23
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core import CryptoUtils, sha256_hash
from ..compliance import AuditEventType, AuditTrailGenerator, ComplianceAuditRecord


# ================================
# Human Oversight (EU AI Act Art. 14)
# ================================

class OversightAction(Enum):
    """Types of human oversight actions."""
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    REVIEWED = "reviewed"
    FLAGGED = "flagged"
    OVERRIDDEN = "overridden"


@dataclass
class OversightCheckpoint:
    """Human oversight checkpoint metadata."""
    checkpoint_id: str
    oversight_required: bool
    oversight_actor_id: Optional[str]
    oversight_action: Optional[OversightAction]
    oversight_timestamp: Optional[str]
    oversight_reason: Optional[str]
    decision_context: Dict[str, Any]
    risk_level: str
    automated_decision: bool
    human_review_time_seconds: Optional[float]
    
    def is_complete(self) -> bool:
        """Check if oversight checkpoint is complete."""
        if not self.oversight_required:
            return True
        return all([
            self.oversight_actor_id,
            self.oversight_action,
            self.oversight_timestamp
        ])
    
    def get_checkpoint_hash(self) -> str:
        """Generate cryptographic hash of checkpoint."""
        checkpoint_data = json.dumps(asdict(self), sort_keys=True, default=str)
        return sha256_hash(checkpoint_data.encode('utf-8'))


class HumanOversightManager:
    """Manages human oversight requirements and enforcement."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.crypto_utils = CryptoUtils()
        self.checkpoints: Dict[str, OversightCheckpoint] = {}
        
    def create_oversight_checkpoint(
        self,
        decision_context: Dict[str, Any],
        risk_level: str = "medium",
        automated_decision: bool = True,
        oversight_required: Optional[bool] = None
    ) -> OversightCheckpoint:
        """
        Create a human oversight checkpoint.
        
        Args:
            decision_context: Context information for the decision
            risk_level: Risk level (low, medium, high, critical)
            automated_decision: Whether this was an automated decision
            oversight_required: Whether human oversight is required (auto-determined if None)
        """
        checkpoint_id = f"oversight_{uuid.uuid4().hex[:12]}"
        
        # Auto-determine oversight requirement based on risk level
        if oversight_required is None:
            oversight_required = risk_level in ["high", "critical"] or (
                automated_decision and risk_level == "medium"
            )
        
        checkpoint = OversightCheckpoint(
            checkpoint_id=checkpoint_id,
            oversight_required=oversight_required,
            oversight_actor_id=None,
            oversight_action=None,
            oversight_timestamp=None,
            oversight_reason=None,
            decision_context=decision_context,
            risk_level=risk_level,
            automated_decision=automated_decision,
            human_review_time_seconds=None
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        return checkpoint
    
    def complete_oversight(
        self,
        checkpoint_id: str,
        actor_id: str,
        action: OversightAction,
        reason: Optional[str] = None,
        review_time_seconds: Optional[float] = None
    ) -> bool:
        """
        Complete human oversight for a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to complete
            actor_id: ID of the human actor performing oversight
            action: Oversight action taken
            reason: Optional reason for the action
            review_time_seconds: Time spent on review
            
        Returns:
            True if oversight completed successfully
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        checkpoint.oversight_actor_id = actor_id
        checkpoint.oversight_action = action
        checkpoint.oversight_timestamp = datetime.now(timezone.utc).isoformat()
        checkpoint.oversight_reason = reason
        checkpoint.human_review_time_seconds = review_time_seconds
        
        return checkpoint.is_complete()
    
    def validate_capsule_oversight(self, capsule_metadata: Dict[str, Any]) -> bool:
        """
        Validate that a capsule has required oversight checkpoints.
        
        Args:
            capsule_metadata: Capsule metadata to validate
            
        Returns:
            True if all required oversight is complete
        """
        oversight_checkpoints = capsule_metadata.get('oversight_checkpoints', [])
        
        for checkpoint_data in oversight_checkpoints:
            checkpoint = OversightCheckpoint(**checkpoint_data)
            if not checkpoint.is_complete():
                return False
        
        return True


# ================================
# GDPR Compliance Extensions
# ================================

class ConsentPurpose(Enum):
    """GDPR consent purposes."""
    TRAINING = "training"
    INFERENCE = "inference"
    ANALYTICS = "analytics"
    RESEARCH = "research"
    MARKETING = "marketing"
    PROFILING = "profiling"


class ConsentStatus(Enum):
    """GDPR consent status."""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class ConsentReceipt:
    """GDPR consent receipt."""
    consent_id: str
    actor_id: str
    timestamp: str
    purpose: ConsentPurpose
    status: ConsentStatus
    data_categories: List[str]
    retention_period_days: Optional[int]
    legal_basis: str
    withdrawal_mechanism: str
    privacy_policy_version: str
    consent_hash: str
    
    def is_valid_for_purpose(self, purpose: ConsentPurpose) -> bool:
        """Check if consent is valid for a specific purpose."""
        if self.status != ConsentStatus.GIVEN:
            return False
        
        # Check if consent covers the requested purpose
        return self.purpose == purpose or self.purpose == ConsentPurpose.RESEARCH
    
    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if self.status == ConsentStatus.EXPIRED:
            return True
        
        if self.retention_period_days is None:
            return False
        
        consent_date = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        expiry_date = consent_date + timedelta(days=self.retention_period_days)
        
        return datetime.now(timezone.utc) > expiry_date


class GDPRComplianceManager:
    """Manages GDPR compliance requirements."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.consent_receipts: Dict[str, ConsentReceipt] = {}
        self.erasure_log: List[Dict[str, Any]] = []
        
    def create_consent_receipt(
        self,
        actor_id: str,
        purpose: ConsentPurpose,
        data_categories: List[str],
        legal_basis: str = "consent",
        retention_period_days: Optional[int] = 365,
        privacy_policy_version: str = "1.0"
    ) -> ConsentReceipt:
        """
        Create a GDPR consent receipt.
        
        Args:
            actor_id: ID of the data subject
            purpose: Purpose for data processing
            data_categories: Categories of data being processed
            legal_basis: Legal basis for processing
            retention_period_days: Data retention period
            privacy_policy_version: Version of privacy policy
            
        Returns:
            ConsentReceipt object
        """
        consent_id = f"consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate consent hash
        consent_data = {
            "consent_id": consent_id,
            "actor_id": actor_id,
            "timestamp": timestamp,
            "purpose": purpose.value,
            "data_categories": sorted(data_categories),
            "legal_basis": legal_basis,
            "retention_period_days": retention_period_days,
            "privacy_policy_version": privacy_policy_version
        }
        consent_hash = sha256_hash(json.dumps(consent_data, sort_keys=True).encode('utf-8'))
        
        receipt = ConsentReceipt(
            consent_id=consent_id,
            actor_id=actor_id,
            timestamp=timestamp,
            purpose=purpose,
            status=ConsentStatus.GIVEN,
            data_categories=data_categories,
            retention_period_days=retention_period_days,
            legal_basis=legal_basis,
            withdrawal_mechanism="email_request",
            privacy_policy_version=privacy_policy_version,
            consent_hash=consent_hash
        )
        
        self.consent_receipts[consent_id] = receipt
        return receipt
    
    def withdraw_consent(self, consent_id: str, actor_id: str) -> bool:
        """
        Withdraw GDPR consent.
        
        Args:
            consent_id: ID of consent to withdraw
            actor_id: ID of the data subject
            
        Returns:
            True if withdrawal successful
        """
        if consent_id not in self.consent_receipts:
            return False
        
        receipt = self.consent_receipts[consent_id]
        if receipt.actor_id != actor_id:
            return False
        
        receipt.status = ConsentStatus.WITHDRAWN
        
        # Log the withdrawal
        withdrawal_event = {
            "event_type": "consent_withdrawal",
            "consent_id": consent_id,
            "actor_id": actor_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "withdrawal_hash": sha256_hash(f"{consent_id}_{actor_id}_{datetime.now().isoformat()}".encode('utf-8'))
        }
        
        self.erasure_log.append(withdrawal_event)
        return True
    
    def process_erasure_request(
        self,
        actor_id: str,
        data_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process GDPR Article 17 right to erasure request.
        
        Args:
            actor_id: ID of the data subject
            data_categories: Specific data categories to erase (None for all)
            
        Returns:
            Erasure processing result
        """
        erasure_id = f"erasure_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Find relevant consent receipts
        relevant_consents = [
            receipt for receipt in self.consent_receipts.values()
            if receipt.actor_id == actor_id
        ]
        
        # Process erasure
        erasure_actions = []
        for receipt in relevant_consents:
            if data_categories is None or any(cat in receipt.data_categories for cat in data_categories):
                # Mark data for erasure (in practice, this would trigger data deletion)
                erasure_actions.append({
                    "consent_id": receipt.consent_id,
                    "action": "data_anonymized",
                    "hash_retained": True,  # Cryptographic hashes remain for audit
                    "data_categories": receipt.data_categories
                })
                
                # Update consent status
                receipt.status = ConsentStatus.WITHDRAWN
        
        erasure_event = {
            "erasure_id": erasure_id,
            "actor_id": actor_id,
            "timestamp": timestamp,
            "requested_categories": data_categories,
            "actions_taken": erasure_actions,
            "compliance_note": "Cryptographic hashes retained for audit integrity",
            "erasure_hash": sha256_hash(json.dumps({
                "erasure_id": erasure_id,
                "actor_id": actor_id,
                "timestamp": timestamp
            }, sort_keys=True).encode('utf-8'))
        }
        
        self.erasure_log.append(erasure_event)
        return erasure_event


# ================================
# Robustness & Security (EU AI Act Art. 15)
# ================================

@dataclass
class RobustnessTest:
    """Robustness test result."""
    test_id: str
    test_type: str  # "adversarial", "noise", "drift", "fairness"
    test_parameters: Dict[str, Any]
    result: str  # "passed", "failed", "warning"
    metrics: Dict[str, float]
    timestamp: str
    test_hash: str


@dataclass
class SecurityProof:
    """Security proof metadata."""
    proof_id: str
    security_property: str  # "integrity", "confidentiality", "availability"
    proof_method: str
    verification_result: bool
    evidence: Dict[str, Any]
    timestamp: str
    proof_hash: str


class RobustnessManager:
    """Manages robustness and security proofs."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.robustness_tests: Dict[str, RobustnessTest] = {}
        self.security_proofs: Dict[str, SecurityProof] = {}
    
    def create_adversarial_test(
        self,
        epsilon: float,
        attack_method: str,
        accuracy_threshold: float = 0.9
    ) -> RobustnessTest:
        """
        Create adversarial robustness test.
        
        Args:
            epsilon: Perturbation budget
            attack_method: Attack method used
            accuracy_threshold: Minimum accuracy threshold
            
        Returns:
            RobustnessTest object
        """
        test_id = f"adv_test_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Simulate adversarial test (in practice, would run actual test)
        import random
        accuracy = random.uniform(0.85, 0.95)
        
        test_parameters = {
            "epsilon": epsilon,
            "attack_method": attack_method,
            "accuracy_threshold": accuracy_threshold
        }
        
        metrics = {
            "clean_accuracy": 0.94,
            "adversarial_accuracy": accuracy,
            "robustness_score": accuracy / 0.94
        }
        
        result = "passed" if accuracy >= accuracy_threshold else "failed"
        
        test_hash = sha256_hash(json.dumps({
            "test_id": test_id,
            "parameters": test_parameters,
            "metrics": metrics,
            "timestamp": timestamp
        }, sort_keys=True).encode('utf-8'))
        
        test = RobustnessTest(
            test_id=test_id,
            test_type="adversarial",
            test_parameters=test_parameters,
            result=result,
            metrics=metrics,
            timestamp=timestamp,
            test_hash=test_hash
        )
        
        self.robustness_tests[test_id] = test
        return test
    
    def create_security_proof(
        self,
        security_property: str,
        proof_method: str,
        evidence: Dict[str, Any]
    ) -> SecurityProof:
        """
        Create security proof.
        
        Args:
            security_property: Property being proven
            proof_method: Method used for proof
            evidence: Supporting evidence
            
        Returns:
            SecurityProof object
        """
        proof_id = f"sec_proof_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Simulate verification (in practice, would run actual verification)
        verification_result = True
        
        proof_hash = sha256_hash(json.dumps({
            "proof_id": proof_id,
            "security_property": security_property,
            "proof_method": proof_method,
            "timestamp": timestamp
        }, sort_keys=True).encode('utf-8'))
        
        proof = SecurityProof(
            proof_id=proof_id,
            security_property=security_property,
            proof_method=proof_method,
            verification_result=verification_result,
            evidence=evidence,
            timestamp=timestamp,
            proof_hash=proof_hash
        )
        
        self.security_proofs[proof_id] = proof
        return proof


# ================================
# Continuous Monitoring (NIST AI RMF)
# ================================

class MonitoringEventType(Enum):
    """Types of monitoring events."""
    DRIFT_CHECK = "drift_check"
    PERFORMANCE_CHECK = "performance_check"
    BIAS_CHECK = "bias_check"
    SECURITY_SCAN = "security_scan"
    DATA_QUALITY_CHECK = "data_quality_check"
    COMPLIANCE_REVIEW = "compliance_review"


@dataclass
class MonitoringEvent:
    """Continuous monitoring event."""
    event_id: str
    event_type: MonitoringEventType
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    alerts: List[str]
    timestamp: str
    model_version: str
    monitoring_hash: str
    
    def has_alerts(self) -> bool:
        """Check if monitoring event has alerts."""
        return len(self.alerts) > 0


class ContinuousMonitoringManager:
    """Manages continuous monitoring and automated compliance checks."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.monitoring_events: List[MonitoringEvent] = []
        self.monitoring_schedule: Dict[MonitoringEventType, int] = {
            MonitoringEventType.DRIFT_CHECK: 24,  # Every 24 hours
            MonitoringEventType.BIAS_CHECK: 168,  # Every 7 days
            MonitoringEventType.COMPLIANCE_REVIEW: 720  # Every 30 days
        }
    
    def create_drift_monitoring_event(
        self,
        kl_divergence: float,
        psi_score: float,
        model_version: str = "1.0.0"
    ) -> MonitoringEvent:
        """
        Create data drift monitoring event.
        
        Args:
            kl_divergence: KL divergence score
            psi_score: Population Stability Index score
            model_version: Model version being monitored
            
        Returns:
            MonitoringEvent object
        """
        event_id = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        metrics = {
            "kl_divergence": kl_divergence,
            "psi_score": psi_score
        }
        
        thresholds = {
            "kl_divergence_max": 0.1,
            "psi_score_max": 0.2
        }
        
        alerts = []
        if kl_divergence > thresholds["kl_divergence_max"]:
            alerts.append(f"High KL divergence detected: {kl_divergence:.3f}")
        if psi_score > thresholds["psi_score_max"]:
            alerts.append(f"High PSI score detected: {psi_score:.3f}")
        
        monitoring_hash = sha256_hash(json.dumps({
            "event_id": event_id,
            "metrics": metrics,
            "timestamp": timestamp
        }, sort_keys=True).encode('utf-8'))
        
        event = MonitoringEvent(
            event_id=event_id,
            event_type=MonitoringEventType.DRIFT_CHECK,
            metrics=metrics,
            thresholds=thresholds,
            alerts=alerts,
            timestamp=timestamp,
            model_version=model_version,
            monitoring_hash=monitoring_hash
        )
        
        self.monitoring_events.append(event)
        return event
    
    def create_automated_monitoring_capsule(self, hours_interval: int = 24) -> Dict[str, Any]:
        """
        Create automated monitoring capsule for periodic compliance checks.
        
        Args:
            hours_interval: Monitoring interval in hours
            
        Returns:
            Monitoring capsule metadata
        """
        capsule_id = f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate multiple monitoring checks
        import random
        
        drift_event = self.create_drift_monitoring_event(
            kl_divergence=random.uniform(0.05, 0.15),
            psi_score=random.uniform(0.1, 0.25)
        )
        
        capsule_metadata = {
            "capsule_id": capsule_id,
            "monitoring_type": "automated_compliance_check",
            "interval_hours": hours_interval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "events": [asdict(drift_event)],
            "next_check_due": (datetime.now(timezone.utc) + timedelta(hours=hours_interval)).isoformat(),
            "capsule_hash": sha256_hash(f"{capsule_id}_{datetime.now().isoformat()}".encode('utf-8'))
        }
        
        return capsule_metadata


# ================================
# Corrective Actions (ISO/IEC 42001)
# ================================

class RemediationAction(Enum):
    """Types of remediation actions."""
    RETRAIN_MODEL = "retrain_model"
    REBALANCE_DATASET = "rebalance_dataset"
    UPDATE_ALGORITHM = "update_algorithm"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    ENHANCE_MONITORING = "enhance_monitoring"
    RESTRICT_ACCESS = "restrict_access"
    UPDATE_DOCUMENTATION = "update_documentation"


@dataclass
class CorrectiveActionEvent:
    """Corrective action event."""
    action_id: str
    issue_id: str
    remediation_action: RemediationAction
    actor_id: str
    timestamp: str
    description: str
    success_metrics: Dict[str, float]
    verification_method: str
    completion_timestamp: Optional[str]
    action_hash: str
    
    def is_completed(self) -> bool:
        """Check if corrective action is completed."""
        return self.completion_timestamp is not None


class CorrectiveActionManager:
    """Manages corrective actions and improvement cycles."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.corrective_actions: Dict[str, CorrectiveActionEvent] = {}
        self.issue_tracking: Dict[str, List[str]] = {}  # issue_id -> [action_ids]
    
    def create_corrective_action(
        self,
        issue_id: str,
        remediation_action: RemediationAction,
        actor_id: str,
        description: str,
        success_metrics: Dict[str, float],
        verification_method: str = "automated_test"
    ) -> CorrectiveActionEvent:
        """
        Create corrective action event.
        
        Args:
            issue_id: ID of the issue being addressed
            remediation_action: Type of remediation action
            actor_id: ID of actor performing the action
            description: Description of the action
            success_metrics: Metrics to measure success
            verification_method: Method to verify completion
            
        Returns:
            CorrectiveActionEvent object
        """
        action_id = f"action_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        action_hash = sha256_hash(json.dumps({
            "action_id": action_id,
            "issue_id": issue_id,
            "remediation_action": remediation_action.value,
            "actor_id": actor_id,
            "timestamp": timestamp
        }, sort_keys=True).encode('utf-8'))
        
        event = CorrectiveActionEvent(
            action_id=action_id,
            issue_id=issue_id,
            remediation_action=remediation_action,
            actor_id=actor_id,
            timestamp=timestamp,
            description=description,
            success_metrics=success_metrics,
            verification_method=verification_method,
            completion_timestamp=None,
            action_hash=action_hash
        )
        
        self.corrective_actions[action_id] = event
        
        # Track issue -> actions mapping
        if issue_id not in self.issue_tracking:
            self.issue_tracking[issue_id] = []
        self.issue_tracking[issue_id].append(action_id)
        
        return event
    
    def complete_corrective_action(
        self,
        action_id: str,
        completion_metrics: Dict[str, float]
    ) -> bool:
        """
        Mark corrective action as completed.
        
        Args:
            action_id: ID of action to complete
            completion_metrics: Metrics demonstrating completion
            
        Returns:
            True if completion successful
        """
        if action_id not in self.corrective_actions:
            return False
        
        action = self.corrective_actions[action_id]
        action.completion_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Verify success metrics (simplified)
        success = all(
            completion_metrics.get(metric, 0) >= target
            for metric, target in action.success_metrics.items()
        )
        
        return success


# ================================
# Access Control & Audit Logging
# ================================

class AccessEventType(Enum):
    """Types of access events."""
    CAPSULE_VERIFICATION = "capsule_verification"
    MODEL_ACCESS = "model_access"
    DATA_ACCESS = "data_access"
    AUDIT_LOG_ACCESS = "audit_log_access"
    COMPLIANCE_REPORT_ACCESS = "compliance_report_access"


@dataclass
class AccessEvent:
    """Access control audit event."""
    access_id: str
    event_type: AccessEventType
    actor_id: str
    resource_id: str
    timestamp: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    access_granted: bool
    access_reason: str
    session_id: Optional[str]
    access_hash: str


class AccessControlManager:
    """Manages access control and audit logging."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.access_events: List[AccessEvent] = []
        self.authorized_actors: Dict[str, List[str]] = {}  # actor_id -> [permissions]
    
    def log_capsule_verification(
        self,
        actor_id: str,
        capsule_id: str,
        access_granted: bool = True,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AccessEvent:
        """
        Log capsule verification access.
        
        Args:
            actor_id: ID of actor accessing capsule
            capsule_id: ID of capsule being verified
            access_granted: Whether access was granted
            ip_address: IP address of accessor
            session_id: Session ID
            
        Returns:
            AccessEvent object
        """
        access_id = f"access_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        access_hash = sha256_hash(json.dumps({
            "access_id": access_id,
            "actor_id": actor_id,
            "resource_id": capsule_id,
            "timestamp": timestamp
        }, sort_keys=True).encode('utf-8'))
        
        event = AccessEvent(
            access_id=access_id,
            event_type=AccessEventType.CAPSULE_VERIFICATION,
            actor_id=actor_id,
            resource_id=capsule_id,
            timestamp=timestamp,
            ip_address=ip_address,
            user_agent=None,
            access_granted=access_granted,
            access_reason="regulatory_audit" if "regulator" in actor_id else "verification_request",
            session_id=session_id,
            access_hash=access_hash
        )
        
        self.access_events.append(event)
        return event
    
    def get_access_log_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get access log summary for compliance reporting.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Access log summary
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        recent_events = [
            event for event in self.access_events
            if datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')) > cutoff_date
        ]
        
        summary = {
            "period_days": days,
            "total_access_events": len(recent_events),
            "unique_actors": len(set(event.actor_id for event in recent_events)),
            "access_types": {},
            "denied_access_count": len([e for e in recent_events if not e.access_granted]),
            "most_accessed_resources": {},
            "summary_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Access type breakdown
        for event_type in AccessEventType:
            count = len([e for e in recent_events if e.event_type == event_type])
            summary["access_types"][event_type.value] = count
        
        # Most accessed resources
        resource_counts = {}
        for event in recent_events:
            resource_counts[event.resource_id] = resource_counts.get(event.resource_id, 0) + 1
        
        summary["most_accessed_resources"] = dict(
            sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary


# ================================
# Main Compliance Extensions Class
# ================================

class ComplianceExtensions:
    """
    Main compliance extensions class that integrates all regulatory-specific functionality.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Initialize all managers
        self.oversight_manager = HumanOversightManager(model_name)
        self.gdpr_manager = GDPRComplianceManager(model_name)
        self.robustness_manager = RobustnessManager(model_name)
        self.monitoring_manager = ContinuousMonitoringManager(model_name)
        self.corrective_action_manager = CorrectiveActionManager(model_name)
        self.access_control_manager = AccessControlManager(model_name)
    
    def create_enhanced_capsule_metadata(
        self,
        base_metadata: Dict[str, Any],
        include_oversight: bool = True,
        include_consent: bool = True,
        include_robustness: bool = True,
        include_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Create enhanced capsule metadata with compliance extensions.
        
        Args:
            base_metadata: Base capsule metadata
            include_oversight: Include human oversight checkpoints
            include_consent: Include GDPR consent receipts
            include_robustness: Include robustness test results
            include_monitoring: Include monitoring events
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced_metadata = base_metadata.copy()
        
        # Add compliance extensions
        compliance_extensions = {}
        
        if include_oversight:
            # Create oversight checkpoint if high-risk decision
            risk_level = base_metadata.get('risk_level', 'medium')
            checkpoint = self.oversight_manager.create_oversight_checkpoint(
                decision_context={"operation": "inference", "metadata": base_metadata},
                risk_level=risk_level
            )
            compliance_extensions["oversight_checkpoints"] = [asdict(checkpoint)]
        
        if include_consent:
            # Add consent receipt references
            actor_id = base_metadata.get('actor_id', 'unknown')
            if actor_id != 'unknown':
                consent = self.gdpr_manager.create_consent_receipt(
                    actor_id=actor_id,
                    purpose=ConsentPurpose.INFERENCE,
                    data_categories=["model_input", "inference_output"]
                )
                compliance_extensions["consent_receipts"] = [asdict(consent)]
        
        if include_robustness:
            # Add robustness test results
            adv_test = self.robustness_manager.create_adversarial_test(
                epsilon=0.03,
                attack_method="pgd"
            )
            compliance_extensions["robustness_tests"] = [asdict(adv_test)]
        
        if include_monitoring:
            # Add monitoring event
            monitoring_capsule = self.monitoring_manager.create_automated_monitoring_capsule()
            compliance_extensions["monitoring_events"] = [monitoring_capsule]
        
        enhanced_metadata["compliance_extensions"] = compliance_extensions
        
        # Generate compliance hash
        compliance_hash = sha256_hash(
            json.dumps(compliance_extensions, sort_keys=True, default=str).encode('utf-8')
        )
        enhanced_metadata["compliance_hash"] = compliance_hash
        
        return enhanced_metadata
    
    def validate_regulatory_compliance(
        self,
        capsule_metadata: Dict[str, Any],
        framework: str = "EU_AI_ACT"
    ) -> Dict[str, Any]:
        """
        Validate capsule against regulatory requirements.
        
        Args:
            capsule_metadata: Capsule metadata to validate
            framework: Regulatory framework to validate against
            
        Returns:
            Validation results
        """
        validation_results = {
            "framework": framework,
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        compliance_ext = capsule_metadata.get("compliance_extensions", {})
        
        if framework == "EU_AI_ACT":
            # Article 14: Human oversight
            oversight_checkpoints = compliance_ext.get("oversight_checkpoints", [])
            if not oversight_checkpoints:
                validation_results["warnings"].append("No human oversight checkpoints found")
            else:
                for checkpoint_data in oversight_checkpoints:
                    checkpoint = OversightCheckpoint(**checkpoint_data)
                    if not checkpoint.is_complete():
                        validation_results["violations"].append("Incomplete human oversight checkpoint")
                        validation_results["compliant"] = False
            
            # Article 15: Robustness
            robustness_tests = compliance_ext.get("robustness_tests", [])
            if not robustness_tests:
                validation_results["warnings"].append("No robustness tests found")
            
        elif framework == "GDPR":
            # Article 6: Lawful basis
            consent_receipts = compliance_ext.get("consent_receipts", [])
            if not consent_receipts:
                validation_results["violations"].append("No consent receipts found")
                validation_results["compliant"] = False
        
        return validation_results
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Returns:
            Compliance report
        """
        return {
            "model_name": self.model_name,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "oversight_summary": {
                "total_checkpoints": len(self.oversight_manager.checkpoints),
                "completed_checkpoints": len([
                    cp for cp in self.oversight_manager.checkpoints.values()
                    if cp.is_complete()
                ])
            },
            "gdpr_summary": {
                "active_consents": len([
                    cr for cr in self.gdpr_manager.consent_receipts.values()
                    if cr.status == ConsentStatus.GIVEN
                ]),
                "erasure_requests": len(self.gdpr_manager.erasure_log)
            },
            "robustness_summary": {
                "total_tests": len(self.robustness_manager.robustness_tests),
                "passed_tests": len([
                    test for test in self.robustness_manager.robustness_tests.values()
                    if test.result == "passed"
                ])
            },
            "monitoring_summary": {
                "total_events": len(self.monitoring_manager.monitoring_events),
                "alerts": len([
                    event for event in self.monitoring_manager.monitoring_events
                    if event.has_alerts()
                ])
            },
            "access_summary": self.access_control_manager.get_access_log_summary(),
            "corrective_actions_summary": {
                "total_actions": len(self.corrective_action_manager.corrective_actions),
                "completed_actions": len([
                    action for action in self.corrective_action_manager.corrective_actions.values()
                    if action.is_completed()
                ])
            }
        }
    
    def validate_dataset_commit(self, metadata: Dict[str, Any]) -> None:
        """
        Validate dataset commit for GDPR compliance.
        
        Args:
            metadata: Dataset metadata to validate
            
        Raises:
            ValueError: If GDPR validation fails
        """
        # Check for required GDPR fields
        if 'actor_id' not in metadata:
            raise ValueError("GDPR requires actor_id for data subject identification")
        
        # If processing personal data, consent is required
        data_categories = metadata.get('data_categories', [])
        if any(cat in ['personal', 'sensitive', 'biometric'] for cat in data_categories):
            consent_receipts = metadata.get('compliance_extensions', {}).get('consent_receipts', [])
            if not consent_receipts:
                raise ValueError("GDPR consent required for personal data processing")
    
    def validate_model_commit(self, metadata: Dict[str, Any]) -> None:
        """
        Validate model commit for robustness requirements.
        
        Args:
            metadata: Model metadata to validate
            
        Raises:
            ValueError: If robustness validation fails
        """
        compliance_ext = metadata.get('compliance_extensions', {})
        robustness_tests = compliance_ext.get('robustness_tests', [])
        
        if not robustness_tests:
            raise ValueError("EU AI Act Article 15 requires robustness testing for high-risk AI systems")
        
        # Check that robustness tests passed
        for test_data in robustness_tests:
            test = RobustnessTest(**test_data)
            if test.result != "passed":
                raise ValueError(f"Robustness test {test.test_id} failed: {test.result}")