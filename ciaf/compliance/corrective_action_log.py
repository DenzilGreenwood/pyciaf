"""
Corrective Action Log Module for CIAF

This module provides comprehensive corrective action tracking and management
for AI model retraining, remediation, and compliance corrections.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ActionType(Enum):
    """Types of corrective actions."""

    MODEL_RETRAINING = "Model Retraining"
    DATA_AUGMENTATION = "Data Augmentation"
    BIAS_CORRECTION = "Bias Correction"
    FAIRNESS_ADJUSTMENT = "Fairness Adjustment"
    SECURITY_PATCH = "Security Patch"
    PERFORMANCE_OPTIMIZATION = "Performance Optimization"
    COMPLIANCE_UPDATE = "Compliance Update"
    DOCUMENTATION_UPDATE = "Documentation Update"
    ACCESS_CONTROL_FIX = "Access Control Fix"
    AUDIT_TRAIL_REPAIR = "Audit Trail Repair"


class ActionStatus(Enum):
    """Status of corrective actions."""

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    VERIFIED = "Verified"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class TriggerType(Enum):
    """Types of triggers for corrective actions."""

    BIAS_DRIFT = "Bias drift detected"
    PERFORMANCE_DEGRADATION = "Performance degradation"
    COMPLIANCE_VIOLATION = "Compliance violation"
    SECURITY_INCIDENT = "Security incident"
    AUDIT_FINDING = "Audit finding"
    REGULATORY_CHANGE = "Regulatory change"
    USER_COMPLAINT = "User complaint"
    SCHEDULED_REVIEW = "Scheduled review"
    AUTOMATED_ALERT = "Automated alert"


@dataclass
class CorrectiveAction:
    """Individual corrective action record."""

    action_id: str
    trigger: str
    trigger_type: TriggerType
    detection_method: str
    action_type: ActionType
    description: str
    approved_by: str
    implemented_by: Optional[str] = None
    date_created: str = None
    date_approved: Optional[str] = None
    date_applied: Optional[str] = None
    date_verified: Optional[str] = None
    status: ActionStatus = ActionStatus.PENDING
    priority: str = "Medium"
    linked_training_snapshot: Optional[str] = None
    linked_model_version: Optional[str] = None
    evidence_files: List[str] = None
    verification_criteria: List[str] = None
    verification_results: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[float] = None
    actual_cost: Optional[float] = None
    effectiveness_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.date_created is None:
            self.date_created = datetime.now(timezone.utc).isoformat()
        if self.evidence_files is None:
            self.evidence_files = []
        if self.verification_criteria is None:
            self.verification_criteria = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to string values
        result["trigger_type"] = self.trigger_type.value
        result["action_type"] = self.action_type.value
        result["status"] = self.status.value
        return result

    def get_action_hash(self) -> str:
        """Generate cryptographic hash of action data."""
        # Create hash-stable representation
        hash_data = {
            "action_id": self.action_id,
            "trigger": self.trigger,
            "action_type": self.action_type.value,
            "date_created": self.date_created,
            "approved_by": self.approved_by,
        }
        data_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class CorrectiveActionSummary:
    """Summary of corrective actions for reporting."""

    total_actions: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    by_trigger: Dict[str, int]
    avg_resolution_time_days: float
    effectiveness_scores: List[float]
    total_cost: float
    period_start: str
    period_end: str


class CorrectiveActionLogger:
    """Corrective action logging and management system."""

    def __init__(self, model_name: str):
        """Initialize corrective action logger."""
        self.model_name = model_name
        self.actions: List[CorrectiveAction] = []
        self.action_index = {}  # For fast lookups

    def create_action(
        self,
        trigger: str,
        trigger_type: TriggerType,
        detection_method: str,
        action_type: ActionType,
        description: str,
        approved_by: str,
        priority: str = "Medium",
        evidence_files: Optional[List[str]] = None,
        verification_criteria: Optional[List[str]] = None,
        cost_estimate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CorrectiveAction:
        """Create a new corrective action."""

        action_id = (
            f"CAL_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"
        )

        action = CorrectiveAction(
            action_id=action_id,
            trigger=trigger,
            trigger_type=trigger_type,
            detection_method=detection_method,
            action_type=action_type,
            description=description,
            approved_by=approved_by,
            priority=priority,
            evidence_files=evidence_files or [],
            verification_criteria=verification_criteria or [],
            cost_estimate=cost_estimate,
            metadata=metadata or {},
        )

        self.actions.append(action)
        self.action_index[action_id] = action

        return action

    def approve_action(
        self, action_id: str, approved_by: str, approval_notes: Optional[str] = None
    ) -> bool:
        """Approve a corrective action."""

        if action_id not in self.action_index:
            return False

        action = self.action_index[action_id]
        action.status = ActionStatus.IN_PROGRESS
        action.date_approved = datetime.now(timezone.utc).isoformat()
        action.approved_by = approved_by

        if approval_notes:
            action.metadata["approval_notes"] = approval_notes

        return True

    def implement_action(
        self,
        action_id: str,
        implemented_by: str,
        linked_training_snapshot: Optional[str] = None,
        linked_model_version: Optional[str] = None,
        implementation_notes: Optional[str] = None,
        actual_cost: Optional[float] = None,
    ) -> bool:
        """Mark action as implemented."""

        if action_id not in self.action_index:
            return False

        action = self.action_index[action_id]
        action.status = ActionStatus.COMPLETED
        action.implemented_by = implemented_by
        action.date_applied = datetime.now(timezone.utc).isoformat()
        action.linked_training_snapshot = linked_training_snapshot
        action.linked_model_version = linked_model_version
        action.actual_cost = actual_cost

        if implementation_notes:
            action.metadata["implementation_notes"] = implementation_notes

        return True

    def verify_action(
        self,
        action_id: str,
        verification_results: Dict[str, Any],
        effectiveness_score: Optional[float] = None,
        verifier: Optional[str] = None,
    ) -> bool:
        """Verify the effectiveness of a corrective action."""

        if action_id not in self.action_index:
            return False

        action = self.action_index[action_id]
        action.status = ActionStatus.VERIFIED
        action.date_verified = datetime.now(timezone.utc).isoformat()
        action.verification_results = verification_results
        action.effectiveness_score = effectiveness_score

        if verifier:
            action.metadata["verifier"] = verifier

        return True

    def get_actions_by_status(self, status: ActionStatus) -> List[CorrectiveAction]:
        """Get all actions with specific status."""
        return [action for action in self.actions if action.status == status]

    def get_actions_by_type(self, action_type: ActionType) -> List[CorrectiveAction]:
        """Get all actions of specific type."""
        return [action for action in self.actions if action.action_type == action_type]

    def get_actions_in_period(
        self, start_date: str, end_date: str
    ) -> List[CorrectiveAction]:
        """Get all actions created within a time period."""
        return [
            action
            for action in self.actions
            if start_date <= action.date_created <= end_date
        ]

    def generate_action_summary(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> CorrectiveActionSummary:
        """Generate summary of corrective actions."""

        if start_date and end_date:
            actions = self.get_actions_in_period(start_date, end_date)
        else:
            actions = self.actions
            start_date = min([a.date_created for a in actions]) if actions else ""
            end_date = max([a.date_created for a in actions]) if actions else ""

        if not actions:
            return CorrectiveActionSummary(
                total_actions=0,
                by_status={},
                by_type={},
                by_trigger={},
                avg_resolution_time_days=0.0,
                effectiveness_scores=[],
                total_cost=0.0,
                period_start=start_date,
                period_end=end_date,
            )

        # Count by status
        by_status = {}
        for status in ActionStatus:
            by_status[status.value] = len([a for a in actions if a.status == status])

        # Count by type
        by_type = {}
        for action_type in ActionType:
            by_type[action_type.value] = len(
                [a for a in actions if a.action_type == action_type]
            )

        # Count by trigger
        by_trigger = {}
        for trigger_type in TriggerType:
            by_trigger[trigger_type.value] = len(
                [a for a in actions if a.trigger_type == trigger_type]
            )

        # Calculate average resolution time
        completed_actions = [a for a in actions if a.date_applied and a.date_created]
        avg_resolution_time = 0.0
        if completed_actions:
            resolution_times = []
            for action in completed_actions:
                created = datetime.fromisoformat(
                    action.date_created.replace("Z", "+00:00")
                )
                applied = datetime.fromisoformat(
                    action.date_applied.replace("Z", "+00:00")
                )
                resolution_times.append((applied - created).days)
            avg_resolution_time = sum(resolution_times) / len(resolution_times)

        # Get effectiveness scores
        effectiveness_scores = [
            a.effectiveness_score for a in actions if a.effectiveness_score is not None
        ]

        # Calculate total cost
        total_cost = sum([a.actual_cost or 0 for a in actions])

        return CorrectiveActionSummary(
            total_actions=len(actions),
            by_status=by_status,
            by_type=by_type,
            by_trigger=by_trigger,
            avg_resolution_time_days=avg_resolution_time,
            effectiveness_scores=effectiveness_scores,
            total_cost=total_cost,
            period_start=start_date,
            period_end=end_date,
        )

    def export_action_log(
        self, format: str = "json", include_metadata: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """Export corrective action log."""

        if include_metadata:
            data = [action.to_dict() for action in self.actions]
        else:
            # Export minimal data for public reporting
            data = []
            for action in self.actions:
                minimal = {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "trigger_type": action.trigger_type.value,
                    "status": action.status.value,
                    "date_created": action.date_created,
                    "date_applied": action.date_applied,
                    "effectiveness_score": action.effectiveness_score,
                }
                data.append(minimal)

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return data

    def create_compliance_metadata(self) -> Dict[str, Any]:
        """Create corrective action metadata for compliance reporting."""

        summary = self.generate_action_summary()

        return {
            "corrective_action_log": {
                "enabled": True,
                "total_actions": summary.total_actions,
                "model_name": self.model_name,
                "summary": {
                    "by_status": summary.by_status,
                    "by_type": summary.by_type,
                    "avg_resolution_days": summary.avg_resolution_time_days,
                    "total_cost": summary.total_cost,
                    "avg_effectiveness": (
                        sum(summary.effectiveness_scores)
                        / len(summary.effectiveness_scores)
                        if summary.effectiveness_scores
                        else 0.0
                    ),
                },
                "recent_actions": [
                    {
                        "action_id": action.action_id,
                        "trigger": action.trigger,
                        "action_type": action.action_type.value,
                        "status": action.status.value,
                        "date_created": action.date_created,
                        "linked_training_snapshot": action.linked_training_snapshot,
                        "approver": action.approved_by,
                        "approved_date": action.date_created,  # Using creation date as approved date
                        "cost": {
                            "estimated_cost": action.cost_estimate or 0.0,
                            "actual_cost": action.actual_cost or 0.0,
                            "currency": "USD",
                            "cost_breakdown": {
                                "data_collection": (
                                    action.actual_cost or action.cost_estimate or 0.0
                                )
                                * 0.4,
                                "model_retraining": (
                                    action.actual_cost or action.cost_estimate or 0.0
                                )
                                * 0.4,
                                "validation_testing": (
                                    action.actual_cost or action.cost_estimate or 0.0
                                )
                                * 0.2,
                            },
                            "budget_variance": (action.actual_cost or 0.0)
                            - (action.cost_estimate or 0.0),
                            "budget_variance_percentage": (
                                (
                                    (
                                        (action.actual_cost or 0.0)
                                        - (action.cost_estimate or 0.0)
                                    )
                                    / (action.cost_estimate or 1.0)
                                )
                                * 100
                                if action.cost_estimate
                                else 0.0
                            ),
                        },
                    }
                    for action in sorted(
                        self.actions, key=lambda x: x.date_created, reverse=True
                    )[:5]
                ],
                "compliance_alignment": {
                    "eu_ai_act": "Article 15 - Transparency and corrective action tracking",
                    "nist_ai_rmf": "Manage function - Risk response and remediation",
                    "iso_27001": "A.16.1 - Management of information security incidents",
                },
                "metadata_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }


# Example usage and demonstration
def demo_corrective_action_log():
    """Demonstrate corrective action logging capabilities."""

    print("\nCORRECTIVE ACTION LOG DEMO")
    print("=" * 50)

    logger = CorrectiveActionLogger("JobClassificationModel_v2.1")

    # Create sample corrective actions
    print("1. Creating Corrective Actions")

    # Bias drift action
    bias_action = logger.create_action(
        trigger="Bias drift detected in gender classification",
        trigger_type=TriggerType.BIAS_DRIFT,
        detection_method="Fairness audit score drop > 15%",
        action_type=ActionType.BIAS_CORRECTION,
        description="Retrained with expanded balanced dataset",
        approved_by="Chief Compliance Officer",
        priority="High",
        evidence_files=["fairness_audit_before_after.pdf", "bias_metrics_Q3_2025.json"],
        verification_criteria=["Fairness score > 0.85", "Bias score < 0.05"],
        cost_estimate=25000.0,
    )

    print(f"   Created Bias Correction Action: {bias_action.action_id}")

    # Security patch action
    security_action = logger.create_action(
        trigger="Adversarial vulnerability detected",
        trigger_type=TriggerType.SECURITY_INCIDENT,
        detection_method="Automated adversarial testing",
        action_type=ActionType.SECURITY_PATCH,
        description="Applied adversarial training and input validation",
        approved_by="Chief Information Security Officer",
        priority="Critical",
        evidence_files=["security_scan_report.pdf"],
        verification_criteria=["Adversarial robustness > 0.8"],
        cost_estimate=15000.0,
    )

    print(f"   Created Security Patch Action: {security_action.action_id}")

    # Approve and implement actions
    print("\n2. Action Lifecycle Management")

    # Approve bias action
    logger.approve_action(bias_action.action_id, "Chief Compliance Officer")
    print(f"Approved action: {bias_action.action_id}")

    # Implement bias action
    logger.implement_action(
        bias_action.action_id,
        implemented_by="Senior ML Engineer",
        linked_training_snapshot="0764fec415c6d27c359bcd5a3248a1d13e9790fafa665e4205cc430b0f1846d1",
        linked_model_version="v2.2",
        actual_cost=23500.0,
    )
    print(f"Implemented action: {bias_action.action_id}")

    # Verify bias action
    verification_results = {
        "fairness_score_before": 0.72,
        "fairness_score_after": 0.89,
        "bias_score_before": 0.18,
        "bias_score_after": 0.04,
        "verification_date": datetime.now(timezone.utc).isoformat(),
        "verified_by": "Compliance Auditor",
    }

    logger.verify_action(
        bias_action.action_id,
        verification_results=verification_results,
        effectiveness_score=0.94,
        verifier="Compliance Auditor",
    )
    print(f"Verified action: {bias_action.action_id} (Effectiveness: 94%)")

    # Generate summary
    print("\n3. Action Summary and Analytics")
    summary = logger.generate_action_summary()

    print(f"   Total Actions: {summary.total_actions}")
    print(f"   Completed Actions: {summary.by_status.get('Completed', 0)}")
    print(f"   Verified Actions: {summary.by_status.get('Verified', 0)}")
    print(f"   Average Resolution Time: {summary.avg_resolution_time_days:.1f} days")
    print(f"   Total Cost: ${summary.total_cost:,.2f}")
    if summary.effectiveness_scores:
        avg_effectiveness = sum(summary.effectiveness_scores) / len(
            summary.effectiveness_scores
        )
        print(f"   Average Effectiveness: {avg_effectiveness:.1%}")

    # Export compliance metadata
    print("\n4. Compliance Metadata Export")
    metadata = logger.create_compliance_metadata()
    print("Corrective action metadata prepared for compliance documentation")

    # Show recent actions
    print(
        f"\n5. Recent Actions ({len(metadata['corrective_action_log']['recent_actions'])})"
    )
    for action in metadata["corrective_action_log"]["recent_actions"]:
        print(
            f"   • {action['action_id']}: {action['action_type']} ({action['status']})"
        )

    return logger, metadata


if __name__ == "__main__":
    demo_corrective_action_log()
