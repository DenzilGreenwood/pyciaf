"""
Audit Trail Generation for CIAF Compliance

This module generates comprehensive audit trails for AI systems, tracking all
training, inference, and model lifecycle events with cryptographic integrity.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core import CryptoUtils
from ..inference import InferenceReceipt
from ..provenance import TrainingSnapshot


class AuditEventType(Enum):
    """Types of audit events tracked by CIAF."""

    MODEL_TRAINING = "model_training"
    DATA_INGESTION = "data_ingestion"
    INFERENCE_REQUEST = "inference_request"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_UPDATE = "model_update"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    USER_ACCESS = "user_access"


@dataclass
class ComplianceAuditRecord:
    """Individual audit record with compliance metadata."""

    # Core audit fields
    event_id: str
    event_type: AuditEventType
    timestamp: str
    model_name: str
    model_version: str
    user_id: Optional[str] = None

    # Event-specific data
    event_data: Dict[str, Any] = None

    # Compliance metadata
    regulatory_frameworks: List[str] = None
    risk_level: str = "low"
    compliance_status: str = "compliant"

    # Cryptographic integrity
    data_hash: str = ""
    previous_hash: str = ""
    audit_hash: str = ""

    # Privacy and security
    contains_pii: bool = False
    encryption_used: bool = True
    access_controls: List[str] = None

    def __post_init__(self):
        """Compute hashes after initialization."""
        if not self.event_data:
            self.event_data = {}
        if not self.regulatory_frameworks:
            self.regulatory_frameworks = []
        if not self.access_controls:
            self.access_controls = []

        # Compute data hash
        data_str = json.dumps(self.event_data, sort_keys=True)
        self.data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # Compute audit record hash
        audit_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
        }
        audit_str = json.dumps(audit_data, sort_keys=True)
        self.audit_hash = hashlib.sha256(audit_str.encode()).hexdigest()


class AuditTrailGenerator:
    """Generates comprehensive audit trails for CIAF systems."""

    def __init__(self, model_name: str, compliance_frameworks: List[str] = None):
        """
        Initialize audit trail generator.

        Args:
            model_name: Name of the model being audited
            compliance_frameworks: List of regulatory frameworks to track
        """
        self.model_name = model_name
        self.compliance_frameworks = compliance_frameworks or ["general"]
        self.crypto_utils = CryptoUtils()
        self.audit_records: List[ComplianceAuditRecord] = []
        self.last_hash = ""

    def record_training_event(
        self,
        training_snapshot: TrainingSnapshot,
        training_params: Dict[str, Any],
        user_id: str = "system",
    ) -> ComplianceAuditRecord:
        """Record a model training event."""

        event_data = {
            "training_snapshot_id": training_snapshot.snapshot_id,
            "merkle_root": training_snapshot.merkle_root_hash,
            "training_params": training_params,
            "data_capsule_count": len(training_snapshot.provenance_capsule_hashes),
            "training_duration": training_params.get("training_duration", "unknown"),
            "dataset_fingerprint": training_params.get("dataset_fingerprint", ""),
            "model_architecture": training_params.get("model_architecture", "unknown"),
        }

        record = ComplianceAuditRecord(
            event_id=f"training_{int(datetime.now().timestamp() * 1000)}",
            event_type=AuditEventType.MODEL_TRAINING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.model_name,
            model_version=training_snapshot.model_version,
            user_id=user_id,
            event_data=event_data,
            regulatory_frameworks=self.compliance_frameworks,
            risk_level=self._assess_training_risk(training_params),
            previous_hash=self.last_hash,
            contains_pii=training_params.get("contains_pii", False),
            access_controls=["authenticated_users", "training_role"],
        )

        self._add_record(record)
        return record

    def record_inference_event(
        self,
        receipt: InferenceReceipt,
        query_metadata: Dict[str, Any] = None,
        user_id: str = "anonymous",
    ) -> ComplianceAuditRecord:
        """Record an inference event."""

        query_metadata = query_metadata or {}

        event_data = {
            "receipt_hash": receipt.receipt_hash,
            "training_snapshot_id": receipt.training_snapshot_id,
            "query_hash": hashlib.sha256(receipt.query.encode()).hexdigest(),
            "output_hash": hashlib.sha256(receipt.ai_output.encode()).hexdigest(),
            "model_version": receipt.model_version,
            "connected_to_previous": receipt.prev_receipt_hash is not None,
            "metadata": query_metadata,
        }

        record = ComplianceAuditRecord(
            event_id=f"inference_{receipt.receipt_hash[:16]}",
            event_type=AuditEventType.INFERENCE_REQUEST,
            timestamp=receipt.timestamp,
            model_name=self.model_name,
            model_version=receipt.model_version,
            user_id=user_id,
            event_data=event_data,
            regulatory_frameworks=self.compliance_frameworks,
            risk_level=self._assess_inference_risk(query_metadata),
            previous_hash=self.last_hash,
            contains_pii=query_metadata.get("contains_pii", False),
            access_controls=["authenticated_users", "inference_role"],
        )

        self._add_record(record)
        return record

    def record_data_access_event(
        self,
        dataset_id: str,
        access_type: str,
        user_id: str,
        data_summary: Dict[str, Any] = None,
    ) -> ComplianceAuditRecord:
        """Record a data access event."""

        data_summary = data_summary or {}

        event_data = {
            "dataset_id": dataset_id,
            "access_type": access_type,
            "record_count": data_summary.get("record_count", 0),
            "data_types": data_summary.get("data_types", []),
            "purpose": data_summary.get("purpose", "training"),
            "retention_period": data_summary.get("retention_period", "as_needed"),
        }

        record = ComplianceAuditRecord(
            event_id=f"data_access_{int(datetime.now().timestamp() * 1000)}",
            event_type=AuditEventType.DATA_ACCESS,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.model_name,
            model_version="current",
            user_id=user_id,
            event_data=event_data,
            regulatory_frameworks=self.compliance_frameworks,
            risk_level=self._assess_data_access_risk(access_type, data_summary),
            previous_hash=self.last_hash,
            contains_pii=data_summary.get("contains_pii", False),
            access_controls=[
                "authenticated_users",
                "data_access_role",
                "purpose_limited",
            ],
        )

        self._add_record(record)
        return record

    def record_compliance_check(
        self, check_type: str, results: Dict[str, Any], user_id: str = "system"
    ) -> ComplianceAuditRecord:
        """Record a compliance validation check."""

        event_data = {
            "check_type": check_type,
            "check_results": results,
            "frameworks_checked": self.compliance_frameworks,
            "passed_checks": results.get("passed", []),
            "failed_checks": results.get("failed", []),
            "warnings": results.get("warnings", []),
            "overall_status": results.get("overall_status", "unknown"),
        }

        record = ComplianceAuditRecord(
            event_id=f"compliance_{check_type}_{int(datetime.now().timestamp() * 1000)}",
            event_type=AuditEventType.COMPLIANCE_CHECK,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.model_name,
            model_version="current",
            user_id=user_id,
            event_data=event_data,
            regulatory_frameworks=self.compliance_frameworks,
            risk_level=self._assess_compliance_risk(results),
            previous_hash=self.last_hash,
            contains_pii=False,
            access_controls=["compliance_officers", "audit_role"],
        )

        self._add_record(record)
        return record

    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> List[ComplianceAuditRecord]:
        """Get filtered audit trail."""

        filtered_records = self.audit_records

        if start_date:
            filtered_records = [
                r
                for r in filtered_records
                if datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
                >= start_date
            ]

        if end_date:
            filtered_records = [
                r
                for r in filtered_records
                if datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
                <= end_date
            ]

        if event_types:
            filtered_records = [
                r for r in filtered_records if r.event_type in event_types
            ]

        return filtered_records

    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify the cryptographic integrity of the audit trail."""

        verification_results = {
            "total_records": len(self.audit_records),
            "integrity_verified": True,
            "broken_connections": [],
            "hash_mismatches": [],
            "timestamp_issues": [],
        }

        previous_hash = ""
        previous_timestamp = None

        for i, record in enumerate(self.audit_records):
            # Check hash connections
            if record.previous_hash != previous_hash:
                verification_results["integrity_verified"] = False
                verification_results["broken_connections"].append(
                    {
                        "record_index": i,
                        "event_id": record.event_id,
                        "expected_previous": previous_hash,
                        "actual_previous": record.previous_hash,
                    }
                )

            # Check timestamp ordering
            current_timestamp = datetime.fromisoformat(
                record.timestamp.replace("Z", "+00:00")
            )
            if previous_timestamp and current_timestamp < previous_timestamp:
                verification_results["timestamp_issues"].append(
                    {
                        "record_index": i,
                        "event_id": record.event_id,
                        "timestamp": record.timestamp,
                    }
                )

            # Verify record hash
            expected_hash = self._compute_record_hash(record)
            if expected_hash != record.audit_hash:
                verification_results["integrity_verified"] = False
                verification_results["hash_mismatches"].append(
                    {
                        "record_index": i,
                        "event_id": record.event_id,
                        "expected_hash": expected_hash,
                        "actual_hash": record.audit_hash,
                    }
                )

            previous_hash = record.audit_hash
            previous_timestamp = current_timestamp

        return verification_results

    def export_audit_trail(self, format: str = "json") -> str:
        """Export audit trail in specified format."""

        if format.lower() == "json":
            records_dict = [asdict(record) for record in self.audit_records]
            # Convert enum to string
            for record_dict in records_dict:
                record_dict["event_type"] = record_dict["event_type"].value

            return json.dumps(
                {
                    "model_name": self.model_name,
                    "compliance_frameworks": self.compliance_frameworks,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_records": len(self.audit_records),
                    "audit_records": records_dict,
                },
                indent=2,
            )

        elif format.lower() == "csv":
            # TODO: Implement CSV export
            raise NotImplementedError("CSV export not yet implemented")

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _add_record(self, record: ComplianceAuditRecord) -> None:
        """Add a record to the audit trail."""
        self.audit_records.append(record)
        self.last_hash = record.audit_hash

    def _assess_training_risk(self, training_params: Dict[str, Any]) -> str:
        """Assess risk level for training events."""
        if training_params.get("contains_pii", False):
            return "high"
        elif training_params.get("public_dataset", True):
            return "low"
        else:
            return "medium"

    def _assess_inference_risk(self, query_metadata: Dict[str, Any]) -> str:
        """Assess risk level for inference events."""
        if query_metadata.get("contains_pii", False):
            return "high"
        elif query_metadata.get("sensitive_domain", False):
            return "medium"
        else:
            return "low"

    def _assess_data_access_risk(
        self, access_type: str, data_summary: Dict[str, Any]
    ) -> str:
        """Assess risk level for data access events."""
        if access_type in ["export", "download"] and data_summary.get(
            "contains_pii", False
        ):
            return "high"
        elif access_type in ["modify", "delete"]:
            return "medium"
        else:
            return "low"

    def _assess_compliance_risk(self, results: Dict[str, Any]) -> str:
        """Assess risk level for compliance check events."""
        if results.get("failed", []):
            return "high"
        elif results.get("warnings", []):
            return "medium"
        else:
            return "low"

    def _compute_record_hash(self, record: ComplianceAuditRecord) -> str:
        """Recompute hash for a record to verify integrity."""
        audit_data = {
            "event_id": record.event_id,
            "event_type": record.event_type.value,
            "timestamp": record.timestamp,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "data_hash": record.data_hash,
            "previous_hash": record.previous_hash,
        }
        audit_str = json.dumps(audit_data, sort_keys=True)
        return hashlib.sha256(audit_str.encode()).hexdigest()


class AuditTrail:
    """
    Simplified audit trail interface for model integration
    """

    def __init__(self, model_id: str, compliance_frameworks: List[str] = None):
        """Initialize audit trail for a specific model"""
        self.model_id = model_id
        self.generator = AuditTrailGenerator(model_id, compliance_frameworks)

    def log_event(self, event_type: str, details: str, metadata: Dict[str, Any] = None):
        """Log an audit event"""
        metadata = metadata or {}

        # Create a compliance check record for general events
        results = {
            "event_details": details,
            "metadata": metadata,
            "overall_status": "logged",
        }

        self.generator.record_compliance_check(
            check_type=event_type,
            results=results,
            user_id=metadata.get("user_id", "system"),
        )

    def get_records(self):
        """Get all audit records"""
        return self.generator.get_audit_trail()

    def export_trail(self, format: str = "json"):
        """Export audit trail"""
        return self.generator.export_audit_trail(format)


# Backwards compatibility alias
AuditTrailGenerator = AuditTrailGenerator
