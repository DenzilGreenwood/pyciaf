"""
CIAF Agentic Events - First-Class Agent Event Model

Defines formal event types and data models for autonomous agent actions,
parallel to the Web AI event model but focused on agent execution.

This bridges the gap between authorization (ActionRequest/ActionReceipt)
and governance events that can be stored, queried, and analyzed.

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from ..core import sha256_hash


class AgentEventType(str, Enum):
    """
    First-class agent governance event types.

    These represent categories of agent behavior that require
    governance, audit, and compliance tracking.
    """

    # Data operations
    READ = "agent_read"
    WRITE = "agent_write"
    DELETE = "agent_delete"
    SEARCH = "agent_search"
    EXPORT = "agent_export"

    # External interactions
    API_CALL = "agent_api_call"
    HTTP_REQUEST = "agent_http_request"
    DATABASE_QUERY = "agent_database_query"
    FILE_ACCESS = "agent_file_access"

    # Autonomous behavior
    DECISION = "agent_decision"
    REASONING = "agent_reasoning"
    PLAN_GENERATION = "agent_plan"
    GOAL_UPDATE = "agent_goal_update"

    # Governance & control
    POLICY_CHECK = "agent_policy_check"
    ELEVATION_REQUEST = "agent_elevation_request"
    HUMAN_OVERRIDE = "agent_human_override"
    APPROVAL_REQUEST = "agent_approval_request"

    # Tool & function usage
    TOOL_CALL = "agent_tool_call"
    FUNCTION_EXECUTION = "agent_function_execution"

    # Inter-agent
    AGENT_MESSAGE = "agent_message"
    AGENT_DELEGATION = "agent_delegation"

    # System events
    SESSION_START = "agent_session_start"
    SESSION_END = "agent_session_end"
    ERROR = "agent_error"


class AgentActionType(str, Enum):
    """
    Formal action types for agent operations.

    Replaces string-typed actions with structured enums
    for better type safety and governance.
    """

    # Data access
    READ_RECORD = "read_record"
    WRITE_RECORD = "write_record"
    UPDATE_RECORD = "update_record"
    DELETE_RECORD = "delete_record"
    SEARCH_RECORDS = "search_records"
    EXPORT_DATA = "export_data"

    # Approvals & workflows
    APPROVE_PAYMENT = "approve_payment"
    APPROVE_CLAIM = "approve_claim"
    APPROVE_CHANGE = "approve_change"
    REJECT_REQUEST = "reject_request"

    # Infrastructure & deployment
    DEPLOY_CODE = "deploy_code"
    ROLLBACK_DEPLOY = "rollback_deploy"
    UPDATE_CONFIG = "update_config"
    RESTART_SERVICE = "restart_service"

    # Database operations
    DATABASE_QUERY = "database_query"
    DATABASE_WRITE = "database_write"
    DATABASE_BACKUP = "database_backup"

    # API operations
    EXTERNAL_API_CALL = "external_api_call"
    INTERNAL_API_CALL = "internal_api_call"

    # Sensitive data
    ACCESS_PHI = "access_phi"
    ACCESS_PII = "access_pii"
    ACCESS_SECRETS = "access_secrets"

    # Custom (extensible)
    CUSTOM = "custom"


class PolicyDecision(str, Enum):
    """Policy evaluation results for agent actions."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_ELEVATION = "require_elevation"
    REQUIRE_APPROVAL = "require_approval"
    WARN = "warn"
    NOT_EVALUATED = "not_evaluated"


class SensitivityLevel(str, Enum):
    """Data sensitivity classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_RESTRICTED = "highly_restricted"


@dataclass
class AgentEvent:
    """
    First-class agent governance event.

    Parallel to WebAIEvent but focused on autonomous agent execution.
    Captures not just web interactions but any agent action requiring
    governance, audit, and compliance tracking.
    """

    # Primary identifiers
    event_id: str
    event_type: AgentEventType
    occurred_at: str  # ISO 8601 timestamp

    # Agent identity
    agent_id: str
    agent_name: str
    principal_type: str  # "agent", "human", "service", "system"
    session_id: str

    # Organizational context
    org_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: Optional[str] = None  # dev/staging/prod

    # Action details
    action: str  # Can be AgentActionType or custom string
    resource_type: str
    resource_id: str

    # Action context
    params: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    correlation_id: Optional[str] = None

    # Authorization & policy
    policy_decision: PolicyDecision = PolicyDecision.NOT_EVALUATED
    policy_rule_id: Optional[str] = None
    policy_reason: Optional[str] = None
    elevation_grant_id: Optional[str] = None
    approved_by: Optional[str] = None

    # Data classification
    sensitivity_level: Optional[SensitivityLevel] = None
    data_classification: Optional[str] = None

    # Execution outcome
    executed: bool = False
    success: bool = False
    error_message: Optional[str] = None

    # Privacy-preserving hashes
    params_hash: str = ""
    input_hash: str = ""
    output_hash: str = ""

    # Cryptographic evidence
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None

    # Hash chaining (audit trail)
    prior_event_hash: str = "0" * 64  # Genesis hash for first event

    # Compliance & obligations
    compliance_frameworks: List[str] = field(default_factory=list)
    policy_obligations: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        event_type: AgentEventType,
        agent_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        agent_name: str = "",
        session_id: str = "",
        **kwargs: Any,
    ) -> AgentEvent:
        """
        Create a new AgentEvent with auto-generated ID and timestamp.

        Args:
            event_type: Type of agent event
            agent_id: Agent identifier
            action: Action being performed
            resource_type: Type of resource
            resource_id: Resource identifier
            agent_name: Display name for agent
            session_id: Session identifier
            **kwargs: Additional event attributes

        Returns:
            AgentEvent instance
        """
        return cls(
            event_id=f"aevt-{uuid.uuid4().hex[:16]}",
            event_type=event_type,
            occurred_at=utc_now_iso(),
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
            principal_type=kwargs.pop("principal_type", "agent"),
            session_id=session_id or f"sess-{uuid.uuid4().hex[:12]}",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result: Dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "principal_type": self.principal_type,
            "session_id": self.session_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
        }

        # Optional organizational context
        if self.org_id:
            result["org_id"] = self.org_id
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        if self.environment:
            result["environment"] = self.environment

        # Action context
        if self.params:
            result["params"] = self.params
        if self.justification:
            result["justification"] = self.justification
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id

        # Policy & authorization
        if self.policy_decision != PolicyDecision.NOT_EVALUATED:
            result["policy_decision"] = self.policy_decision.value
        if self.policy_rule_id:
            result["policy_rule_id"] = self.policy_rule_id
        if self.policy_reason:
            result["policy_reason"] = self.policy_reason
        if self.elevation_grant_id:
            result["elevation_grant_id"] = self.elevation_grant_id
        if self.approved_by:
            result["approved_by"] = self.approved_by

        # Classification
        if self.sensitivity_level:
            result["sensitivity_level"] = self.sensitivity_level.value
        if self.data_classification:
            result["data_classification"] = self.data_classification

        # Execution
        result["executed"] = self.executed
        result["success"] = self.success
        if self.error_message:
            result["error_message"] = self.error_message

        # Hashes
        if self.params_hash:
            result["params_hash"] = self.params_hash
        if self.input_hash:
            result["input_hash"] = self.input_hash
        if self.output_hash:
            result["output_hash"] = self.output_hash

        # Evidence
        if self.signature:
            result["signature"] = self.signature
        if self.signature_algorithm:
            result["signature_algorithm"] = self.signature_algorithm
        result["prior_event_hash"] = self.prior_event_hash

        # Compliance
        if self.compliance_frameworks:
            result["compliance_frameworks"] = self.compliance_frameworks
        if self.policy_obligations:
            result["policy_obligations"] = self.policy_obligations

        # Metadata
        if self.metadata:
            result["metadata"] = self.metadata
        if self.tags:
            result["tags"] = self.tags

        return result

    def get_event_hash(self) -> str:
        """
        Generate cryptographic hash of this event.

        Used for hash chaining and tamper detection.
        """
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "agent_id": self.agent_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "params_hash": self.params_hash,
            "policy_decision": self.policy_decision.value,
            "executed": self.executed,
            "success": self.success,
            "prior_event_hash": self.prior_event_hash,
        }
        return sha256_hash(str(event_data).encode('utf-8'))

    def requires_elevation(self) -> bool:
        """Check if action required privilege elevation."""
        return self.elevation_grant_id is not None

    def was_approved(self) -> bool:
        """Check if action required and received approval."""
        return self.approved_by is not None

    def is_sensitive(self) -> bool:
        """Check if event involves sensitive data."""
        if self.sensitivity_level in [
            SensitivityLevel.RESTRICTED,
            SensitivityLevel.HIGHLY_RESTRICTED,
        ]:
            return True
        if self.action in ["access_phi", "access_pii", "access_secrets"]:
            return True
        return False

    def is_high_risk(self) -> bool:
        """Check if event is high-risk."""
        high_risk_actions = {
            "delete_record",
            "database_write",
            "deploy_code",
            "approve_payment",
            "access_phi",
            "export_data",
        }
        return self.action in high_risk_actions or self.is_sensitive()


@dataclass
class AgentEventBatch:
    """
    Batch of agent events for efficient processing.

    Used for bulk operations like batch storage or analysis.
    """

    batch_id: str
    events: List[AgentEvent]
    created_at: str
    agent_id: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        events: List[AgentEvent],
        agent_id: str,
        session_id: Optional[str] = None,
    ) -> AgentEventBatch:
        """Create a new event batch."""
        return cls(
            batch_id=f"batch-{uuid.uuid4().hex[:16]}",
            events=events,
            created_at=utc_now_iso(),
            agent_id=agent_id,
            session_id=session_id,
        )

    def count(self) -> int:
        """Count events in batch."""
        return len(self.events)

    def filter_by_type(self, event_type: AgentEventType) -> List[AgentEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def filter_high_risk(self) -> List[AgentEvent]:
        """Get high-risk events."""
        return [e for e in self.events if e.is_high_risk()]

    def filter_sensitive(self) -> List[AgentEvent]:
        """Get sensitive data events."""
        return [e for e in self.events if e.is_sensitive()]


def utc_now_iso() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AgentEvent",
    "AgentEventBatch",
    "AgentEventType",
    "AgentActionType",
    "PolicyDecision",
    "SensitivityLevel",
    "utc_now_iso",
]
