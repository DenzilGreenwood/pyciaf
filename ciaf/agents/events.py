"""
CIAF Agentic Events - First-Class Agent Event Model

Defines formal event types and data models for autonomous agent actions,
parallel to the Web AI event model but focused on agent execution.

This bridges the gap between authorization (ActionRequest/ActionReceipt)
and governance events that can be stored, queried, and analyzed.

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models with validation
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..core import sha256_hash

# SHA-256 hash pattern
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


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



class AgentEvent(BaseModel):
    """
    First-class agent governance event.

    Parallel to WebAIEvent but focused on autonomous agent execution.
    Captures not just web interactions but any agent action requiring
    governance, audit, and compliance tracking.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Primary identifiers (required)
    event_id: str = Field(..., description="Unique event identifier")
    event_type: AgentEventType = Field(..., description="Type of agent event")
    occurred_at: str = Field(..., description="ISO 8601 timestamp")

    # Agent identity (required)
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Display name for agent")
    principal_type: str = Field(..., description="Type: agent, human, service, system")
    session_id: str = Field(..., description="Session identifier")

    # Action details (required)
    action: str = Field(..., description="Action being performed (AgentActionType or custom)")
    resource_type: str = Field(..., description="Type of resource")
    resource_id: str = Field(..., description="Resource identifier")

    # Organizational context (optional)
    org_id: Optional[str] = Field(None, description="Organization identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    environment: Optional[str] = Field(None, description="Environment (dev/staging/prod)")

    # Action context (optional with defaults)
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    justification: str = Field(default="", description="Justification for action")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")

    # Authorization & policy (optional with defaults)
    policy_decision: PolicyDecision = Field(
        default=PolicyDecision.NOT_EVALUATED,
        description="Policy evaluation result"
    )
    policy_rule_id: Optional[str] = Field(None, description="Policy rule identifier")
    policy_reason: Optional[str] = Field(None, description="Policy decision reason")
    elevation_grant_id: Optional[str] = Field(None, description="Elevation grant ID if applicable")
    approved_by: Optional[str] = Field(None, description="Approver principal ID")

    # Data classification (optional)
    sensitivity_level: Optional[SensitivityLevel] = Field(None, description="Data sensitivity level")
    data_classification: Optional[str] = Field(None, description="Data classification label")

    # Execution outcome (optional with defaults)
    executed: bool = Field(default=False, description="Whether action was executed")
    success: bool = Field(default=False, description="Whether execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Privacy-preserving hashes
    params_hash: str = Field(default="", description="SHA-256 hash of parameters")
    input_hash: str = Field(default="", description="SHA-256 hash of input")
    output_hash: str = Field(default="", description="SHA-256 hash of output")

    # Cryptographic evidence
    signature: Optional[str] = Field(None, description="Cryptographic signature")
    signature_algorithm: Optional[str] = Field(None, description="Signature algorithm used")

    # Hash chaining (audit trail)
    prior_event_hash: str = Field(
        default="0" * 64,
        description="Hash of prior event for chain linking"
    )

    # Compliance & obligations
    compliance_frameworks: List[str] = Field(
        default_factory=list,
        description="Applicable compliance frameworks"
    )
    policy_obligations: List[str] = Field(
        default_factory=list,
        description="Policy obligations triggered"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Event tags")

    @field_validator('occurred_at')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

    @field_validator('params_hash', 'input_hash', 'output_hash', 'prior_event_hash')
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        """Validate SHA-256 hash format if provided."""
        if v and v != "0" * 64:  # Allow genesis hash
            if not SHA256_PATTERN.match(v):
                raise ValueError(f"Invalid SHA-256 hash format: {v}")
        return v

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
        # Use Pydantic's model_dump with exclude_none to omit None values
        return self.model_dump(mode='json', exclude_none=True)

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
        return sha256_hash(str(event_data).encode("utf-8"))

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



class AgentEventBatch(BaseModel):
    """
    Batch of agent events for efficient processing.

    Used for bulk operations like batch storage or analysis.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    batch_id: str = Field(..., description="Unique batch identifier")
    events: List[AgentEvent] = Field(..., description="List of agent events")
    created_at: str = Field(..., description="Batch creation timestamp")
    agent_id: str = Field(..., description="Primary agent ID")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")

    @field_validator('created_at')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

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
