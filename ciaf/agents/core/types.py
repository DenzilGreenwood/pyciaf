"""
Core data types for CIAF Agentic Execution Boundaries.

Defines Pydantic models for identity, resources, permissions, and execution tracking
that integrate with CIAF's cryptographic provenance system.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models with schema validation
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict

# SHA-256 hash pattern from schema
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class PrincipalType(str, Enum):
    """Types of principals that can perform actions."""

    AGENT = "agent"
    HUMAN = "human"
    SERVICE = "service"
    SYSTEM = "system"


class Identity(BaseModel):
    """
    Immutable identity for an agent or human principal.

    Integrates with CIAF's cryptographic anchoring system.
    Schema: identity.schema.json
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True, extra="forbid")

    principal_id: str = Field(..., description="Unique principal identifier")
    principal_type: PrincipalType = Field(..., description="Type of principal")
    display_name: str = Field(..., description="Human-readable display name")
    roles: Set[str] = Field(default_factory=set, description="Assigned roles")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional principal attributes"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant/organization identifier")
    environment: Optional[str] = Field(
        None, description="Environment where principal operates"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

    @computed_field
    @property
    def attributes_dict(self) -> Dict[str, Any]:
        """Get attributes as a dictionary."""
        return dict(self.attributes)

    def get_fingerprint(self) -> str:
        """Generate cryptographic fingerprint of this identity."""
        # Handle both PrincipalType enum and string
        principal_type_str = (
            self.principal_type.value
            if isinstance(self.principal_type, PrincipalType)
            else str(self.principal_type)
        )
        try:
            from ciaf.core.crypto import CryptoUtils 
        
            identity_data = {
                "principal_id": self.principal_id,
                "principal_type": principal_type_str,
                "roles": sorted(list(self.roles)),
                "tenant_id": self.tenant_id,
                "environment": self.environment,
            }
        except ImportError:
            # Fallback if core module is not available (e.g., during initial development)
            print("Warning: CryptoUtils not available, using fallback hash for identity fingerprint.")
        
        return CryptoUtils.sha256_hash(str(identity_data).encode("utf-8"))


class Resource(BaseModel):
    """
    A resource that can be accessed or modified.

    Resources are CIAF-tracked entities with cryptographic identity.
    Schema: resource.schema.json
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True, extra="forbid")

    resource_id: str = Field(..., description="Unique resource identifier")
    resource_type: str = Field(
        ..., description="Resource type (e.g., model, dataset, file)"
    )
    owner_tenant: Optional[str] = Field(None, description="Owning tenant identifier")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Resource attributes"
    )
    sensitivity_level: str = Field(
        default="standard",
        description="Resource sensitivity classification (standard, sensitive, critical, confidential)",
    )

    @computed_field
    @property
    def attributes_dict(self) -> Dict[str, Any]:
        """Get attributes as a dictionary."""
        return dict(self.attributes)


class Permission(BaseModel):
    """
    A permission granting an action on a resource type.

    Supports both RBAC and ABAC through condition functions.
    Schema: permission.schema.json
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Required for Callable field
    )

    action: str = Field(
        ..., description="Permitted action (e.g., read, write, execute, delete)"
    )
    resource_type: str = Field(
        ..., description="Type of resource this permission applies to"
    )
    condition: Optional[Callable[[Identity, Resource], bool]] = Field(
        None,
        exclude=True,  # Don't serialize Callable to JSON
        description="ABAC condition function",
    )
    description: str = Field(
        default="", description="Human-readable description of permission"
    )
    requires_elevation: bool = Field(
        default=False,
        description="Whether this permission requires privilege elevation",
    )

    def allows(self, identity: Identity, resource: Resource) -> bool:
        """Check if this permission allows the action."""
        if self.condition is None:
            return True
        return self.condition(identity, resource)


class RoleDefinition(BaseModel):
    """
    A named collection of permissions.

    Roles are the primary RBAC mechanism in CIAF agents.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Required for Permission with Callable fields
    )

    name: str = Field(..., description="Role name")
    permissions: List[Permission] = Field(
        default_factory=list, description="List of permissions"
    )
    description: str = Field(default="", description="Human-readable description")
    inherits_from: Set[str] = Field(
        default_factory=set, description="Roles this role inherits from"
    )


class ActionRequest(BaseModel):
    """
    A request to perform an action on a resource.

    Central type for authorization and audit trails.
    Schema: action-request.schema.json
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    action: str = Field(
        ..., description="Action to perform (e.g., read, write, execute)"
    )
    resource: Resource = Field(..., description="Target resource")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )
    justification: str = Field(default="", description="Justification for the action")
    requested_by: Optional[Identity] = Field(None, description="Requesting principal")
    correlation_id: Optional[str] = Field(
        None, description="Correlation identifier for tracing"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Request timestamp",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

    def get_params_hash(self) -> str:
        """Get cryptographic hash of parameters."""
        from ciaf.core import sha256_hash
        return sha256_hash(str(sorted(self.params.items())).encode("utf-8"))


class ExecutionResult(BaseModel):
    """
    Result of an action execution attempt.

    Contains authorization decision and execution outcome.
    Schema: execution-result.schema.json
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    request: ActionRequest = Field(..., description="Original action request")
    allowed: bool = Field(..., description="Whether action was allowed")
    reason: str = Field(..., description="Reason for decision")
    executed: bool = Field(
        default=False, description="Whether action was actually executed"
    )
    result: Any = Field(None, description="Execution result data (any type)")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    elevation_grant_id: Optional[str] = Field(
        None, description="Elevation grant ID used if applicable"
    )
    policy_obligations: List[str] = Field(
        default_factory=list, description="Policy obligations that must be fulfilled"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Result timestamp",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")


class ElevationGrant(BaseModel):
    """
    A just-in-time privilege elevation grant.

    Implements PAM-style temporary privilege escalation.
    Schema: elevation-grant.schema.json
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    grant_id: str = Field(..., description="Unique grant identifier")
    principal_id: str = Field(
        ..., description="Principal receiving elevated privileges"
    )
    elevated_role: str = Field(..., description="Role being granted temporarily")
    scope: Dict[str, Any] = Field(
        default_factory=dict, description="Scope limitations for the grant"
    )
    approved_by: str = Field(
        ..., description="Approver principal ID"
    )  # Required per schema
    ticket_reference: str = Field(
        default="", description="Reference to approval ticket or workflow"
    )
    valid_from: str = Field(
        ..., description="Grant validity start time"
    )  # Required per schema
    valid_until: str = Field(
        ..., description="Grant expiration time"
    )  # Required per schema
    purpose: str = Field(default="", description="Purpose/justification for elevation")
    used_count: int = Field(
        default=0, ge=0, description="Number of times grant has been used"
    )
    max_uses: Optional[int] = Field(
        None, ge=1, description="Maximum number of uses (optional)"
    )

    @field_validator("valid_from", "valid_until")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

    def is_valid(self, now: Optional[datetime] = None) -> bool:
        """Check if this grant is currently valid."""
        if now is None:
            now = datetime.now(timezone.utc)

        valid_from_dt = datetime.fromisoformat(self.valid_from.replace("Z", "+00:00"))
        valid_until_dt = datetime.fromisoformat(self.valid_until.replace("Z", "+00:00"))

        time_valid = valid_from_dt <= now <= valid_until_dt

        if self.max_uses is not None:
            uses_valid = self.used_count < self.max_uses
        else:
            uses_valid = True

        return time_valid and uses_valid


class ActionReceipt(BaseModel):
    """
    Cryptographic receipt of an action execution.

    Integrates with CIAF's audit trail and evidence system.
    Schema: action-receipt.schema.json
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    receipt_id: str = Field(..., description="Unique receipt identifier")
    timestamp: str = Field(..., description="Execution timestamp")
    principal_id: str = Field(..., description="Principal who performed the action")
    principal_type: PrincipalType = Field(..., description="Type of principal")
    action: str = Field(..., description="Action performed")
    resource_id: str = Field(..., description="Resource identifier")
    resource_type: str = Field(..., description="Resource type")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")
    decision: bool = Field(
        default=False, description="Authorization decision (true=allowed)"
    )
    reason: str = Field(default="", description="Decision reason")
    elevation_grant_id: Optional[str] = Field(
        None, description="Elevation grant ID if privileges were elevated"
    )
    approved_by: Optional[str] = Field(
        None, description="Approver principal ID for elevated actions"
    )
    params_hash: str = Field(
        default="", description="SHA-256 hash of action parameters"
    )
    policy_obligations: List[str] = Field(
        default_factory=list, description="Policy obligations that must be fulfilled"
    )
    prior_receipt_hash: str = Field(
        default="0" * 64, description="Hash of prior receipt for chain linking"
    )
    # Signature envelope (follows common/signature-envelope.json)
    # Accepts both Dict (SignatureEnvelope) and str (legacy) for backward compatibility
    signature: Union[Dict[str, Any], str] = Field(
        default="", description="Cryptographic signature envelope or legacy signature"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}")

    @field_validator("params_hash", "prior_receipt_hash")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        """Validate SHA-256 hash format (64 character hex string)."""
        if v and v != "0" * 64:  # Allow genesis hash (all zeros)
            if not SHA256_PATTERN.match(v):
                raise ValueError(f"Invalid SHA-256 hash format: {v}")
        return v

    def get_receipt_hash(self) -> str:
        """
        Generate cryptographic hash of this receipt.

        Follows CIAF's hash-chaining pattern for audit trails.
        """
        receipt_data = {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "principal_id": self.principal_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "decision": self.decision,
            "params_hash": self.params_hash,
            "prior_receipt_hash": self.prior_receipt_hash,
        }
        from ciaf.core import sha256_hash
        return sha256_hash(str(receipt_data).encode("utf-8"))
