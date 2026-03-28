"""
Core data types for CIAF Agentic Execution Boundaries.

Defines dataclasses for identity, resources, permissions, and execution tracking
that integrate with CIAF's cryptographic provenance system.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from ...core import sha256_hash


class PrincipalType(str, Enum):
    """Types of principals that can perform actions."""

    AGENT = "agent"
    HUMAN = "human"
    SERVICE = "service"
    SYSTEM = "system"


@dataclass(frozen=True)
class Identity:
    """
    Immutable identity for an agent or human principal.

    Integrates with CIAF's cryptographic anchoring system.
    """

    principal_id: str
    principal_type: PrincipalType
    display_name: str
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    environment: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        """Ensure immutable collections are properly frozen."""
        if not isinstance(self.roles, frozenset):
            object.__setattr__(self, "roles", frozenset(self.roles))
        if isinstance(self.attributes, dict):
            object.__setattr__(self, "attributes", frozenset(self.attributes.items()))

    @property
    def attributes_dict(self) -> Dict[str, Any]:
        """Get attributes as a dictionary."""
        if isinstance(self.attributes, frozenset):
            return dict(self.attributes)
        return dict(self.attributes)

    def get_fingerprint(self) -> str:
        """Generate cryptographic fingerprint of this identity."""
        identity_data = {
            "principal_id": self.principal_id,
            "principal_type": self.principal_type.value,
            "roles": sorted(list(self.roles)),
            "tenant_id": self.tenant_id,
            "environment": self.environment,
        }
        return sha256_hash(str(identity_data))


@dataclass(frozen=True)
class Resource:
    """
    A resource that can be accessed or modified.

    Resources are CIAF-tracked entities with cryptographic identity.
    """

    resource_id: str
    resource_type: str
    owner_tenant: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    sensitivity_level: str = "standard"  # standard, sensitive, critical

    def __post_init__(self):
        """Ensure immutable attributes."""
        if isinstance(self.attributes, dict):
            object.__setattr__(self, "attributes", frozenset(self.attributes.items()))

    @property
    def attributes_dict(self) -> Dict[str, Any]:
        """Get attributes as a dictionary."""
        if isinstance(self.attributes, frozenset):
            return dict(self.attributes)
        return dict(self.attributes)


@dataclass
class Permission:
    """
    A permission granting an action on a resource type.

    Supports both RBAC and ABAC through condition functions.
    """

    action: str
    resource_type: str
    condition: Optional[Callable[[Identity, Resource], bool]] = None
    description: str = ""
    requires_elevation: bool = False

    def allows(self, identity: Identity, resource: Resource) -> bool:
        """Check if this permission allows the action."""
        if self.condition is None:
            return True
        return self.condition(identity, resource)


@dataclass
class RoleDefinition:
    """
    A named collection of permissions.

    Roles are the primary RBAC mechanism in CIAF agents.
    """

    name: str
    permissions: list[Permission] = field(default_factory=list)
    description: str = ""
    inherits_from: Set[str] = field(default_factory=set)


@dataclass
class ActionRequest:
    """
    A request to perform an action on a resource.

    Central type for authorization and audit trails.
    """

    action: str
    resource: Resource
    params: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    requested_by: Optional[Identity] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def get_params_hash(self) -> str:
        """Get cryptographic hash of parameters."""
        return sha256_hash(str(sorted(self.params.items())))


@dataclass
class ExecutionResult:
    """
    Result of an action execution attempt.

    Contains authorization decision and execution outcome.
    """

    request: ActionRequest
    allowed: bool
    reason: str
    executed: bool = False
    result: Any = None
    error: Optional[str] = None
    elevation_grant_id: Optional[str] = None
    policy_obligations: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ElevationGrant:
    """
    A just-in-time privilege elevation grant.

    Implements PAM-style temporary privilege escalation.
    """

    grant_id: str
    principal_id: str
    elevated_role: str
    scope: Dict[str, Any] = field(default_factory=dict)
    approved_by: str = ""
    ticket_reference: str = ""
    valid_from: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    valid_until: str = ""
    purpose: str = ""
    used_count: int = 0
    max_uses: Optional[int] = None

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


@dataclass
class ActionReceipt:
    """
    Cryptographic receipt of an action execution.

    Integrates with CIAF's audit trail and evidence system.
    """

    receipt_id: str
    timestamp: str
    principal_id: str
    principal_type: PrincipalType
    action: str
    resource_id: str
    resource_type: str
    correlation_id: Optional[str] = None
    decision: bool = False
    reason: str = ""
    elevation_grant_id: Optional[str] = None
    approved_by: Optional[str] = None
    params_hash: str = ""
    policy_obligations: list[str] = field(default_factory=list)
    prior_receipt_hash: str = "0" * 64
    signature: str = ""

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
        return sha256_hash(str(receipt_data))
