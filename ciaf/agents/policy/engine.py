"""
Policy Engine implementation for authorization decisions.

Combines IAM permissions, PAM elevation grants, and ABAC conditions
to produce comprehensive authorization decisions.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import List, Optional, Set

from ..core.interfaces import PolicyEvaluator
from ..core.types import (
    ActionRequest,
    ElevationGrant,
    ExecutionResult,
    Permission,
)
from ..iam import IAMStore
from ..pam import PAMStore


class PolicyEngine(PolicyEvaluator):
    """
    Central policy evaluation engine for CIAF agent authorization.

    Integrates:
    - IAM (Identity and Access Management) for standing permissions
    - PAM (Privileged Access Management) for elevation grants
    - ABAC (Attribute-Based Access Control) for contextual conditions
    - CIAF compliance frameworks for policy obligations
    """

    def __init__(
        self,
        iam_store: IAMStore,
        pam_store: PAMStore,
        sensitive_actions: Optional[Set[str]] = None,
        compliance_frameworks: Optional[List[str]] = None,
    ):
        """
        Initialize the policy engine.

        Args:
            iam_store: IAM store for identity and role management
            pam_store: PAM store for elevation grants
            sensitive_actions: Set of actions requiring elevation
            compliance_frameworks: List of applicable compliance frameworks
        """
        self._iam = iam_store
        self._pam = pam_store
        self._sensitive_actions = sensitive_actions or set()
        self._compliance_frameworks = compliance_frameworks or []

    def evaluate(self, request: ActionRequest) -> ExecutionResult:
        """
        Evaluate an action request against all policies.

        Decision flow:
        1. Resolve identity and check if it exists
        2. Check standing IAM permissions (RBAC + ABAC)
        3. If action requires elevation, check for active PAM grant
        4. Apply compliance policy obligations
        5. Return authorization decision

        Args:
            request: The action request to evaluate

        Returns:
            ExecutionResult with authorization decision
        """
        # Identity resolution
        if request.requested_by is None:
            return ExecutionResult(
                request=request,
                allowed=False,
                reason="No identity provided in request",
            )

        identity = self._iam.get_identity(request.requested_by.principal_id)
        if identity is None:
            return ExecutionResult(
                request=request,
                allowed=False,
                reason=f"Identity {request.requested_by.principal_id} not found",
            )

        # Check standing IAM permissions
        permissions = self._iam.get_permissions(identity.principal_id)
        matching_permissions = self._find_matching_permissions(
            permissions, request.action, request.resource.resource_type
        )

        # Evaluate ABAC conditions
        allowed_by_iam = False
        requires_elevation = False

        for perm in matching_permissions:
            # Check if permission has ABAC condition
            if perm.allows(identity, request.resource):
                if perm.requires_elevation:
                    requires_elevation = True
                else:
                    allowed_by_iam = True
                    break

        # Check if action is globally sensitive
        if request.action in self._sensitive_actions:
            requires_elevation = True

        # Handle elevation requirement
        grant_id = None
        if requires_elevation:
            # Look for active elevation grant
            grant = self._find_active_grant(identity.principal_id, request.action)
            if grant is None:
                return ExecutionResult(
                    request=request,
                    allowed=False,
                    reason=f"Action '{request.action}' requires privilege elevation. "
                    "No active grant found.",
                )
            grant_id = grant.grant_id
            allowed_by_iam = True  # Grant provides permission

        # Apply compliance obligations
        obligations = self._determine_obligations(request)

        # Final decision
        if allowed_by_iam:
            return ExecutionResult(
                request=request,
                allowed=True,
                reason="Allowed by IAM and runtime boundary policy",
                elevation_grant_id=grant_id,
                policy_obligations=obligations,
            )
        else:
            return ExecutionResult(
                request=request,
                allowed=False,
                reason=f"No permission for action '{request.action}' "
                f"on resource type '{request.resource.resource_type}'",
            )

    def requires_elevation(self, request: ActionRequest) -> bool:
        """
        Check if an action requires privilege elevation.

        Args:
            request: The action request

        Returns:
            True if elevation required, False otherwise
        """
        # Check if globally sensitive
        if request.action in self._sensitive_actions:
            return True

        # Check if any matching permission requires elevation
        if request.requested_by is None:
            return False

        permissions = self._iam.get_permissions(request.requested_by.principal_id)
        matching = self._find_matching_permissions(
            permissions, request.action, request.resource.resource_type
        )

        return any(perm.requires_elevation for perm in matching)

    def get_active_grant(
        self, principal_id: str, action: str
    ) -> Optional[ElevationGrant]:
        """
        Find an active elevation grant for a principal and action.

        Args:
            principal_id: The principal identifier
            action: The action being performed

        Returns:
            ElevationGrant if found and valid, None otherwise
        """
        return self._find_active_grant(principal_id, action)

    def _find_matching_permissions(
        self, permissions: List[Permission], action: str, resource_type: str
    ) -> List[Permission]:
        """Find permissions that match the action and resource type."""
        matching = []
        for perm in permissions:
            if perm.action == action and perm.resource_type == resource_type:
                matching.append(perm)
        return matching

    def _find_active_grant(
        self, principal_id: str, action: str
    ) -> Optional[ElevationGrant]:
        """
        Find an active grant that provides access to the action.

        For simplicity, we look for grants with elevated roles that
        would grant the required permission.
        """
        active_grants = self._pam.get_active_grants(principal_id)

        # In a full implementation, we would check if the elevated role
        # provides the required permission for the action.
        # For now, we return the first active grant if any exists.
        if active_grants:
            return active_grants[0]

        return None

    def _determine_obligations(self, request: ActionRequest) -> List[str]:
        """
        Determine policy obligations based on request context.

        Obligations are requirements that must be fulfilled alongside
        the authorization (e.g., enhanced logging, two-person review).

        Args:
            request: The action request

        Returns:
            List of obligation identifiers
        """
        obligations = []

        # High-sensitivity resources require enhanced logging
        if request.resource.sensitivity_level == "critical":
            obligations.append("heightened_logging")

        # Certain actions require two-person review
        if request.action in {"approve_payment", "delete_data", "modify_policy"}:
            obligations.append("two_person_review")

        # Compliance-specific obligations
        if "HIPAA" in self._compliance_frameworks:
            if request.resource.resource_type in {"patient_record", "phi_data"}:
                obligations.append("hipaa_audit_logging")

        if "SOX" in self._compliance_frameworks:
            if request.action in {"approve_transaction", "financial_reporting"}:
                obligations.append("sox_controls")

        return obligations

    def add_sensitive_action(self, action: str) -> None:
        """
        Mark an action as requiring elevation.

        Args:
            action: The action name
        """
        self._sensitive_actions.add(action)

    def remove_sensitive_action(self, action: str) -> None:
        """
        Unmark an action as requiring elevation.

        Args:
            action: The action name
        """
        self._sensitive_actions.discard(action)

    def get_sensitive_actions(self) -> Set[str]:
        """Get all actions marked as requiring elevation."""
        return self._sensitive_actions.copy()
