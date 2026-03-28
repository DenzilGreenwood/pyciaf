"""
PAM Store implementation for privilege elevation management.

Handles just-in-time privilege grants with approval tracking,
time-bounds, and usage limits.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..core.types import ElevationGrant


class PAMStore:
    """
    Privileged Access Management store for JIT elevation grants.

    Manages temporary privilege escalation with approval workflows
    and comprehensive audit trails.
    """

    def __init__(self):
        """Initialize the PAM store with empty grant storage."""
        self._grants: Dict[str, ElevationGrant] = {}

    def create_grant(
        self,
        principal_id: str,
        elevated_role: str,
        duration_minutes: int = 60,
        approved_by: str = "",
        ticket_reference: str = "",
        purpose: str = "",
        scope: Optional[Dict] = None,
        max_uses: Optional[int] = None,
    ) -> ElevationGrant:
        """
        Create a new privilege elevation grant.

        Args:
            principal_id: The principal receiving elevation
            elevated_role: Role to elevate to
            duration_minutes: How long the grant is valid (default 60 minutes)
            approved_by: Identity that approved this elevation
            ticket_reference: External ticket/request ID
            purpose: Business justification for elevation
            scope: Optional scope restrictions (e.g., specific resources)
            max_uses: Maximum number of times this grant can be used

        Returns:
            The created ElevationGrant
        """
        now = datetime.now(timezone.utc)
        grant_id = f"grant-{uuid.uuid4().hex[:12]}"

        grant = ElevationGrant(
            grant_id=grant_id,
            principal_id=principal_id,
            elevated_role=elevated_role,
            scope=scope or {},
            approved_by=approved_by,
            ticket_reference=ticket_reference,
            valid_from=now.isoformat(),
            valid_until=(now + timedelta(minutes=duration_minutes)).isoformat(),
            purpose=purpose,
            used_count=0,
            max_uses=max_uses,
        )

        self._grants[grant_id] = grant
        return grant

    def get_grant(self, grant_id: str) -> Optional[ElevationGrant]:
        """
        Retrieve a grant by ID.

        Args:
            grant_id: The grant identifier

        Returns:
            ElevationGrant if found, None otherwise
        """
        return self._grants.get(grant_id)

    def get_active_grants(self, principal_id: str) -> List[ElevationGrant]:
        """
        Get all currently valid grants for a principal.

        Args:
            principal_id: The principal identifier

        Returns:
            List of active ElevationGrants
        """
        now = datetime.now(timezone.utc)
        active_grants = []

        for grant in self._grants.values():
            if grant.principal_id == principal_id and grant.is_valid(now):
                active_grants.append(grant)

        return active_grants

    def find_grant_for_action(
        self, principal_id: str, elevated_role: str
    ) -> Optional[ElevationGrant]:
        """
        Find an active grant that provides a specific elevated role.

        Args:
            principal_id: The principal identifier
            elevated_role: The role being checked

        Returns:
            ElevationGrant if found and valid, None otherwise
        """
        now = datetime.now(timezone.utc)

        for grant in self._grants.values():
            if (
                grant.principal_id == principal_id
                and grant.elevated_role == elevated_role
                and grant.is_valid(now)
            ):
                return grant

        return None

    def use_grant(self, grant_id: str) -> bool:
        """
        Mark a grant as used (increments usage counter).

        Args:
            grant_id: The grant identifier

        Returns:
            True if successfully used, False if grant not found or exhausted

        Raises:
            ValueError: If grant is no longer valid
        """
        grant = self.get_grant(grant_id)
        if grant is None:
            return False

        if not grant.is_valid():
            raise ValueError(f"Grant {grant_id} is no longer valid")

        # Increment usage count (need to work around frozen dataclass)
        updated_grant = ElevationGrant(
            grant_id=grant.grant_id,
            principal_id=grant.principal_id,
            elevated_role=grant.elevated_role,
            scope=grant.scope,
            approved_by=grant.approved_by,
            ticket_reference=grant.ticket_reference,
            valid_from=grant.valid_from,
            valid_until=grant.valid_until,
            purpose=grant.purpose,
            used_count=grant.used_count + 1,
            max_uses=grant.max_uses,
        )

        self._grants[grant_id] = updated_grant
        return True

    def revoke_grant(self, grant_id: str) -> bool:
        """
        Revoke a grant immediately.

        Args:
            grant_id: The grant identifier

        Returns:
            True if revoked, False if not found
        """
        if grant_id in self._grants:
            # Set expiry to now to invalidate it
            grant = self._grants[grant_id]
            now = datetime.now(timezone.utc).isoformat()

            revoked_grant = ElevationGrant(
                grant_id=grant.grant_id,
                principal_id=grant.principal_id,
                elevated_role=grant.elevated_role,
                scope=grant.scope,
                approved_by=grant.approved_by,
                ticket_reference=grant.ticket_reference,
                valid_from=grant.valid_from,
                valid_until=now,  # Set to now to invalidate
                purpose=grant.purpose,
                used_count=grant.used_count,
                max_uses=grant.max_uses,
            )

            self._grants[grant_id] = revoked_grant
            return True

        return False

    def get_grants_by_principal(self, principal_id: str) -> List[ElevationGrant]:
        """
        Get all grants for a principal (active and expired).

        Args:
            principal_id: The principal identifier

        Returns:
            List of all grants for the principal
        """
        return [g for g in self._grants.values() if g.principal_id == principal_id]

    def get_grants_by_approver(self, approver_id: str) -> List[ElevationGrant]:
        """
        Get all grants approved by a specific identity.

        Args:
            approver_id: The approver's identity

        Returns:
            List of grants approved by this identity
        """
        return [g for g in self._grants.values() if g.approved_by == approver_id]

    def list_all_grants(self) -> List[ElevationGrant]:
        """Get all grants in the system."""
        return list(self._grants.values())

    def cleanup_expired_grants(self) -> int:
        """
        Remove all expired grants from storage.

        Returns:
            Number of grants removed
        """
        now = datetime.now(timezone.utc)
        expired_ids = [
            grant_id
            for grant_id, grant in self._grants.items()
            if not grant.is_valid(now)
        ]

        for grant_id in expired_ids:
            del self._grants[grant_id]

        return len(expired_ids)

    def clear(self) -> None:
        """Clear all grants (useful for testing)."""
        self._grants.clear()
