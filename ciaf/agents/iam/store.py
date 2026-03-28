"""
IAM Store implementation for CIAF Agentic Execution Boundaries.

Provides identity storage, role management, and RBAC/ABAC evaluation.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import Dict, List, Optional, Set

from ..core.interfaces import IdentityProvider
from ..core.types import Identity, Permission, RoleDefinition


class IAMStore(IdentityProvider):
    """
    In-memory Identity and Access Management store.

    Implements role-based access control (RBAC) with attribute-based
    conditions (ABAC) for CIAF agent systems.
    """

    def __init__(self):
        """Initialize the IAM store with empty collections."""
        self._identities: Dict[str, Identity] = {}
        self._roles: Dict[str, RoleDefinition] = {}
        self._role_inheritance_cache: Dict[str, Set[str]] = {}

    def add_identity(self, identity: Identity) -> None:
        """
        Register a new identity in the store.

        Args:
            identity: The Identity to register

        Raises:
            ValueError: If principal_id already exists
        """
        if identity.principal_id in self._identities:
            raise ValueError(
                f"Identity {identity.principal_id} already exists. "
                "Use update_identity to modify."
            )
        self._identities[identity.principal_id] = identity
        # Invalidate role cache for this principal
        if identity.principal_id in self._role_inheritance_cache:
            del self._role_inheritance_cache[identity.principal_id]

    def get_identity(self, principal_id: str) -> Optional[Identity]:
        """
        Retrieve an identity by principal ID.

        Args:
            principal_id: The principal identifier

        Returns:
            Identity if found, None otherwise
        """
        return self._identities.get(principal_id)

    def update_identity(self, identity: Identity) -> None:
        """
        Update an existing identity.

        Args:
            identity: The updated Identity

        Raises:
            ValueError: If identity does not exist
        """
        if identity.principal_id not in self._identities:
            raise ValueError(
                f"Identity {identity.principal_id} not found. "
                "Use add_identity to create."
            )
        self._identities[identity.principal_id] = identity
        # Invalidate role cache
        if identity.principal_id in self._role_inheritance_cache:
            del self._role_inheritance_cache[identity.principal_id]

    def add_role(self, role: RoleDefinition) -> None:
        """
        Register a role definition.

        Args:
            role: The RoleDefinition to register

        Raises:
            ValueError: If role name already exists
        """
        if role.name in self._roles:
            raise ValueError(
                f"Role {role.name} already exists. " "Use update_role to modify."
            )
        self._roles[role.name] = role
        # Clear all role inheritance caches as relationships may have changed
        self._role_inheritance_cache.clear()

    def get_role(self, role_name: str) -> Optional[RoleDefinition]:
        """
        Retrieve a role definition by name.

        Args:
            role_name: The role name

        Returns:
            RoleDefinition if found, None otherwise
        """
        return self._roles.get(role_name)

    def update_role(self, role: RoleDefinition) -> None:
        """
        Update an existing role definition.

        Args:
            role: The updated RoleDefinition

        Raises:
            ValueError: If role does not exist
        """
        if role.name not in self._roles:
            raise ValueError(f"Role {role.name} not found. " "Use add_role to create.")
        self._roles[role.name] = role
        # Clear all role caches as permissions may have changed
        self._role_inheritance_cache.clear()

    def get_effective_roles(self, principal_id: str) -> Set[str]:
        """
        Get all effective roles for a principal (including inherited).

        Uses caching for performance on repeated lookups.

        Args:
            principal_id: The principal identifier

        Returns:
            Set of all effective role names
        """
        # Check cache first
        if principal_id in self._role_inheritance_cache:
            return self._role_inheritance_cache[principal_id]

        identity = self.get_identity(principal_id)
        if identity is None:
            return set()

        effective_roles: Set[str] = set(identity.roles)
        roles_to_process = list(identity.roles)

        # Resolve role inheritance
        while roles_to_process:
            role_name = roles_to_process.pop()
            role = self.get_role(role_name)
            if role and role.inherits_from:
                for inherited_role in role.inherits_from:
                    if inherited_role not in effective_roles:
                        effective_roles.add(inherited_role)
                        roles_to_process.append(inherited_role)

        # Cache the result
        self._role_inheritance_cache[principal_id] = effective_roles
        return effective_roles

    def get_permissions(self, principal_id: str) -> List[Permission]:
        """
        Get all permissions granted to a principal.

        Resolves role inheritance and aggregates all permissions.

        Args:
            principal_id: The principal identifier

        Returns:
            List of all effective permissions
        """
        effective_roles = self.get_effective_roles(principal_id)
        permissions: List[Permission] = []

        for role_name in effective_roles:
            role = self.get_role(role_name)
            if role:
                permissions.extend(role.permissions)

        return permissions

    def has_permission(
        self, principal_id: str, action: str, resource_type: str
    ) -> bool:
        """
        Check if a principal has a specific permission.

        Args:
            principal_id: The principal identifier
            action: The action being checked
            resource_type: The resource type

        Returns:
            True if permission exists, False otherwise
        """
        permissions = self.get_permissions(principal_id)
        for perm in permissions:
            if perm.action == action and perm.resource_type == resource_type:
                return True
        return False

    def list_identities(self) -> List[Identity]:
        """Get all registered identities."""
        return list(self._identities.values())

    def list_roles(self) -> List[RoleDefinition]:
        """Get all registered roles."""
        return list(self._roles.values())

    def remove_identity(self, principal_id: str) -> bool:
        """
        Remove an identity from the store.

        Args:
            principal_id: The principal identifier

        Returns:
            True if removed, False if not found
        """
        if principal_id in self._identities:
            del self._identities[principal_id]
            if principal_id in self._role_inheritance_cache:
                del self._role_inheritance_cache[principal_id]
            return True
        return False

    def remove_role(self, role_name: str) -> bool:
        """
        Remove a role definition.

        Args:
            role_name: The role name

        Returns:
            True if removed, False if not found
        """
        if role_name in self._roles:
            del self._roles[role_name]
            self._role_inheritance_cache.clear()
            return True
        return False

    def clear(self) -> None:
        """Clear all identities and roles (useful for testing)."""
        self._identities.clear()
        self._roles.clear()
        self._role_inheritance_cache.clear()
