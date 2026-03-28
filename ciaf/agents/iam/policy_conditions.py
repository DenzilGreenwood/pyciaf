"""
Policy condition functions for ABAC (Attribute-Based Access Control).

Provides reusable condition functions for Permission objects that enable
fine-grained access control based on identity and resource attributes.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import Any, Callable

from ..core.types import Identity, Resource


def same_tenant_only(identity: Identity, resource: Resource) -> bool:
    """
    Allow access only if identity and resource belong to same tenant.

    Args:
        identity: The requesting identity
        resource: The target resource

    Returns:
        True if tenant IDs match, False otherwise
    """
    if identity.tenant_id is None or resource.owner_tenant is None:
        return False
    return identity.tenant_id == resource.owner_tenant


def same_environment_only(identity: Identity, resource: Resource) -> bool:
    """
    Allow access only if identity and resource are in same environment.

    Args:
        identity: The requesting identity
        resource: The target resource

    Returns:
        True if environments match, False otherwise
    """
    identity_env = identity.environment or "production"
    resource_env = resource.attributes_dict.get("environment", "production")
    return bool(identity_env == resource_env)


def sensitivity_level_check(
    max_level: str = "critical",
) -> Callable[[Identity, Resource], bool]:
    """
    Create a condition that checks resource sensitivity level.

    Sensitivity hierarchy: standard < sensitive < critical

    Args:
        max_level: Maximum allowed sensitivity level

    Returns:
        Condition function for Permission
    """
    level_hierarchy = {"standard": 0, "sensitive": 1, "critical": 2}

    def condition(identity: Identity, resource: Resource) -> bool:
        resource_level = resource.sensitivity_level or "standard"
        max_rank = level_hierarchy.get(max_level, 0)
        resource_rank = level_hierarchy.get(resource_level, 0)
        return resource_rank <= max_rank

    return condition


def create_attribute_matcher(
    attribute_name: str, allowed_values: set[Any]
) -> Callable[[Identity, Resource], bool]:
    """
    Create a condition that checks if resource attribute matches allowed values.

    Args:
        attribute_name: Name of the resource attribute to check
        allowed_values: Set of allowed attribute values

    Returns:
        Condition function for Permission
    """

    def condition(identity: Identity, resource: Resource) -> bool:
        resource_value = resource.attributes_dict.get(attribute_name)
        return resource_value in allowed_values

    return condition


def identity_has_attribute(
    attribute_name: str, required_value: Any
) -> Callable[[Identity, Resource], bool]:
    """
    Create a condition that checks if identity has a specific attribute value.

    Args:
        attribute_name: Name of the identity attribute to check
        required_value: Required attribute value

    Returns:
        Condition function for Permission
    """

    def condition(identity: Identity, resource: Resource) -> bool:
        identity_value = identity.attributes_dict.get(attribute_name)
        return bool(identity_value == required_value)

    return condition


def combine_and(
    *conditions: Callable[[Identity, Resource], bool]
) -> Callable[[Identity, Resource], bool]:
    """
    Combine multiple conditions with AND logic.

    Args:
        conditions: Variable number of condition functions

    Returns:
        Combined condition function that requires all conditions to pass
    """

    def combined(identity: Identity, resource: Resource) -> bool:
        return all(cond(identity, resource) for cond in conditions)

    return combined


def combine_or(
    *conditions: Callable[[Identity, Resource], bool]
) -> Callable[[Identity, Resource], bool]:
    """
    Combine multiple conditions with OR logic.

    Args:
        conditions: Variable number of condition functions

    Returns:
        Combined condition function that requires any condition to pass
    """

    def combined(identity: Identity, resource: Resource) -> bool:
        return any(cond(identity, resource) for cond in conditions)

    return combined
