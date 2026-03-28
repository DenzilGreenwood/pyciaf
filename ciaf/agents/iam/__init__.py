"""
Identity and Access Management (IAM) components for CIAF agents.

Provides identity resolution, role management, and permission evaluation
integrated with CIAF's cryptographic provenance system.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .store import IAMStore
from .policy_conditions import (
    same_tenant_only,
    same_environment_only,
    sensitivity_level_check,
    create_attribute_matcher,
    identity_has_attribute,
    combine_and,
    combine_or,
)

__all__ = [
    "IAMStore",
    "same_tenant_only",
    "same_environment_only",
    "sensitivity_level_check",
    "create_attribute_matcher",
    "identity_has_attribute",
    "combine_and",
    "combine_or",
]
