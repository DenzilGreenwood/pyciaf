"""
Core types and interfaces for CIAF Agentic Execution Boundaries.

This module defines the fundamental types for identity management, resource
access control, and action authorization in autonomous AI agent systems.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .types import (
    Identity,
    PrincipalType,
    Resource,
    ActionRequest,
    Permission,
    RoleDefinition,
    ExecutionResult,
    ElevationGrant,
    ActionReceipt,
)
from .interfaces import (
    IdentityProvider,
    PolicyEvaluator,
    EvidenceRecorder,
    ToolMediator,
)

__all__ = [
    "Identity",
    "PrincipalType",
    "Resource",
    "ActionRequest",
    "Permission",
    "RoleDefinition",
    "ExecutionResult",
    "ElevationGrant",
    "ActionReceipt",
    "IdentityProvider",
    "PolicyEvaluator",
    "EvidenceRecorder",
    "ToolMediator",
]
