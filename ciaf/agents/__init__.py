"""
CIAF Agentic Execution Boundaries

Identity and access management (IAM) and privileged access management (PAM)
controls for autonomous AI agents, with cryptographic provenance and audit trails.

This module implements zero-trust execution boundaries ensuring agents only
execute authorized actions with verifiable cryptographic proof.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

# Core types and interfaces
from .core import (
    ActionReceipt,
    ActionRequest,
    ElevationGrant,
    EvidenceRecorder,
    ExecutionResult,
    Identity,
    IdentityProvider,
    Permission,
    PolicyEvaluator,
    PrincipalType,
    Resource,
    RoleDefinition,
    ToolMediator,
)

# IAM components
from .iam import (
    IAMStore,
    combine_and,
    combine_or,
    create_attribute_matcher,
    identity_has_attribute,
    same_environment_only,
    same_tenant_only,
    sensitivity_level_check,
)

# PAM components
from .pam import PAMStore

# Policy evaluation
from .policy import PolicyEngine

# Evidence and receipts
from .evidence import EvidenceVault

# Tool execution
from .execution import ToolExecutor

__version__ = "1.0.0"

__all__ = [
    # Core types
    "Identity",
    "PrincipalType",
    "Resource",
    "ActionRequest",
    "Permission",
    "RoleDefinition",
    "ExecutionResult",
    "ElevationGrant",
    "ActionReceipt",
    # Core interfaces
    "IdentityProvider",
    "PolicyEvaluator",
    "EvidenceRecorder",
    "ToolMediator",
    # IAM
    "IAMStore",
    "same_tenant_only",
    "same_environment_only",
    "sensitivity_level_check",
    "create_attribute_matcher",
    "identity_has_attribute",
    "combine_and",
    "combine_or",
    # PAM
    "PAMStore",
    # Policy
    "PolicyEngine",
    # Evidence
    "EvidenceVault",
    # Execution
    "ToolExecutor",
]
