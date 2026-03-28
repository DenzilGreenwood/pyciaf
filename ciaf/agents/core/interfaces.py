"""
Protocol interfaces for CIAF Agentic Execution Boundaries.

Defines clean contracts for swappable components following CIAF's
Protocol-based architecture pattern.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import Callable, List, Optional, Protocol, runtime_checkable

from .types import (
    ActionReceipt,
    ActionRequest,
    ElevationGrant,
    ExecutionResult,
    Identity,
)


@runtime_checkable
class IdentityProvider(Protocol):
    """Protocol for identity resolution and management."""

    def get_identity(self, principal_id: str) -> Optional[Identity]:
        """Retrieve an identity by principal ID."""
        ...

    def add_identity(self, identity: Identity) -> None:
        """Register a new identity."""
        ...

    def get_effective_roles(self, principal_id: str) -> set[str]:
        """Get all effective roles for a principal (including inherited)."""
        ...


@runtime_checkable
class PolicyEvaluator(Protocol):
    """Protocol for policy evaluation and authorization decisions."""

    def evaluate(self, request: ActionRequest) -> ExecutionResult:
        """Evaluate an action request against policies."""
        ...

    def requires_elevation(self, request: ActionRequest) -> bool:
        """Check if an action requires privilege elevation."""
        ...

    def get_active_grant(
        self, principal_id: str, action: str
    ) -> Optional[ElevationGrant]:
        """Find an active elevation grant for a principal and action."""
        ...


@runtime_checkable
class EvidenceRecorder(Protocol):
    """Protocol for recording cryptographic evidence of actions."""

    def record_action(self, result: ExecutionResult) -> ActionReceipt:
        """Record an action execution and return a signed receipt."""
        ...

    def verify_receipt(self, receipt: ActionReceipt) -> bool:
        """Verify a receipt's cryptographic integrity."""
        ...

    def get_receipts_by_principal(self, principal_id: str) -> List[ActionReceipt]:
        """Retrieve all receipts for a given principal."""
        ...

    def verify_chain(self) -> bool:
        """Verify the complete receipt chain integrity."""
        ...


@runtime_checkable
class ToolMediator(Protocol):
    """Protocol for mediated tool execution."""

    def execute_tool(self, tool_name: str, request: ActionRequest) -> ExecutionResult:
        """Execute a tool with mediation and controls."""
        ...

    def register_tool(
        self, tool_name: str, tool_function: Callable, schema: dict
    ) -> None:
        """Register a mediated tool with its execution schema."""
        ...
