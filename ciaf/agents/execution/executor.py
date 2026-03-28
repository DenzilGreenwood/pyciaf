"""
Tool Executor implementation for mediated agent tool execution.

Centralizes authorization, execution, and evidence recording for all
agent tool calls within the execution boundary.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import Any, Callable, Dict, Optional

from ..core.interfaces import ToolMediator
from ..core.types import ActionRequest, ExecutionResult
from ..evidence import EvidenceVault
from ..pam import PAMStore
from ..policy import PolicyEngine


class ToolExecutor(ToolMediator):
    """
    Mediated tool executor with authorization and audit.

    Orchestrates the complete execution flow:
    1. Policy evaluation (authorization decision)
    2. Mediated tool execution (if authorized)
    3. Evidence recording (cryptographic receipt)
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        evidence_vault: EvidenceVault,
        pam_store: Optional[PAMStore] = None,
    ):
        """
        Initialize the tool executor.

        Args:
            policy_engine: Policy engine for authorization
            evidence_vault: Evidence vault for receipt generation
            pam_store: Optional PAM store for grant usage tracking
        """
        self._policy = policy_engine
        self._evidence = evidence_vault
        self._pam = pam_store
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, dict] = {}

    def register_tool(
        self, tool_name: str, tool_function: Callable, schema: Optional[dict] = None
    ) -> None:
        """
        Register a tool for mediated execution.

        Args:
            tool_name: Unique tool identifier
            tool_function: The actual tool implementation
            schema: Optional JSON schema for parameter validation
        """
        self._tools[tool_name] = tool_function
        if schema:
            self._tool_schemas[tool_name] = schema

    def execute_tool(
        self, tool_name: str, request: ActionRequest
    ) -> ExecutionResult:
        """
        Execute a tool with full mediation and audit.

        Execution flow:
        1. Check if tool exists
        2. Evaluate authorization policy
        3. If authorized, execute tool with parameters
        4. Generate cryptographic receipt
        5. Update PAM grant usage if applicable

        Args:
            tool_name: The tool to execute
            request: The action request with parameters

        Returns:
            ExecutionResult with execution outcome and receipt
        """
        # Check tool registration
        if tool_name not in self._tools:
            result = ExecutionResult(
                request=request,
                allowed=False,
                reason=f"Tool '{tool_name}' not registered",
            )
            # Record denial
            self._evidence.record_action(result)
            return result

        # Policy evaluation
        auth_result = self._policy.evaluate(request)

        if not auth_result.allowed:
            # Record denial
            self._evidence.record_action(auth_result)
            return auth_result

        # Execute the tool
        try:
            tool_function = self._tools[tool_name]
            execution_output = tool_function(**request.params)

            # Mark as executed
            auth_result.executed = True
            auth_result.result = execution_output

        except Exception as e:
            # Execution failed
            auth_result.executed = False
            auth_result.error = str(e)
            auth_result.reason += f" | Execution error: {str(e)}"

        # Record evidence
        receipt = self._evidence.record_action(auth_result)

        # Update PAM grant usage if applicable
        if auth_result.elevation_grant_id and self._pam:
            try:
                self._pam.use_grant(auth_result.elevation_grant_id)
            except ValueError:
                # Grant no longer valid - this shouldn't happen but handle it
                pass

        # Attach receipt to result (for caller's reference)
        auth_result.result = {
            "tool_output": auth_result.result,
            "receipt_id": receipt.receipt_id,
            "receipt": receipt,
        }

        return auth_result

    def execute(self, request: ActionRequest) -> ExecutionResult:
        """
        Execute an action request (alias for policy-only evaluation).

        For simple authorization without tool execution.

        Args:
            request: The action request

        Returns:
            ExecutionResult with authorization decision
        """
        # Evaluate policy
        result = self._policy.evaluate(request)

        # Record evidence
        self._evidence.record_action(result)

        return result

    def get_registered_tools(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[dict]:
        """
        Get the schema for a registered tool.

        Args:
            tool_name: The tool identifier

        Returns:
            Tool schema if registered, None otherwise
        """
        return self._tool_schemas.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the executor.

        Args:
            tool_name: The tool identifier

        Returns:
            True if removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            if tool_name in self._tool_schemas:
                del self._tool_schemas[tool_name]
            return True
        return False
