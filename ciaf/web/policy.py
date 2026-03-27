"""
CIAF Web - Policy Engine

Policy evaluation and enforcement for organizational AI usage governance.

Policy decisions:
- ALLOW: Permit the action
- WARN: Allow but notify user/admin
- REDACT: Remove sensitive parts
- BLOCK: Prevent the action
- ESCALATE: Route for manual review

Policy rules evaluate:
- Tool approval status (approved vs shadow AI)
- Content sensitivity
- User/group permissions
- Context (time, location, device)
- Compliance requirements

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum

from .events import (
    WebAIEvent,
    EventType,
    PolicyDecision,
    DataClassification,
)


class PolicyCondition(str, Enum):
    """Conditions that can trigger policy rules."""

    SHADOW_AI_DETECTED = "shadow_ai_detected"
    HIGH_SENSITIVITY = "high_sensitivity"
    RESTRICTED_DATA = "restricted_data"
    UNAPPROVED_TOOL = "unapproved_tool"
    FILE_UPLOAD = "file_upload"
    LARGE_CONTENT = "large_content"
    OFF_HOURS = "off_hours"
    UNTRUSTED_DEVICE = "untrusted_device"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class PolicyRule:
    """
    A policy rule for AI usage governance.

    Rules evaluate conditions and produce policy decisions.
    """

    rule_id: str
    name: str
    description: str
    conditions: List[PolicyCondition]
    decision: PolicyDecision
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    reason_template: str = ""
    metadata: Dict = field(default_factory=dict)

    # Custom evaluation function (optional)
    evaluator: Optional[Callable[[WebAIEvent], bool]] = None

    def matches(
        self, event: WebAIEvent, context: Optional[PolicyContext] = None
    ) -> bool:
        """
        Check if event matches this rule's conditions.

        Args:
            event: Event to evaluate
            context: Additional context

        Returns:
            True if conditions match
        """
        if not self.enabled:
            return False

        # Use custom evaluator if provided
        if self.evaluator:
            return self.evaluator(event)

        # Check built-in conditions
        for condition in self.conditions:
            if condition == PolicyCondition.SHADOW_AI_DETECTED:
                if event.is_shadow_ai():
                    return True

            elif condition == PolicyCondition.HIGH_SENSITIVITY:
                if event.is_high_risk(threshold=0.7):
                    return True

            elif condition == PolicyCondition.RESTRICTED_DATA:
                if event.data_classification in [
                    DataClassification.RESTRICTED,
                    DataClassification.HIGHLY_RESTRICTED,
                ]:
                    return True

            elif condition == PolicyCondition.UNAPPROVED_TOOL:
                if event.tool_approved is False:
                    return True

            elif condition == PolicyCondition.FILE_UPLOAD:
                if event.event_type == EventType.FILE_UPLOAD:
                    return True

            elif condition == PolicyCondition.LARGE_CONTENT:
                # Check metadata for content size
                if context and context.content_size and context.content_size > 10000:
                    return True

        return False

    def get_reason(self, event: WebAIEvent) -> str:
        """Get policy decision reason for this rule."""
        if self.reason_template:
            return self.reason_template.format(
                tool_name=event.tool_name or "unknown",
                classification=(
                    event.data_classification.value
                    if event.data_classification
                    else "unknown"
                ),
            )
        return self.description


@dataclass
class PolicyContext:
    """Additional context for policy evaluation."""

    content_size: Optional[int] = None
    device_trusted: bool = True
    time_of_day: Optional[str] = None
    user_groups: List[str] = field(default_factory=list)
    compliance_mode: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    decision: PolicyDecision
    matched_rule: Optional[PolicyRule] = None
    reason: str = ""
    allow_override: bool = False
    requires_review: bool = False
    metadata: Dict = field(default_factory=dict)

    def is_allowed(self) -> bool:
        """Check if action is allowed."""
        return self.decision == PolicyDecision.ALLOW

    def is_blocked(self) -> bool:
        """Check if action is blocked."""
        return self.decision == PolicyDecision.BLOCK

    def needs_warning(self) -> bool:
        """Check if warning should be shown."""
        return self.decision == PolicyDecision.WARN

    def needs_redaction(self) -> bool:
        """Check if content should be redacted."""
        return self.decision == PolicyDecision.REDACT


# Default policy rules for organizations
DEFAULT_POLICY_RULES = [
    PolicyRule(
        rule_id="block_highly_restricted",
        name="Block Highly Restricted Data",
        description="Block sharing of highly restricted data with any AI tool",
        conditions=[PolicyCondition.RESTRICTED_DATA],
        decision=PolicyDecision.BLOCK,
        priority=10,  # Highest priority
        reason_template="Highly restricted data detected. Classification: {classification}",
    ),
    PolicyRule(
        rule_id="warn_shadow_ai",
        name="Warn on Shadow AI Usage",
        description="Warn when unapproved AI tools are detected",
        conditions=[PolicyCondition.SHADOW_AI_DETECTED],
        decision=PolicyDecision.WARN,
        priority=20,
        reason_template="Unapproved AI tool detected: {tool_name}. Please use approved enterprise tools.",
    ),
    PolicyRule(
        rule_id="escalate_high_sensitivity",
        name="Escalate High Sensitivity Content",
        description="Escalate high sensitivity content for review",
        conditions=[PolicyCondition.HIGH_SENSITIVITY],
        decision=PolicyDecision.ESCALATE,
        priority=30,
        reason_template="High sensitivity content detected. Manual review required.",
    ),
    PolicyRule(
        rule_id="redact_pii_shadow_ai",
        name="Redact PII in Shadow AI",
        description="Redact PII when using unapproved tools",
        conditions=[
            PolicyCondition.SHADOW_AI_DETECTED,
            PolicyCondition.HIGH_SENSITIVITY,
        ],
        decision=PolicyDecision.REDACT,
        priority=25,
        reason_template="PII detected with unapproved tool. Content will be redacted.",
    ),
    PolicyRule(
        rule_id="allow_approved_tools",
        name="Allow Approved Tools",
        description="Allow usage of approved enterprise AI tools",
        conditions=[],
        decision=PolicyDecision.ALLOW,
        priority=100,
        reason_template="Approved AI tool usage",
    ),
]


class PolicyEngine:
    """
    Evaluate and enforce AI usage policies.

    The policy engine evaluates events against organizational rules
    and determines whether to allow, warn, redact, block, or escalate.
    """

    def __init__(
        self,
        rules: Optional[List[PolicyRule]] = None,
        use_defaults: bool = True,
    ):
        """
        Initialize policy engine.

        Args:
            rules: Custom policy rules
            use_defaults: Whether to include default rules
        """
        self.rules = []

        if use_defaults:
            self.rules.extend(DEFAULT_POLICY_RULES)

        if rules:
            self.rules.extend(rules)

        # Sort by priority (lower = higher priority)
        self.rules.sort(key=lambda r: r.priority)

    def evaluate(
        self,
        event: WebAIEvent,
        context: Optional[PolicyContext] = None,
    ) -> PolicyResult:
        """
        Evaluate event against policies.

        Args:
            event: Event to evaluate
            context: Additional context

        Returns:
            PolicyResult with decision
        """
        # Check each rule in priority order
        for rule in self.rules:
            if rule.matches(event, context):
                return PolicyResult(
                    decision=rule.decision,
                    matched_rule=rule,
                    reason=rule.get_reason(event),
                    requires_review=(rule.decision == PolicyDecision.ESCALATE),
                )

        # No rules matched - default to allow
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="No policy restrictions apply",
        )

    def evaluate_batch(
        self,
        events: List[WebAIEvent],
        context: Optional[PolicyContext] = None,
    ) -> List[PolicyResult]:
        """
        Evaluate multiple events.

        Args:
            events: Events to evaluate
            context: Shared context

        Returns:
            List of PolicyResults
        """
        return [self.evaluate(event, context) for event in events]

    def add_rule(self, rule: PolicyRule):
        """Add a policy rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, rule_id: str):
        """Remove a policy rule by ID."""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def enable_rule(self, rule_id: str):
        """Enable a policy rule."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = True

    def disable_rule(self, rule_id: str):
        """Disable a policy rule."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = False

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """Get a policy rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None


# Convenience functions


def evaluate_policy(
    event: WebAIEvent,
    rules: Optional[List[PolicyRule]] = None,
    context: Optional[PolicyContext] = None,
) -> PolicyResult:
    """
    Evaluate event against policies.

    Args:
        event: Event to evaluate
        rules: Custom rules (uses defaults if None)
        context: Additional context

    Returns:
        PolicyResult
    """
    engine = PolicyEngine(rules=rules)
    return engine.evaluate(event, context)


def should_allow(event: WebAIEvent, **kwargs) -> bool:
    """Quick check if event should be allowed."""
    result = evaluate_policy(event, **kwargs)
    return result.is_allowed()


def should_block(event: WebAIEvent, **kwargs) -> bool:
    """Quick check if event should be blocked."""
    result = evaluate_policy(event, **kwargs)
    return result.is_blocked()


__all__ = [
    "PolicyEngine",
    "PolicyRule",
    "PolicyResult",
    "PolicyContext",
    "PolicyCondition",
    "DEFAULT_POLICY_RULES",
    "evaluate_policy",
    "should_allow",
    "should_block",
]
