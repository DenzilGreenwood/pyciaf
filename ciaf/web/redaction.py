"""
CIAF Web - Content Redaction

Privacy-preserving content redaction for sensitive data.

Redaction strategies:
- PII removal (SSN, credit cards, emails)
- Hash-based replacement (preserve structure)
- Token replacement (preserve format)
- Selective field redaction

Use cases:
- Redact PII before sending to public AI
- Hash sensitive fields while preserving structure
- Remove credentials and API keys
- Sanitize content for logging

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable
import re
import hashlib


class RedactionRule:
    """
    Rule for redacting sensitive content.

    Defines what to redact and how to replace it.
    """

    def __init__(
        self,
        name: str,
        pattern: str,
        replacement: str = "[REDACTED]",
        replacement_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize redaction rule.

        Args:
            name: Rule identifier
            pattern: Regex pattern to match
            replacement: Static replacement text
            replacement_fn: Dynamic replacement function (overrides replacement)
        """
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.replacement = replacement
        self.replacement_fn = replacement_fn

    def redact(self, text: str) -> str:
        """Apply redaction to text."""
        if self.replacement_fn:
            return self.pattern.sub(lambda m: self.replacement_fn(m.group(0)), text)
        else:
            return self.pattern.sub(self.replacement, text)


# Built-in redaction rules
REDACTION_RULES = {
    "ssn": RedactionRule(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        replacement="[SSN-REDACTED]",
    ),
    "credit_card": RedactionRule(
        name="credit_card",
        pattern=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        replacement="[CARD-REDACTED]",
    ),
    "email": RedactionRule(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        replacement="[EMAIL-REDACTED]",
    ),
    "phone": RedactionRule(
        name="phone",
        pattern=r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        replacement="[PHONE-REDACTED]",
    ),
    "api_key": RedactionRule(
        name="api_key",
        pattern=r"api[_-]?key[\s:=]+['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
        replacement="[API-KEY-REDACTED]",
    ),
    "password": RedactionRule(
        name="password",
        pattern=r"password[\s:=]+['\"]?[^\s'\"]{8,}['\"]?",
        replacement="[PASSWORD-REDACTED]",
    ),
    "bearer_token": RedactionRule(
        name="bearer_token",
        pattern=r"bearer\s+[a-zA-Z0-9_-]{20,}",
        replacement="[TOKEN-REDACTED]",
    ),
    "private_key": RedactionRule(
        name="private_key",
        pattern=r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (RSA |EC |DSA )?PRIVATE KEY-----",
        replacement="[PRIVATE-KEY-REDACTED]",
    ),
}


def hash_replacement(value: str, length: int = 8) -> str:
    """
    Create hash-based replacement that preserves partial structure.

    Args:
        value: Value to hash
        length: Length of hash prefix to use

    Returns:
        Hash-based replacement
    """
    hash_val = hashlib.sha256(value.encode()).hexdigest()
    return f"[HASH:{hash_val[:length]}...]"


def create_hash_rule(name: str, pattern: str) -> RedactionRule:
    """
    Create redaction rule that replaces with hash.

    Args:
        name: Rule name
        pattern: Regex pattern

    Returns:
        RedactionRule with hash replacement
    """
    return RedactionRule(
        name=name,
        pattern=pattern,
        replacement_fn=hash_replacement,
    )


class ContentRedactor:
    """
    Redact sensitive content from text.

    Supports multiple redaction strategies and custom rules.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, RedactionRule]] = None,
        use_defaults: bool = True,
    ):
        """
        Initialize redactor.

        Args:
            rules: Custom redaction rules
            use_defaults: Whether to include default PII rules
        """
        self.rules = {}

        if use_defaults:
            self.rules.update(REDACTION_RULES)

        if rules:
            self.rules.update(rules)

    def redact(self, text: str, rule_names: Optional[List[str]] = None) -> str:
        """
        Redact sensitive content from text.

        Args:
            text: Text to redact
            rule_names: Specific rules to apply (None = all rules)

        Returns:
            Redacted text
        """
        if not text:
            return text

        redacted = text

        # Apply rules
        for name, rule in self.rules.items():
            if rule_names is None or name in rule_names:
                redacted = rule.redact(redacted)

        return redacted

    def add_rule(self, name: str, rule: RedactionRule):
        """Add custom redaction rule."""
        self.rules[name] = rule

    def remove_rule(self, name: str):
        """Remove redaction rule."""
        self.rules.pop(name, None)


def redact_pii(text: str) -> str:
    """
    Redact common PII from text.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    redactor = ContentRedactor()
    return redactor.redact(
        text,
        rule_names=["ssn", "credit_card", "email", "phone"],
    )


def redact_credentials(text: str) -> str:
    """
    Redact credentials and secrets from text.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    redactor = ContentRedactor()
    return redactor.redact(
        text,
        rule_names=["api_key", "password", "bearer_token", "private_key"],
    )


def redact_content(text: str, aggressive: bool = False) -> str:
    """
    Redact sensitive content.

    Args:
        text: Text to redact
        aggressive: If True, redact all PII and credentials

    Returns:
        Redacted text
    """
    redactor = ContentRedactor()

    if aggressive:
        # Redact everything
        return redactor.redact(text)
    else:
        # Redact only high-risk items
        return redactor.redact(
            text,
            rule_names=["ssn", "credit_card", "api_key", "password", "private_key"],
        )


def hash_sensitive_fields(data: Dict, fields: List[str]) -> Dict:
    """
    Hash specific fields in dictionary while preserving structure.

    Args:
        data: Data dictionary
        fields: Field names to hash

    Returns:
        Dictionary with hashed fields
    """
    result = data.copy()

    for field in fields:
        if field in result and result[field]:
            value = str(result[field])
            result[field] = hash_replacement(value)

    return result


def mask_content(text: str, show_first: int = 4, show_last: int = 4) -> str:
    """
    Mask content while showing first/last characters.

    Example: "secret123456" -> "secr****3456"

    Args:
        text: Text to mask
        show_first: Number of first characters to show
        show_last: Number of last characters to show

    Returns:
        Masked text
    """
    if len(text) <= show_first + show_last:
        return "*" * len(text)

    return (
        text[:show_first]
        + "*" * (len(text) - show_first - show_last)
        + text[-show_last:]
    )


__all__ = [
    "ContentRedactor",
    "RedactionRule",
    "REDACTION_RULES",
    "redact_content",
    "redact_pii",
    "redact_credentials",
    "hash_sensitive_fields",
    "hash_replacement",
    "create_hash_rule",
    "mask_content",
]
