"""
CIAF Web - Content Classifiers

Classify content for sensitivity and risk scoring.

Classification levels:
- PUBLIC: Can be shared externally
- INTERNAL: Internal use only
- CONFIDENTIAL: Limited distribution
- RESTRICTED: Highly sensitive
- HIGHLY_RESTRICTED: Strictly controlled

Sensitivity scoring (0.0-1.0):
- 0.0-0.3: Low sensitivity
- 0.3-0.6: Medium sensitivity
- 0.6-0.8: High sensitivity
- 0.8-1.0: Critical sensitivity

Detection methods:
- Pattern matching (regex)
- Keyword scanning
- PII detection
- Proprietary data markers
- Context analysis

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Set
import re

from .events import DataClassification


@dataclass
class ClassificationRule:
    """
    Rule for classifying content.

    Each rule checks for specific patterns and assigns
    a classification and sensitivity score.
    """
    rule_id: str
    name: str
    classification: DataClassification
    patterns: List[str]  # Regex patterns
    keywords: List[str] = None
    weight: float = 1.0  # Score multiplier
    description: Optional[str] = None

    def matches(self, text: str) -> bool:
        """Check if text matches this rule."""
        text_lower = text.lower()

        # Check patterns
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check keywords
        if self.keywords:
            for keyword in self.keywords:
                if keyword.lower() in text_lower:
                    return True

        return False


# Built-in classification rules
CLASSIFICATION_RULES = [
    # PII Patterns
    ClassificationRule(
        rule_id="pii_ssn",
        name="Social Security Number",
        classification=DataClassification.RESTRICTED,
        patterns=[r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
        weight=1.0,
        description="US Social Security Number",
    ),
    ClassificationRule(
        rule_id="pii_credit_card",
        name="Credit Card Number",
        classification=DataClassification.RESTRICTED,
        patterns=[
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Standard format
            r"\b\d{15,16}\b",  # Continuous digits
        ],
        weight=1.0,
        description="Credit card number",
    ),
    ClassificationRule(
        rule_id="pii_email",
        name="Email Address",
        classification=DataClassification.INTERNAL,
        patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        weight=0.5,
        description="Email address",
    ),
    ClassificationRule(
        rule_id="pii_phone",
        name="Phone Number",
        classification=DataClassification.INTERNAL,
        patterns=[
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US format
            r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",  # (XXX) XXX-XXXX
        ],
        weight=0.6,
        description="Phone number",
    ),

    # Financial Data
    ClassificationRule(
        rule_id="financial_bank_account",
        name="Bank Account Number",
        classification=DataClassification.RESTRICTED,
        patterns=[r"\b\d{8,17}\b"],
        keywords=["account number", "bank account", "routing number"],
        weight=0.9,
        description="Bank account information",
    ),
    ClassificationRule(
        rule_id="financial_salary",
        name="Salary Information",
        classification=DataClassification.CONFIDENTIAL,
        keywords=["salary", "compensation", "pay rate", "annual income"],
        patterns=[r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?"],
        weight=0.7,
        description="Salary or compensation data",
    ),

    # Health Data
    ClassificationRule(
        rule_id="health_phi",
        name="Protected Health Information",
        classification=DataClassification.HIGHLY_RESTRICTED,
        keywords=[
            "medical record", "diagnosis", "prescription", "patient",
            "health condition", "treatment plan", "medical history"
        ],
        patterns=[],
        weight=1.0,
        description="Protected health information (PHI)",
    ),

    # Business Confidential
    ClassificationRule(
        rule_id="business_confidential",
        name="Confidential Business Data",
        classification=DataClassification.CONFIDENTIAL,
        keywords=[
            "confidential", "proprietary", "trade secret", "internal only",
            "do not distribute", "restricted distribution"
        ],
        patterns=[r"\bconfidential\b", r"\bproprietary\b"],
        weight=0.8,
        description="Confidential business information",
    ),
    ClassificationRule(
        rule_id="business_strategy",
        name="Strategic Business Information",
        classification=DataClassification.CONFIDENTIAL,
        keywords=[
            "business strategy", "strategic plan", "market strategy",
            "competitive advantage", "roadmap", "M&A", "acquisition"
        ],
        patterns=[],
        weight=0.7,
        description="Strategic business plans",
    ),

    # Technical Secrets
    ClassificationRule(
        rule_id="tech_api_key",
        name="API Key",
        classification=DataClassification.RESTRICTED,
        patterns=[
            r"api[_-]?key[\s:=]+['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
            r"bearer\s+[a-zA-Z0-9_-]{20,}",
        ],
        weight=1.0,
        description="API keys or tokens",
    ),
    ClassificationRule(
        rule_id="tech_password",
        name="Password",
        classification=DataClassification.RESTRICTED,
        patterns=[
            r"password[\s:=]+['\"]?[^\s'\"]{8,}['\"]?",
            r"pwd[\s:=]+['\"]?[^\s'\"]{8,}['\"]?",
        ],
        weight=1.0,
        description="Passwords",
    ),
    ClassificationRule(
        rule_id="tech_private_key",
        name="Private Key",
        classification=DataClassification.HIGHLY_RESTRICTED,
        patterns=[
            r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
            r"private[_-]?key[\s:=]",
        ],
        weight=1.0,
        description="Cryptographic private keys",
    ),

    # Code and IP
    ClassificationRule(
        rule_id="code_proprietary",
        name="Proprietary Code",
        classification=DataClassification.CONFIDENTIAL,
        keywords=["proprietary algorithm", "patent pending", "copyright"],
        patterns=[r"\(c\)\s*\d{4}", r"copyright\s*\d{4}"],
        weight=0.7,
        description="Proprietary source code",
    ),
]


class ContentClassifier:
    """
    Classify content for sensitivity and risk.

    Uses pattern matching, keyword detection, and rule-based
    classification to assign data classifications and sensitivity scores.
    """

    def __init__(self, custom_rules: Optional[List[ClassificationRule]] = None):
        """
        Initialize classifier.

        Args:
            custom_rules: Optional custom classification rules
        """
        self.rules = CLASSIFICATION_RULES.copy()
        if custom_rules:
            self.rules.extend(custom_rules)

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text for sensitivity.

        Args:
            text: Text content to classify

        Returns:
            ClassificationResult with classification and score
        """
        if not text or len(text.strip()) == 0:
            return ClassificationResult(
                classification=DataClassification.UNKNOWN,
                sensitivity_score=0.0,
                matched_rules=[],
                confidence=1.0,
            )

        matched_rules = []
        max_classification = DataClassification.PUBLIC
        sensitivity_scores = []

        # Check each rule
        for rule in self.rules:
            if rule.matches(text):
                matched_rules.append(rule)
                sensitivity_scores.append(rule.weight)

                # Track highest classification
                if _classification_level(rule.classification) > _classification_level(max_classification):
                    max_classification = rule.classification

        # Calculate sensitivity score
        if sensitivity_scores:
            # Use max score capped at 1.0
            sensitivity_score = min(1.0, max(sensitivity_scores))
        else:
            # No matches - assume public or internal
            sensitivity_score = 0.1
            max_classification = DataClassification.PUBLIC

        # Confidence based on number of matches
        confidence = min(1.0, 0.5 + (len(matched_rules) * 0.1))

        return ClassificationResult(
            classification=max_classification,
            sensitivity_score=sensitivity_score,
            matched_rules=matched_rules,
            confidence=confidence,
            findings_count=len(matched_rules),
        )

    def add_rule(self, rule: ClassificationRule):
        """Add a custom classification rule."""
        self.rules.append(rule)


@dataclass
class ClassificationResult:
    """Result of content classification."""
    classification: DataClassification
    sensitivity_score: float  # 0.0-1.0
    matched_rules: List[ClassificationRule]
    confidence: float = 1.0
    findings_count: int = 0
    metadata: Dict = None

    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """Check if content is high risk."""
        return self.sensitivity_score >= threshold

    def is_restricted(self) -> bool:
        """Check if content is restricted or highly restricted."""
        return self.classification in [
            DataClassification.RESTRICTED,
            DataClassification.HIGHLY_RESTRICTED
        ]

    def should_block(self) -> bool:
        """Determine if content should be blocked (highly sensitive)."""
        return (
            self.classification == DataClassification.HIGHLY_RESTRICTED or
            self.sensitivity_score >= 0.9
        )

    def should_warn(self) -> bool:
        """Determine if content should trigger warning."""
        return (
            self.classification in [DataClassification.RESTRICTED, DataClassification.CONFIDENTIAL] or
            self.sensitivity_score >= 0.6
        )


class SensitivityScorer:
    """
    Score content sensitivity using multiple signals.

    Provides a normalized 0.0-1.0 score where:
    - 0.0-0.3: Low sensitivity (public/internal)
    - 0.3-0.6: Medium sensitivity
    - 0.6-0.8: High sensitivity
    - 0.8-1.0: Critical sensitivity
    """

    def score(self, text: str) -> float:
        """
        Calculate sensitivity score for text.

        Args:
            text: Content to score

        Returns:
            Sensitivity score (0.0-1.0)
        """
        classifier = ContentClassifier()
        result = classifier.classify(text)
        return result.sensitivity_score


def _classification_level(classification: DataClassification) -> int:
    """Get numeric level for classification (for comparison)."""
    levels = {
        DataClassification.PUBLIC: 0,
        DataClassification.INTERNAL: 1,
        DataClassification.CONFIDENTIAL: 2,
        DataClassification.RESTRICTED: 3,
        DataClassification.HIGHLY_RESTRICTED: 4,
        DataClassification.UNKNOWN: 0,
    }
    return levels.get(classification, 0)


# Convenience functions

def classify_content(text: str) -> ClassificationResult:
    """
    Classify text content.

    Args:
        text: Content to classify

    Returns:
        ClassificationResult
    """
    classifier = ContentClassifier()
    return classifier.classify(text)


def score_sensitivity(text: str) -> float:
    """
    Score content sensitivity.

    Args:
        text: Content to score

    Returns:
        Sensitivity score (0.0-1.0)
    """
    scorer = SensitivityScorer()
    return scorer.score(text)


__all__ = [
    "ContentClassifier",
    "SensitivityScorer",
    "ClassificationRule",
    "ClassificationResult",
    "CLASSIFICATION_RULES",
    "classify_content",
    "score_sensitivity",
]
