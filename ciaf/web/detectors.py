"""
CIAF Web - AI Tool Detectors

Detect AI tool usage across web browsers including approved enterprise tools
and shadow AI (unapproved public tools).

Detection methods:
- Domain pattern matching
- Page structure analysis
- API endpoint detection
- JavaScript signature detection

Tool categories detected:
- LLM chat interfaces (ChatGPT, Claude, Gemini, etc.)
- Code assistants (GitHub Copilot, Cursor, etc.)
- Image generators (Midjourney, DALL-E, Stable Diffusion)
- Document AI (Notion AI, Google Docs AI)
- Search AI (Perplexity, Bing Chat)

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Set
from urllib.parse import urlparse

from .events import ToolCategory


@dataclass
class AIToolSignature:
    """
    Signature for detecting a specific AI tool.

    Includes domain patterns, API endpoints, and other identifiers.
    """

    tool_name: str
    tool_category: ToolCategory
    domains: List[str]  # Domain patterns (can include wildcards)
    api_patterns: List[str] = None  # API endpoint patterns
    js_signatures: List[str] = None  # JavaScript identifiers
    page_title_patterns: List[str] = None
    vendor: Optional[str] = None
    is_enterprise: bool = False  # True if typically deployed as enterprise tool

    def matches_domain(self, domain: str) -> bool:
        """Check if domain matches this tool."""
        domain_lower = domain.lower()
        for pattern in self.domains:
            if pattern.startswith("*."):
                # Wildcard subdomain
                suffix = pattern[2:]
                if domain_lower.endswith(suffix):
                    return True
            elif pattern == domain_lower:
                return True
        return False

    def matches_api(self, url: str) -> bool:
        """Check if URL matches API pattern."""
        if not self.api_patterns:
            return False
        url_lower = url.lower()
        for pattern in self.api_patterns:
            if pattern in url_lower:
                return True
        return False


# Known AI tool signatures
AI_TOOL_SIGNATURES = [
    # LLM Chat Interfaces
    AIToolSignature(
        tool_name="ChatGPT",
        tool_category=ToolCategory.LLM_CHAT,
        domains=["chat.openai.com", "chatgpt.com"],
        api_patterns=["/backend-api/conversation", "/api/chat"],
        vendor="OpenAI",
    ),
    AIToolSignature(
        tool_name="Claude",
        tool_category=ToolCategory.LLM_CHAT,
        domains=["claude.ai"],
        api_patterns=["/api/organizations", "/api/chat"],
        vendor="Anthropic",
    ),
    AIToolSignature(
        tool_name="Gemini",
        tool_category=ToolCategory.LLM_CHAT,
        domains=["gemini.google.com", "bard.google.com"],
        api_patterns=["/api/", "/_/BardChatUi"],
        vendor="Google",
    ),
    AIToolSignature(
        tool_name="Copilot",
        tool_category=ToolCategory.LLM_CHAT,
        domains=["copilot.microsoft.com"],
        api_patterns=["/api/chat"],
        vendor="Microsoft",
    ),
    AIToolSignature(
        tool_name="Perplexity",
        tool_category=ToolCategory.SEARCH_AI,
        domains=["perplexity.ai", "*.perplexity.ai"],
        api_patterns=["/api/", "/socket"],
        vendor="Perplexity AI",
    ),
    # Code Assistants
    AIToolSignature(
        tool_name="GitHub Copilot",
        tool_category=ToolCategory.CODE_ASSISTANT,
        domains=["github.com/features/copilot", "copilot-proxy.githubusercontent.com"],
        api_patterns=["/copilot", "/api/copilot"],
        vendor="GitHub",
        is_enterprise=True,
    ),
    AIToolSignature(
        tool_name="Cursor",
        tool_category=ToolCategory.CODE_ASSISTANT,
        domains=["cursor.sh", "*.cursor.sh"],
        vendor="Cursor",
    ),
    AIToolSignature(
        tool_name="Codeium",
        tool_category=ToolCategory.CODE_ASSISTANT,
        domains=["codeium.com", "*.codeium.com"],
        api_patterns=["/api/", "/exa"],
        vendor="Codeium",
    ),
    # Image Generation
    AIToolSignature(
        tool_name="Midjourney",
        tool_category=ToolCategory.IMAGE_GEN,
        domains=["midjourney.com", "*.midjourney.com"],
        vendor="Midjourney",
    ),
    AIToolSignature(
        tool_name="DALL-E",
        tool_category=ToolCategory.IMAGE_GEN,
        domains=["labs.openai.com"],
        api_patterns=["/api/labs"],
        vendor="OpenAI",
    ),
    AIToolSignature(
        tool_name="Stable Diffusion Online",
        tool_category=ToolCategory.IMAGE_GEN,
        domains=["stablediffusionweb.com", "stabilityai.us"],
        vendor="Stability AI",
    ),
    # Document AI
    AIToolSignature(
        tool_name="Notion AI",
        tool_category=ToolCategory.PRODUCTIVITY,
        domains=["notion.so", "*.notion.so"],
        api_patterns=["/api/v3/ai"],
        vendor="Notion",
        is_enterprise=True,
    ),
    AIToolSignature(
        tool_name="Google Docs AI",
        tool_category=ToolCategory.DOCUMENT_AI,
        domains=["docs.google.com"],
        api_patterns=["/document/d/", "/_/docos"],
        js_signatures=["docs-coral", "docs-ai"],
        vendor="Google",
        is_enterprise=True,
    ),
    # Enterprise platforms
    AIToolSignature(
        tool_name="Azure OpenAI",
        tool_category=ToolCategory.LLM_CHAT,
        domains=["*.openai.azure.com"],
        api_patterns=["/openai/deployments"],
        vendor="Microsoft",
        is_enterprise=True,
    ),
]


class AIToolDetector:
    """
    Detect AI tool usage from web activity.

    Identifies both approved enterprise tools and shadow AI
    (unapproved public tools).
    """

    def __init__(self, approved_tools: Optional[Set[str]] = None):
        """
        Initialize detector.

        Args:
            approved_tools: Set of approved tool names (case-insensitive)
        """
        self.signatures = AI_TOOL_SIGNATURES
        self.approved_tools = (
            {t.lower() for t in approved_tools} if approved_tools else set()
        )

    def detect(self, url: str) -> Optional[DetectionResult]:
        """
        Detect AI tool from URL.

        Args:
            url: Full URL to analyze

        Returns:
            DetectionResult if tool detected, None otherwise
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check each signature
        for sig in self.signatures:
            if sig.matches_domain(domain):
                is_approved = sig.tool_name.lower() in self.approved_tools
                return DetectionResult(
                    tool_name=sig.tool_name,
                    tool_category=sig.tool_category,
                    tool_domain=domain,
                    tool_approved=is_approved,
                    vendor=sig.vendor,
                    is_enterprise=sig.is_enterprise,
                    confidence=1.0,
                    detection_method="domain_match",
                )

        # Check API patterns
        for sig in self.signatures:
            if sig.matches_api(url):
                is_approved = sig.tool_name.lower() in self.approved_tools
                return DetectionResult(
                    tool_name=sig.tool_name,
                    tool_category=sig.tool_category,
                    tool_domain=domain,
                    tool_approved=is_approved,
                    vendor=sig.vendor,
                    is_enterprise=sig.is_enterprise,
                    confidence=0.9,
                    detection_method="api_pattern_match",
                )

        return None

    def is_ai_tool(self, url: str) -> bool:
        """Quick check if URL is an AI tool."""
        return self.detect(url) is not None

    def is_approved_tool(self, url: str) -> Optional[bool]:
        """
        Check if URL is an approved tool.

        Returns:
            True if approved, False if shadow AI, None if not AI tool
        """
        result = self.detect(url)
        if result is None:
            return None
        return result.tool_approved

    def detect_shadow_ai(self, url: str) -> Optional[DetectionResult]:
        """Detect shadow AI (unapproved tools) specifically."""
        result = self.detect(url)
        if result and not result.tool_approved:
            return result
        return None

    def add_approved_tool(self, tool_name: str):
        """Add a tool to approved list."""
        self.approved_tools.add(tool_name.lower())

    def remove_approved_tool(self, tool_name: str):
        """Remove a tool from approved list."""
        self.approved_tools.discard(tool_name.lower())


@dataclass
class DetectionResult:
    """Result of AI tool detection."""

    tool_name: str
    tool_category: ToolCategory
    tool_domain: str
    tool_approved: bool
    vendor: Optional[str] = None
    is_enterprise: bool = False
    confidence: float = 1.0
    detection_method: str = "unknown"
    metadata: Dict = None

    def is_shadow_ai(self) -> bool:
        """Check if this is shadow AI."""
        return not self.tool_approved


# Convenience functions


def detect_ai_tool(
    url: str, approved_tools: Optional[Set[str]] = None
) -> Optional[DetectionResult]:
    """
    Detect AI tool from URL.

    Args:
        url: URL to check
        approved_tools: Set of approved tool names

    Returns:
        DetectionResult if AI tool detected
    """
    detector = AIToolDetector(approved_tools)
    return detector.detect(url)


def is_approved_tool(url: str, approved_tools: Set[str]) -> Optional[bool]:
    """
    Check if URL is approved AI tool.

    Args:
        url: URL to check
        approved_tools: Set of approved tool names

    Returns:
        True if approved, False if shadow AI, None if not AI tool
    """
    detector = AIToolDetector(approved_tools)
    return detector.is_approved_tool(url)


def detect_shadow_ai(url: str, approved_tools: Set[str]) -> Optional[DetectionResult]:
    """
    Detect shadow AI specifically.

    Args:
        url: URL to check
        approved_tools: Set of approved tool names

    Returns:
        DetectionResult if shadow AI detected
    """
    detector = AIToolDetector(approved_tools)
    return detector.detect_shadow_ai(url)


__all__ = [
    "AIToolDetector",
    "AIToolSignature",
    "DetectionResult",
    "AI_TOOL_SIGNATURES",
    "detect_ai_tool",
    "is_approved_tool",
    "detect_shadow_ai",
]
