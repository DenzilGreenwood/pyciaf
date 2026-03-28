"""
Policy evaluation engine for CIAF Agentic Execution Boundaries.

Integrates IAM, PAM, and CIAF compliance frameworks to provide
comprehensive authorization decisions.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .engine import PolicyEngine

__all__ = ["PolicyEngine"]
