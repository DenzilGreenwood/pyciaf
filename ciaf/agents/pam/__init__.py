"""
Privileged Access Management (PAM) for CIAF agents.

Provides just-in-time (JIT) privilege elevation with approval workflows
and time-bound access grants.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .store import PAMStore

__all__ = ["PAMStore"]
