"""
Evidence Strength Tracking for CIAF

Implements fallback disclosure and evidence strength tracking to ensure
audit trails clearly distinguish between real and simulated evidence.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

class EvidenceStrength(Enum):
    """Evidence strength levels for audit trails."""
    REAL = "real"
    SIMULATED = "simulated"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"

@dataclass
class FallbackReason:
    """Details about why fallback was used."""
    component: str
    reason: str
    missing_sources: list
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "reason": self.reason,
            "missing_sources": self.missing_sources,
            "timestamp": self.timestamp
        }

class EvidenceTracker:
    """Tracks evidence strength and fallback reasons across CIAF operations."""
    
    def __init__(self):
        self.fallback_log = []
        
    def record_fallback(self, component: str, reason: str, missing_sources: list) -> FallbackReason:
        """Record a fallback event with full context."""
        fallback = FallbackReason(
            component=component,
            reason=reason,
            missing_sources=missing_sources,
            timestamp=datetime.utcnow().isoformat()
        )
        self.fallback_log.append(fallback)
        return fallback
    
    def determine_evidence_strength(self, component_states: Dict[str, bool]) -> EvidenceStrength:
        """Determine overall evidence strength based on component states."""
        if all(component_states.values()):
            return EvidenceStrength.REAL
        elif any(component_states.values()):
            return EvidenceStrength.FALLBACK
        else:
            return EvidenceStrength.SIMULATED
    
    def get_fallback_summary(self) -> Dict[str, Any]:
        """Get summary of all fallback events."""
        return {
            "total_fallbacks": len(self.fallback_log),
            "components_affected": list(set(f.component for f in self.fallback_log)),
            "latest_fallbacks": [f.to_dict() for f in self.fallback_log[-5:]]
        }

# Global evidence tracker instance
_evidence_tracker = EvidenceTracker()

def get_evidence_tracker() -> EvidenceTracker:
    """Get the global evidence tracker."""
    return _evidence_tracker

def record_fallback(component: str, reason: str, missing_sources: list = None) -> FallbackReason:
    """Record a fallback event globally."""
    if missing_sources is None:
        missing_sources = []
    return _evidence_tracker.record_fallback(component, reason, missing_sources)

def get_evidence_strength(component_states: Dict[str, bool]) -> EvidenceStrength:
    """Determine evidence strength for current operation."""
    return _evidence_tracker.determine_evidence_strength(component_states)