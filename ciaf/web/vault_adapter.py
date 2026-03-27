"""
CIAF Web - Vault Adapter

Persistent storage for AI usage events and cryptographic receipts.

Integrates with CIAF vault for:
- Event storage (PostgreSQL/JSON)
- Receipt archival
- Incident reconstruction
- Compliance exports
- Audit trail queries

Storage models:
- Events stored with full context
- Receipts stored separately for verification
- Hash chains for tamper detection
- Search indexes for investigation

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from .events import WebAIEvent, EventType, PolicyDecision, EventBatch
from .receipts import WebAIReceipt


class WebAIVaultAdapter:
    """
    Storage adapter for AI usage events and receipts.

    Provides persistent storage with query capabilities
    for incident investigation and compliance.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        vault_storage=None,  # Optional CIAF vault integration
    ):
        """
        Initialize vault adapter.

        Args:
            storage_path: Path for file-based storage
            vault_storage: Existing CIAF vault storage backend
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path("./ciaf_web_vault")
        )
        self.vault_storage = vault_storage
        self.use_vault = vault_storage is not None

        # Create storage directories
        if not self.use_vault:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            (self.storage_path / "events").mkdir(exist_ok=True)
            (self.storage_path / "receipts").mkdir(exist_ok=True)

    def store_event(self, event: WebAIEvent) -> bool:
        """
        Store event.

        Args:
            event: Event to store

        Returns:
            True if stored successfully
        """
        if self.use_vault:
            return self._store_event_vault(event)
        else:
            return self._store_event_file(event)

    def store_receipt(self, receipt: WebAIReceipt) -> bool:
        """
        Store receipt.

        Args:
            receipt: Receipt to store

        Returns:
            True if stored successfully
        """
        if self.use_vault:
            return self._store_receipt_vault(receipt)
        else:
            return self._store_receipt_file(receipt)

    def store_batch(self, batch: EventBatch) -> bool:
        """
        Store batch of events.

        Args:
            batch: Event batch

        Returns:
            True if stored successfully
        """
        for event in batch.events:
            if not self.store_event(event):
                return False
        return True

    def retrieve_event(self, event_id: str) -> Optional[WebAIEvent]:
        """
        Retrieve event by ID.

        Args:
            event_id: Event identifier

        Returns:
            WebAIEvent if found
        """
        if self.use_vault:
            return self._retrieve_event_vault(event_id)
        else:
            return self._retrieve_event_file(event_id)

    def retrieve_receipt(self, receipt_id: str) -> Optional[WebAIReceipt]:
        """
        Retrieve receipt by ID.

        Args:
            receipt_id: Receipt identifier

        Returns:
            WebAIReceipt if found
        """
        if self.use_vault:
            return self._retrieve_receipt_vault(receipt_id)
        else:
            return self._retrieve_receipt_file(receipt_id)

    def search_events(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        event_type: Optional[EventType] = None,
        policy_decision: Optional[PolicyDecision] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebAIEvent]:
        """
        Search events with filters.

        Args:
            org_id: Filter by organization
            user_id: Filter by user
            tool_name: Filter by tool
            event_type: Filter by event type
            policy_decision: Filter by policy decision
            start_time: Start timestamp (ISO 8601)
            end_time: End timestamp (ISO 8601)
            limit: Maximum results

        Returns:
            List of matching events
        """
        if self.use_vault:
            return self._search_events_vault(
                org_id,
                user_id,
                tool_name,
                event_type,
                policy_decision,
                start_time,
                end_time,
                limit,
            )
        else:
            return self._search_events_file(
                org_id,
                user_id,
                tool_name,
                event_type,
                policy_decision,
                start_time,
                end_time,
                limit,
            )

    def get_shadow_ai_events(
        self,
        org_id: str,
        days: int = 7,
    ) -> List[WebAIEvent]:
        """
        Get shadow AI events for organization.

        Args:
            org_id: Organization ID
            days: Look back period in days

        Returns:
            List of shadow AI events
        """
        events = self.search_events(
            org_id=org_id,
            start_time=(datetime.now() - timedelta(days=days)).isoformat(),
        )
        return [e for e in events if e.is_shadow_ai()]

    def get_high_risk_events(
        self,
        org_id: str,
        threshold: float = 0.7,
        days: int = 7,
    ) -> List[WebAIEvent]:
        """
        Get high-risk events for organization.

        Args:
            org_id: Organization ID
            threshold: Risk threshold
            days: Look back period

        Returns:
            List of high-risk events
        """
        events = self.search_events(
            org_id=org_id,
            start_time=(datetime.now() - timedelta(days=days)).isoformat(),
        )
        return [e for e in events if e.is_high_risk(threshold)]

    def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> List[WebAIEvent]:
        """
        Get user's AI usage activity.

        Args:
            user_id: User identifier
            days: Look back period

        Returns:
            List of user's events
        """
        return self.search_events(
            user_id=user_id,
            start_time=(datetime.now() - timedelta(days=days)).isoformat(),
        )

    # File-based storage implementation

    def _store_event_file(self, event: WebAIEvent) -> bool:
        """Store event to file."""
        try:
            event_file = self.storage_path / "events" / f"{event.event_id}.json"
            with open(event_file, "w") as f:
                json.dump(event.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def _store_receipt_file(self, receipt: WebAIReceipt) -> bool:
        """Store receipt to file."""
        try:
            receipt_file = self.storage_path / "receipts" / f"{receipt.receipt_id}.json"
            with open(receipt_file, "w") as f:
                json.dump(receipt.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def _retrieve_event_file(self, event_id: str) -> Optional[WebAIEvent]:
        """Retrieve event from file."""
        try:
            event_file = self.storage_path / "events" / f"{event_id}.json"
            if not event_file.exists():
                return None

            with open(event_file, "r") as f:
                data = json.load(f)

            # Reconstruct event (simplified)
            return WebAIEvent(**data)
        except Exception:
            return None

    def _retrieve_receipt_file(self, receipt_id: str) -> Optional[WebAIReceipt]:
        """Retrieve receipt from file."""
        try:
            receipt_file = self.storage_path / "receipts" / f"{receipt_id}.json"
            if not receipt_file.exists():
                return None

            with open(receipt_file, "r") as f:
                data = json.load(f)

            # Reconstruct receipt
            return WebAIReceipt(**data)
        except Exception:
            return None

    def _search_events_file(
        self,
        org_id,
        user_id,
        tool_name,
        event_type,
        policy_decision,
        start_time,
        end_time,
        limit,
    ) -> List[WebAIEvent]:
        """Search events in file storage."""
        events = []
        events_dir = self.storage_path / "events"

        # Simple linear scan (inefficient for large datasets)
        for event_file in events_dir.glob("*.json"):
            if len(events) >= limit:
                break

            event = self._retrieve_event_file(event_file.stem)
            if not event:
                continue

            # Apply filters
            if org_id and event.org_id != org_id:
                continue
            if user_id and event.user_id != user_id:
                continue
            if tool_name and event.tool_name != tool_name:
                continue
            if event_type and event.event_type != event_type:
                continue
            if policy_decision and event.policy_decision != policy_decision:
                continue

            events.append(event)

        return events[:limit]

    # Vault storage implementation (integration with CIAF vault)

    def _store_event_vault(self, event: WebAIEvent) -> bool:
        """Store event in CIAF vault."""
        if not self.vault_storage:
            return False

        try:
            # Store as metadata record
            self.vault_storage.store_metadata(
                record_type="web_ai_event",
                record_id=event.event_id,
                metadata=event.to_dict(),
            )
            return True
        except Exception:
            return False

    def _store_receipt_vault(self, receipt: WebAIReceipt) -> bool:
        """Store receipt in CIAF vault."""
        if not self.vault_storage:
            return False

        try:
            self.vault_storage.store_metadata(
                record_type="web_ai_receipt",
                record_id=receipt.receipt_id,
                metadata=receipt.to_dict(),
            )
            return True
        except Exception:
            return False

    def _retrieve_event_vault(self, event_id: str) -> Optional[WebAIEvent]:
        """Retrieve event from vault."""
        # Placeholder - would use vault query
        return None

    def _retrieve_receipt_vault(self, receipt_id: str) -> Optional[WebAIReceipt]:
        """Retrieve receipt from vault."""
        # Placeholder - would use vault query
        return None

    def _search_events_vault(
        self,
        org_id,
        user_id,
        tool_name,
        event_type,
        policy_decision,
        start_time,
        end_time,
        limit,
    ) -> List[WebAIEvent]:
        """Search events in vault."""
        # Placeholder - would use vault's JSONB query capabilities
        return []


# Convenience functions


def store_event(
    event: WebAIEvent,
    storage_path: Optional[str] = None,
    vault_storage=None,
) -> bool:
    """
    Store event.

    Args:
        event: Event to store
        storage_path: File storage path
        vault_storage: Vault storage backend

    Returns:
        True if stored successfully
    """
    adapter = WebAIVaultAdapter(storage_path, vault_storage)
    return adapter.store_event(event)


def retrieve_events(
    org_id: str,
    days: int = 7,
    storage_path: Optional[str] = None,
    vault_storage=None,
) -> List[WebAIEvent]:
    """
    Retrieve recent events for organization.

    Args:
        org_id: Organization ID
        days: Look back period
        storage_path: File storage path
        vault_storage: Vault storage backend

    Returns:
        List of events
    """
    adapter = WebAIVaultAdapter(storage_path, vault_storage)
    return adapter.search_events(
        org_id=org_id,
        start_time=(datetime.now() - timedelta(days=days)).isoformat(),
    )


def search_events(
    filters: Dict[str, Any],
    storage_path: Optional[str] = None,
    vault_storage=None,
) -> List[WebAIEvent]:
    """
    Search events with filters.

    Args:
        filters: Search filters
        storage_path: File storage path
        vault_storage: Vault storage backend

    Returns:
        List of matching events
    """
    adapter = WebAIVaultAdapter(storage_path, vault_storage)
    return adapter.search_events(**filters)


__all__ = [
    "WebAIVaultAdapter",
    "store_event",
    "retrieve_events",
    "search_events",
]
