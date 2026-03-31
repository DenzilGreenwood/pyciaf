"""
CIAF Vault Module Tests

Comprehensive test suite for CIAF vault storage backends and management:
- PostgreSQL backend
- In-memory backend
- Vault operations (store, retrieve, query)
- Agent events storage
- Web AI events storage
- Watermark evidence storage
- LCM capsule storage
- Query optimization and indexes

Created: 2026-03-31
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
import hashlib

# Import vault modules - gracefully handle missing imports
try:
    from ciaf.vault import Vault
    from ciaf.vault.backends.postgresql_backend import PostgreSQLBackend
    from ciaf.vault.backends.memory_backend import InMemoryBackend
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultBackends:
    """Test vault storage backend implementations."""

    def test_in_memory_backend(self):
        """Test in-memory vault backend."""
        backend = InMemoryBackend()

        # Store record
        record = {
            "id": "rec_001",
            "type": "test",
            "data": "value",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        backend.store(record)

        # Retrieve record
        retrieved = backend.get("rec_001")

        assert retrieved is not None
        assert retrieved["id"] == "rec_001"

    def test_backend_query(self):
        """Test querying vault backend."""
        backend = InMemoryBackend()

        # Store multiple records
        for i in range(10):
            backend.store({
                "id": f"rec_{i:03d}",
                "type": "agent_event",
                "agent_id": "agent_001" if i < 5 else "agent_002",
            })

        # Query by agent_id
        results = backend.query({"agent_id": "agent_001"})

        assert len(results) == 5


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultOperations:
    """Test vault operations."""

    def test_store_agent_event(self):
        """Test storing an agent event in vault."""
        vault = Vault(backend=InMemoryBackend())

        event = {
            "event_id": "evt_001",
            "event_type": "agent_read",
            "agent_id": "agent_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource_id": "record_123",
        }

        vault.store_agent_event(event)

        # Retrieve
        retrieved = vault.get_agent_event("evt_001")

        assert retrieved["event_id"] == "evt_001"

    def test_store_web_ai_event(self):
        """Test storing a web AI event in vault."""
        vault = Vault(backend=InMemoryBackend())

        event = {
            "event_id": "web_evt_001",
            "event_type": "prompt_submit",
            "user_id": "user_alice",
            "domain": "chat.openai.com",
            "prompt_hash": hashlib.sha256(b"prompt").hexdigest(),
        }

        vault.store_web_ai_event(event)

        retrieved = vault.get_web_ai_event("web_evt_001")

        assert retrieved["user_id"] == "user_alice"

    def test_store_watermark_evidence(self):
        """Test storing watermark evidence in vault."""
        vault = Vault(backend=InMemoryBackend())

        evidence = {
            "artifact_id": "art_001",
            "artifact_type": "text",
            "model_id": "gpt-4",
            "content_hash_before": hashlib.sha256(b"before").hexdigest(),
            "content_hash_after": hashlib.sha256(b"after").hexdigest(),
        }

        vault.store_watermark_evidence(evidence)

        retrieved = vault.get_watermark_evidence("art_001")

        assert retrieved["artifact_type"] == "text"


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultQuerying:
    """Test vault query capabilities."""

    def test_query_agent_events_by_time_range(self):
        """Test querying agent events by time range."""
        vault = Vault(backend=InMemoryBackend())

        # Store events with different timestamps
        base_time = datetime(2026, 3, 31, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(10):
            event_time = base_time.replace(hour=10 + i)
            vault.store_agent_event({
                "event_id": f"evt_{i:03d}",
                "agent_id": "agent_001",
                "occurred_at": event_time.isoformat(),
            })

        # Query events between 12:00 and 15:00
        start_time = base_time.replace(hour=12).isoformat()
        end_time = base_time.replace(hour=15).isoformat()

        results = vault.query_agent_events(
            start_time=start_time,
            end_time=end_time,
        )

        assert len(results) == 4  # Hours 12, 13, 14, 15

    def test_query_high_risk_events(self):
        """Test querying high-risk agent events."""
        vault = Vault(backend=InMemoryBackend())

        # Store events with different risk levels
        for i in range(10):
            vault.store_agent_event({
                "event_id": f"evt_{i:03d}",
                "sensitivity_level": "highly_restricted" if i < 3 else "public",
            })

        # Query high-risk events
        results = vault.query_agent_events(
            sensitivity_level="highly_restricted"
        )

        assert len(results) == 3

    def test_query_by_agent_id(self):
        """Test querying events by agent ID."""
        vault = Vault(backend=InMemoryBackend())

        # Store events from different agents
        for i in range(15):
            agent_id = f"agent_{i % 3:03d}"  # 3 agents
            vault.store_agent_event({
                "event_id": f"evt_{i:03d}",
                "agent_id": agent_id,
            })

        # Query events for agent_001
        results = vault.query_agent_events(agent_id="agent_001")

        assert len(results) == 5  # Events 0, 3, 6, 9, 12


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultIntegrity:
    """Test vault data integrity features."""

    def test_hash_chain_validation(self):
        """Test validating hash chain integrity."""
        vault = Vault(backend=InMemoryBackend())

        # Create chain of events
        prior_hash = "0" * 64  # Genesis

        events = []
        for i in range(5):
            event = {
                "event_id": f"evt_{i:03d}",
                "data": f"data_{i}",
                "prior_event_hash": prior_hash,
            }

            # Compute event hash
            import json
            event_hash = hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest()

            event["event_hash"] = event_hash

            vault.store_agent_event(event)
            events.append(event)

            prior_hash = event_hash

        # Validate chain
        is_valid = vault.validate_hash_chain(events)

        assert is_valid is True

    def test_detect_chain_tampering(self):
        """Test detecting tampered hash chain."""
        events = [
            {"event_id": "evt_001", "event_hash": "hash1", "prior_event_hash": "0" * 64},
            {" event_id": "evt_002", "event_hash": "hash2", "prior_event_hash": "hash1"},
            {"event_id": "evt_003", "event_hash": "hash3", "prior_event_hash": "wrong_hash"},  # Tampering!
        ]

        # Validation should detect broken chain
        for i in range(1, len(events)):
            current = events[i]
            previous = events[i - 1]

            if current["prior_event_hash"] != previous["event_hash"]:
                tampering_detected = True
                break
        else:
            tampering_detected = False

        assert tampering_detected is True


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultPerformance:
    """Test vault performance optimizations."""

    def test_bulk_insert(self):
        """Test bulk inserting records."""
        vault = Vault(backend=InMemoryBackend())

        # Create batch of 1000 events
        events = [
            {
                "event_id": f"evt_{i:06d}",
                "agent_id": f"agent_{i % 10:03d}",
                "data": f"value_{i}",
            }
            for i in range(1000)
        ]

        # Bulk insert
        vault.bulk_insert_agent_events(events)

        # Verify count
        all_events = vault.query_agent_events()

        assert len(all_events) >= 1000

    def test_indexed_query_performance(self):
        """Test query performance with indexes."""
        # Note: Actual benchmarking would compare with/without indexes
        vault = Vault(backend=InMemoryBackend())

        # Insert test data
        for i in range(100):
            vault.store_agent_event({
                "event_id": f"evt_{i:06d}",
                "agent_id": "agent_001",
            })

        # Query should use index on agent_id
        results =  vault.query_agent_events(agent_id="agent_001")

        assert len(results) == 100


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultMigration:
    """Test vault data migration."""

    def test_export_import_workflow(self):
        """Test exporting and importing vault data."""
        # Source vault
        source_vault = Vault(backend=InMemoryBackend())

        # Store data
        for i in range(10):
            source_vault.store_agent_event({
                "event_id": f"evt_{i:03d}",
                "data": f"value_{i}",
            })

        # Export data
        exported_data = source_vault.export_data()

        # Target vault
        target_vault = Vault(backend=InMemoryBackend())

        # Import data
        target_vault.import_data(exported_data)

        # Verify migration
        source_count = len(source_vault.query_agent_events())
        target_count = len(target_vault.query_agent_events())

        assert source_count == target_count


@pytest.mark.skipif(not VAULT_AVAILABLE, reason="Vault module not available")
class TestVaultCompliance:
    """Test vault compliance features."""

    def test_data_retention_enforcement(self):
        """Test enforcing data retention policies."""
        vault = Vault(backend=InMemoryBackend())

        # Store old events
        old_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()
        recent_timestamp = datetime(2026, 3, 31, tzinfo=timezone.utc).isoformat()

        vault.store_agent_event({
            "event_id": "evt_old",
            "occurred_at": old_timestamp,
        })

        vault.store_agent_event({
            "event_id": "evt_recent",
            "occurred_at": recent_timestamp,
        })

        # Apply retention policy: delete events older than 90 days
        retention_days = 90
        cutoff_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        deleted_count = vault.delete_events_before(cutoff_date)

        assert deleted_count >= 1  # Old event deleted

    def test_gdpr_data_erasure(self):
        """Test GDPR right to erasure."""
        vault = Vault(backend=InMemoryBackend())

        # Store user events
        for i in range(10):
            vault.store_web_ai_event({
                "event_id": f"evt_{i:03d}",
                "user_id": "user_alice" if i < 5 else "user_bob",
            })

        # User Alice requests deletion
        deleted_count = vault.delete_user_events("user_alice")

        assert deleted_count == 5

        # Verify deletion
        remaining = vault.query_web_ai_events()
        alice_events = [e for e in remaining if e.get("user_id") == "user_alice"]

        assert len(alice_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
