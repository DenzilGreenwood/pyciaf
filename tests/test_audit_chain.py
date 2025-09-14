"""
Test audit trail chain validation.

This test ensures that audit trail chains maintain integrity
across train → inference cycles.
"""

import pytest
from datetime import datetime
from ciaf.core.crypto import sha256_hash


class MockAuditRecord:
    """Mock audit record for testing."""
    
    def __init__(self, event_id: str, event_type: str, timestamp: str, previous_hash: str = None):
        self.event_id = event_id
        self.event_type = event_type
        self.timestamp = timestamp
        self.previous_hash = previous_hash or "0" * 64
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash for this record."""
        data = f"{self.event_id}|{self.event_type}|{self.timestamp}|{self.previous_hash}"
        return sha256_hash(data.encode())


class TestAuditChain:
    """Test audit trail chain validation."""
    
    def test_audit_chain_integrity(self):
        """Test that audit chain maintains integrity through multiple events."""
        # Create audit chain
        records = []
        
        # Genesis record (training start)
        genesis = MockAuditRecord(
            event_id="train_001",
            event_type="training_started",
            timestamp="2025-09-12T10:00:00Z"
        )
        records.append(genesis)
        
        # Training completion
        train_complete = MockAuditRecord(
            event_id="train_002",
            event_type="training_completed",
            timestamp="2025-09-12T10:30:00Z",
            previous_hash=genesis.hash
        )
        records.append(train_complete)
        
        # First inference
        inference1 = MockAuditRecord(
            event_id="infer_001",
            event_type="inference_performed",
            timestamp="2025-09-12T11:00:00Z",
            previous_hash=train_complete.hash
        )
        records.append(inference1)
        
        # Second inference
        inference2 = MockAuditRecord(
            event_id="infer_002",
            event_type="inference_performed",
            timestamp="2025-09-12T11:15:00Z",
            previous_hash=inference1.hash
        )
        records.append(inference2)
        
        # Validate chain integrity
        assert self._validate_chain(records)
    
    def test_audit_chain_tampering_detection(self):
        """Test that tampering is detected in audit chain."""
        # Create valid chain
        record1 = MockAuditRecord("event_1", "type_1", "2025-09-12T10:00:00Z")
        record2 = MockAuditRecord("event_2", "type_2", "2025-09-12T10:01:00Z", record1.hash)
        record3 = MockAuditRecord("event_3", "type_3", "2025-09-12T10:02:00Z", record2.hash)
        
        valid_chain = [record1, record2, record3]
        assert self._validate_chain(valid_chain)
        
        # Tamper with middle record
        record2_tampered = MockAuditRecord("event_2_modified", "type_2", "2025-09-12T10:01:00Z", record1.hash)
        tampered_chain = [record1, record2_tampered, record3]
        
        # Chain should be invalid due to broken link
        assert not self._validate_chain(tampered_chain)
    
    def test_audit_chain_missing_link(self):
        """Test detection of missing links in audit chain."""
        record1 = MockAuditRecord("event_1", "type_1", "2025-09-12T10:00:00Z")
        record2 = MockAuditRecord("event_2", "type_2", "2025-09-12T10:01:00Z", record1.hash)
        # Skip record, creating gap
        record4 = MockAuditRecord("event_4", "type_4", "2025-09-12T10:03:00Z", "fake_hash")
        
        broken_chain = [record1, record2, record4]
        
        # Chain should be invalid due to missing link
        assert not self._validate_chain(broken_chain)
    
    def test_audit_chain_single_record(self):
        """Test validation of single record (genesis)."""
        genesis = MockAuditRecord("genesis", "system_init", "2025-09-12T09:00:00Z")
        single_chain = [genesis]
        
        assert self._validate_chain(single_chain)
    
    def test_audit_chain_duplicate_detection(self):
        """Test detection of duplicate records in chain."""
        record1 = MockAuditRecord("event_1", "type_1", "2025-09-12T10:00:00Z")
        record2 = MockAuditRecord("event_2", "type_2", "2025-09-12T10:01:00Z", record1.hash)
        
        # Duplicate the second record
        duplicate_chain = [record1, record2, record2]
        
        # Should detect duplicate
        assert not self._validate_chain_no_duplicates(duplicate_chain)
    
    def _validate_chain(self, records) -> bool:
        """Validate audit chain integrity."""
        if not records:
            return False
        
        if len(records) == 1:
            return True
        
        for i in range(1, len(records)):
            current = records[i]
            previous = records[i - 1]
            
            # Check that current record references previous correctly
            if current.previous_hash != previous.hash:
                return False
            
            # Verify hash integrity of current record
            expected_hash = sha256_hash(
                f"{current.event_id}|{current.event_type}|{current.timestamp}|{current.previous_hash}".encode()
            )
            if current.hash != expected_hash:
                return False
        
        return True
    
    def _validate_chain_no_duplicates(self, records) -> bool:
        """Check for duplicate records in chain."""
        seen_hashes = set()
        for record in records:
            if record.hash in seen_hashes:
                return False
            seen_hashes.add(record.hash)
        return True