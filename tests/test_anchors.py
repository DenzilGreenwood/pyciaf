"""
Test anchor derivation determinism.

This test ensures that anchor derivation is deterministic and reproducible
across multiple runs with the same inputs.
"""

import pytest
from ciaf.core.crypto import derive_anchor_from_master, sha256_hash


class TestAnchorDeterminism:
    """Test anchor derivation determinism."""
    
    def test_master_to_dataset_anchor_deterministic(self):
        """Test that master to dataset anchor derivation is deterministic."""
        master_password = "test_master_password"
        dataset_id = "test_dataset"
        
        # Derive anchor multiple times
        anchor1 = derive_anchor_from_master(master_password, dataset_id)
        anchor2 = derive_anchor_from_master(master_password, dataset_id)
        anchor3 = derive_anchor_from_master(master_password, dataset_id)
        
        # All derivations should be identical
        assert anchor1 == anchor2 == anchor3
        assert len(anchor1) > 0
    
    def test_dataset_to_capsule_anchor_deterministic(self):
        """Test that dataset to capsule anchor derivation is deterministic."""
        dataset_anchor = b"test_dataset_anchor_bytes"
        capsule_id = "capsule_001"
        
        # Derive capsule anchor multiple times
        capsule_anchor1 = sha256_hash(dataset_anchor + capsule_id.encode())
        capsule_anchor2 = sha256_hash(dataset_anchor + capsule_id.encode())
        capsule_anchor3 = sha256_hash(dataset_anchor + capsule_id.encode())
        
        # All derivations should be identical
        assert capsule_anchor1 == capsule_anchor2 == capsule_anchor3
        assert len(capsule_anchor1) == 64  # SHA256 hex string length
    
    def test_different_inputs_produce_different_anchors(self):
        """Test that different inputs produce different anchors."""
        master_password = "test_master_password"
        
        anchor1 = derive_anchor_from_master(master_password, "dataset1")
        anchor2 = derive_anchor_from_master(master_password, "dataset2")
        anchor3 = derive_anchor_from_master("different_password", "dataset1")
        
        # All anchors should be different
        assert anchor1 != anchor2
        assert anchor1 != anchor3
        assert anchor2 != anchor3
    
    def test_anchor_format_consistency(self):
        """Test that anchors have consistent format."""
        master_password = "test_master_password"
        dataset_id = "test_dataset"
        
        anchor = derive_anchor_from_master(master_password, dataset_id)
        
        # Anchor should be a non-empty string
        assert isinstance(anchor, str)
        assert len(anchor) > 0
        # Should be hex-encoded (SHA256 = 64 chars)
        assert len(anchor) == 64
        assert all(c in "0123456789abcdef" for c in anchor.lower())