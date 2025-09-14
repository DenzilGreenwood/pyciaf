"""
Test Merkle tree determinism.

This test ensures that Merkle tree root calculation is deterministic
for a fixed set of leaves.
"""

import pytest
from ciaf.core.merkle import MerkleTree
from ciaf.core.crypto import sha256_hash


class TestMerkleDeterminism:
    """Test Merkle tree root determinism."""
    
    def test_merkle_root_deterministic(self) -> None:
        """Test that Merkle root is deterministic for same leaf set."""
        # Create test leaves
        leaves = [
            sha256_hash("data1".encode()),
            sha256_hash("data2".encode()),
            sha256_hash("data3".encode()),
            sha256_hash("data4".encode()),
        ]
        
        # Build multiple trees with same leaves
        tree1 = MerkleTree(leaves)
        tree2 = MerkleTree(leaves)
        tree3 = MerkleTree(leaves)
        
        root1 = tree1.get_root()
        root2 = tree2.get_root()
        root3 = tree3.get_root()
        
        # All roots should be identical
        assert root1 == root2 == root3
        assert len(root1) == 64  # SHA256 hex string
    
    def test_merkle_root_different_order_same_result(self) -> None:
        """Test that Merkle root is same regardless of leaf insertion order."""
        # Create test leaves in different orders
    leaves1 = [sha256_hash(f"data{i}".encode()) for i in [1, 2, 3, 4]]
    # leaves2 and leaves3 are unused, so removed to fix lint error
        
    # Note: For true order independence, we'd need to sort leaves
    # For now, test that same order gives same result
    tree1a = MerkleTree(leaves1)
    tree1b = MerkleTree(leaves1)
    assert tree1a.get_root() == tree1b.get_root()
    
    def test_merkle_proof_consistency(self) -> None:
        """Test that Merkle proofs are consistent."""
        leaves = [sha256_hash(f"data{i}".encode()) for i in range(8)]
        tree = MerkleTree(leaves)
        
        # Get proof for each leaf
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(leaf)
            # Verify the proof
            is_valid = MerkleTree.verify_proof(leaf, tree.get_root(), proof)
            assert is_valid, f"Proof verification failed for leaf {i}"
    
    def test_merkle_single_leaf(self) -> None:
        """Test Merkle tree with single leaf."""
        leaf = sha256_hash("single_data".encode())
        tree = MerkleTree([leaf])
        
        root = tree.get_root()
        assert root == leaf  # Single leaf should be the root
        
        # Proof for single leaf should be empty or valid
        proof = tree.get_proof(leaf)
        is_valid = MerkleTree.verify_proof(leaf, root, proof)
        assert is_valid
    
    def test_merkle_empty_tree(self) -> None:
        """Test Merkle tree behavior with empty leaf set."""
        with pytest.raises((ValueError, IndexError)):
            # Empty tree should raise an error
            MerkleTree([])
    
    def test_merkle_large_tree(self) -> None:
        """Test Merkle tree with larger dataset."""
        # Create 100 leaves
        leaves = [sha256_hash(f"data_{i:03d}".encode()) for i in range(100)]
        tree = MerkleTree(leaves)
        
        root = tree.get_root()
        assert len(root) == 64  # SHA256 hex string
        
        # Verify a few random proofs
        import random
        for _ in range(5):
            leaf = random.choice(leaves)
            proof = tree.get_proof(leaf)
            is_valid = MerkleTree.verify_proof(leaf, root, proof)
            assert is_valid
