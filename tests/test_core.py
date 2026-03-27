"""
Minimal test scaffold for CIAF core functionality.

Tests the key components using drop-in tests as requested.
"""

import sys
import os

# Add the project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import directly from core modules to avoid API framework
from ciaf.core.merkle import MerkleTree
from ciaf.core.crypto import sha256_hash
from ciaf.core.canonicalization import (
    Policy,
    make_anchor,
    CapsuleBuilder,
    create_production_signer,
)
from ciaf.lcm.dataset_manager import LCMDatasetAnchor, DatasetMetadata, DatasetSplit
from ciaf.core.enums import RecordType


def test_merkle_proof_roundtrip():
    """Test Merkle tree proof generation and verification."""
    leaves = [sha256_hash(f"m{i}".encode()) for i in range(5)]
    tree = MerkleTree(leaves)
    root = tree.root
    for leaf in leaves:
        proof = tree.get_proof(leaf)
        assert MerkleTree.verify_proof(leaf, root, proof)


def test_anchor_sign_verify():
    """Test anchor signature creation and verification."""
    signer = create_production_signer("k1")
    policy = Policy(policy_id="pol1", schema_version="1.0", domain_labels=["test"])
    anchor = make_anchor(root="ab" * 32, policy=policy, signer=signer)
    assert signer.verify(anchor.get_anchor_bytes(), anchor.signature)


def test_capsule_builder_signature_flag():
    """Test CapsuleBuilder with signature verification flag."""
    signer = create_production_signer("k2")
    policy = Policy(policy_id="pol2", schema_version="1.0", domain_labels=["test"])
    root = "cd" * 32
    anchor = make_anchor(root=root, policy=policy, signer=signer)
    leaves = [sha256_hash(b"x"), sha256_hash(b"y")]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(leaves[0])

    cap = CapsuleBuilder.build(
        metadata={"foo": "bar"},
        merkle_path=proof,
        anchor=anchor,
        record_type=RecordType.DATASET,
        leaf_hash=leaves[0],
        verify_signature=True,
        public_key_pem=signer.get_public_key_pem(),
    )
    assert cap["proofs"]["inclusion_proof_valid"] is True
    assert cap["verification"]["signature_valid"] is True


def test_dataset_anchor_serde():
    """Test LCMDatasetAnchor functionality."""
    metadata = DatasetMetadata(
        name="test_dataset",
        version="1.0",
        description="Test dataset",
        features=["a", "b", "c"],
        total_samples=3,
    )

    ds = LCMDatasetAnchor(
        dataset_id="D1",
        split=DatasetSplit.TRAIN,
        metadata=metadata,
        master_password="M1",
    )

    for i in range(3):
        ds.add_sample_hash(sha256_hash(f"sample_{i}".encode()))

    root1 = ds.get_merkle_root()
    assert root1 is not None

    # Test JSON serialization
    js = ds.to_json()
    assert "dataset_id" in js
    assert "split" in js


if __name__ == "__main__":
    test_merkle_proof_roundtrip()
    print("PASS: test_merkle_proof_roundtrip")

    test_anchor_sign_verify()
    print("PASS: test_anchor_sign_verify")

    test_capsule_builder_signature_flag()
    print("PASS: test_capsule_builder_signature_flag")

    test_dataset_anchor_serde()
    print("PASS: test_dataset_anchor_serde")

    print("All tests passed!")
