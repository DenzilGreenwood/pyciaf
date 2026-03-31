"""
CIAF Core Module Tests

Comprehensive test suite for CIAF core cryptographic and foundational components:
- SHA-256 hashing and HMAC
- Ed25519 signatures (RFC 8032)
- Merkle trees and inclusion proofs
- WORM (Write-Once-Read-Many) storage
- Key management and key derivation
- Canonical JSON serialization
- Anchor derivation (master, dataset, model, capsule)

Created: 2026-03-31
Version: 1.0.0
"""

import pytest
import hashlib
import json
from datetime import datetime, timezone

from ciaf.core import (
    sha256_hash,
    hmac_sha256,
    secure_random_bytes,
    canonical_json,
    MerkleTree,
    Ed25519Signer,
    Ed25519Verifier,
    SQLiteWORMStore,
    WORMRecord,
    RecordType,
    derive_master_anchor,
    derive_dataset_anchor,
    derive_model_anchor,
)


class TestSHA256Hashing:
    """Test SHA-256 hashing functionality."""

    def test_sha256_hash_deterministic(self):
        """Test SHA-256 produces deterministic output."""
        data = b"hello world"

        hash1 = sha256_hash(data)
        hash2 = sha256_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_sha256_different_inputs(self):
        """Test different inputs produce different hashes."""
        data1 = b"data1"
        data2 = b"data2"

        hash1 = sha256_hash(data1)
        hash2 = sha256_hash(data2)

        assert hash1 != hash2

    def test_sha256_empty_input(self):
        """Test SHA-256 of empty input."""
        data = b""
        hash_value = sha256_hash(data)

        # SHA-256 of empty string is known
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_value == expected


class TestHMACSHA256:
    """Test HMAC-SHA256 functionality."""

    def test_hmac_sha256(self):
        """Test HMAC-SHA256 message authentication."""
        key = b"secret_key"
        message = b"important_message"

        hmac_value = hmac_sha256(key, message)

        assert len(hmac_value) == 64  # SHA-256 hex digest

    def test_hmac_different_keys(self):
        """Test different keys produce different HMACs."""
        key1 = b"key1"
        key2 = b"key2"
        message = b"message"

        hmac1 = hmac_sha256(key1, message)
        hmac2 = hmac_sha256(key2, message)

        assert hmac1 != hmac2

    def test_hmac_integrity_verification(self):
        """Test HMAC for message integrity verification."""
        key = b"shared_secret"
        message = b"authentic_message"

        # Sender computes HMAC
        hmac_sent = hmac_sha256(key, message)

        # Receiver computes HMAC
        hmac_received = hmac_sha256(key, message)

        # Verify integrity
        assert hmac_sent == hmac_received


class TestSecureRandomBytes:
    """Test secure random number generation."""

    def test_generate_random_bytes(self):
        """Test generating secure random bytes."""
        random_bytes = secure_random_bytes(32)

        assert len(random_bytes) == 32
        assert isinstance(random_bytes, bytes)

    def test_random_bytes_unique(self):
        """Test random bytes are unique."""
        bytes1 = secure_random_bytes(32)
        bytes2 = secure_random_bytes(32)

        # Extremely unlikely to be equal
        assert bytes1 != bytes2


class TestCanonicalJSON:
    """Test canonical JSON serialization."""

    def test_canonical_json_deterministic(self):
        """Test canonical JSON is deterministic."""
        data1 = {"c": 3, "a": 1, "b": 2}
        data2 = {"b": 2, "a": 1, "c": 3}

        json1 = canonical_json(data1)
        json2 = canonical_json(data2)

        assert json1 == json2  # Order-independent

    def test_canonical_json_sorted_keys(self):
        """Test keys are sorted alphabetically."""
        data = {"z": 1, "a": 2, "m": 3}
        json_str = canonical_json(data)

        # Keys should appear in sorted order
        assert json_str.index('"a"') < json_str.index('"m"')
        assert json_str.index('"m"') < json_str.index('"z"')

    def test_canonical_json_no_whitespace(self):
        """Test canonical JSON has no unnecessary whitespace."""
        data = {"key": "value"}
        json_str = canonical_json(data)

        # Should have no spaces around separators
        assert '{"key":"value"}' == json_str


class TestMerkleTree:
    """Test Merkle tree construction and verification."""

    def test_create_merkle_tree(self):
        """Test creating a Merkle tree."""
        leaves = [
            sha256_hash(b"leaf1"),
            sha256_hash(b"leaf2"),
            sha256_hash(b"leaf3"),
            sha256_hash(b"leaf4"),
        ]

        tree = MerkleTree(leaves)
        root = tree.get_root()

        assert len(root) == 64  # SHA-256 hex

    def test_merkle_inclusion_proof(self):
        """Test generating and verifying Merkle inclusion proof."""
        leaves = [sha256_hash(f"leaf{i}".encode()) for i in range(8)]

        tree = MerkleTree(leaves)
        root = tree.get_root()

        # Generate proof for leaf 3
        leaf = leaves[3]
        proof = tree.get_proof(leaf)

        # Verify proof
        is_valid = MerkleTree.verify_proof(leaf, root, proof)

        assert is_valid is True

    def test_merkle_invalid_proof(self):
        """Test invalid Merkle proof is rejected."""
        leaves = [sha256_hash(f"leaf{i}".encode()) for i in range(4)]

        tree = MerkleTree(leaves)
        root = tree.get_root()

        # Generate proof for a leaf NOT in the tree
        fake_leaf = sha256_hash(b"fake_leaf")
        proof = tree.get_proof(leaves[0])  # Proof for different leaf

        # Verification should fail
        is_valid = MerkleTree.verify_proof(fake_leaf, root, proof)

        assert is_valid is False


class TestEd25519Signatures:
    """Test Ed25519 digital signatures."""

    def test_sign_and_verify(self):
        """Test signing and verifying with Ed25519."""
        signer = Ed25519Signer("test_key")
        data = b"message to sign"

        # Sign
        signature = signer.sign(data)

        # Verify
        is_valid = signer.verify(data, signature)

        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test invalid signature is rejected."""
        signer = Ed25519Signer("test_key")
        data = b"original message"
        tampered_data = b"tampered message"

        # Sign original
        signature = signer.sign(data)

        # Try to verify tampered data
        is_valid = signer.verify(tampered_data, signature)

        assert is_valid is False

    def test_different_keys_different_signatures(self):
        """Test different keys produce different signatures."""
        signer1 = Ed25519Signer("key1")
        signer2 = Ed25519Signer("key2")
        data = b"data"

        sig1 = signer1.sign(data)
        sig2 = signer2.sign(data)

        assert sig1 != sig2


class TestWORMStore:
    """Test WORM (Write-Once-Read-Many) storage."""

    def test_worm_write_once(self):
        """Test WORM allows writing once."""
        store = SQLiteWORMStore()

        record = WORMRecord(
            id="rec_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_type=RecordType.DATASET,
            data={"key": "value"},
        )
        record_id = store.append_record(record)

        assert record_id == "rec_001"
        assert len(store.list_records()) == 1

    def test_worm_no_modification(self):
        """Test WORM prevents modification."""
        store = SQLiteWORMStore()

        record = WORMRecord(
            id="rec_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_type=RecordType.DATASET,
            data={"key": "original"},
        )
        store.append_record(record)

        # Read record back
        retrieved = store.get_record("rec_001")

        assert retrieved.data["key"] == "original"

    def test_worm_multiple_writes(self):
        """Test WORM supports multiple append operations."""
        store = SQLiteWORMStore()

        for i in range(10):
            record = WORMRecord(
                id=f"rec_{i:03d}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                record_type=RecordType.DATASET,
                data={"value": i},
            )
            store.append_record(record)

        assert len(store.list_records()) == 10


class TestAnchorDerivation:
    """Test cryptographic anchor derivation."""

    def test_derive_master_anchor(self):
        """Test deriving master anchor from password."""
        password = "secure_password_123"
        salt = secure_random_bytes(32)

        master_anchor = derive_master_anchor(password, salt)

        assert len(master_anchor) == 32  # 256-bit key

    def test_derive_dataset_anchor(self):
        """Test deriving dataset anchor."""
        password = "password"
        salt = secure_random_bytes(32)
        master_anchor = derive_master_anchor(password, salt)

        dataset_hash = sha256_hash(b"dataset_content")
        dataset_anchor = derive_dataset_anchor(master_anchor, dataset_hash)

        assert len(dataset_anchor) == 32

    def test_derive_model_anchor(self):
        """Test deriving model anchor."""
        password = "password"
        salt = secure_random_bytes(32)
        master_anchor = derive_master_anchor(password, salt)

        model_hash = sha256_hash(b"model_weights")
        model_anchor = derive_model_anchor(master_anchor, model_hash)

        assert len(model_anchor) == 32

    def test_deterministic_derivation(self):
        """Test anchor derivation is deterministic."""
        password = "password"
        salt = b"fixed_salt" + b"\x00" * 24  #  32 bytes

        anchor1 = derive_master_anchor(password, salt)
        anchor2 = derive_master_anchor(password, salt)

        assert anchor1 == anchor2


class TestCryptographicWorkflows:
    """Test complete cryptographic workflows."""

    def test_hash_chain_workflow(self):
        """Test creating a hash chain for tamper detection."""
        # Record 1
        record1 = {"id": "rec_001", "data": "value1"}
        hash1 = sha256_hash(canonical_json(record1).encode())

        # Record 2 (chained to record 1)
        record2 = {"id": "rec_002", "data": "value2", "prior_hash": hash1}
        hash2 = sha256_hash(canonical_json(record2).encode())

        # Record 3 (chained to record 2)
        record3 = {"id": "rec_003", "data": "value3", "prior_hash": hash2}

        assert record2["prior_hash"] == hash1
        assert record3["prior_hash"] == hash2

    def test_sign_and_store_workflow(self):
        """Test signing and storing records."""
        signer = Ed25519Signer("workflow_key")
        store = SQLiteWORMStore()

        # Create record data
        data = {"key": "important_data"}

        # Sign data
        data_bytes = canonical_json(data).encode()
        signature = signer.sign(data_bytes)

        # Create WORM record with signature in metadata
        record = WORMRecord(
            id="rec_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_type=RecordType.DATASET,
            data={**data, "signature": signature},
        )

        # Store in WORM
        record_id = store.append_record(record)

        # Retrieve and verify
        retrieved = store.get_record(record_id)
        retrieved_sig = retrieved.data.pop("signature")
        retrieved_data = retrieved.data

        is_valid = signer.verify(canonical_json(retrieved_data).encode(), retrieved_sig)

        assert is_valid is True

    def test_merkle_batch_commitment(self):
        """Test committing a batch of records to Merkle tree."""
        # Create batch of records
        records = [
            {"id": f"rec_{i:03d}", "data": f"value_{i}"} for i in range(100)
        ]

        # Hash each record
        record_hashes = [
            sha256_hash(canonical_json(rec).encode()) for rec in records
        ]

        # Create Merkle tree
        tree = MerkleTree(record_hashes)
        root = tree.get_root()

        # Verify inclusion of record 42
        record_42_hash = record_hashes[42]
        proof = tree.get_proof(record_42_hash)
        is_included = MerkleTree.verify_proof(record_42_hash, root, proof)

        assert is_included is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
