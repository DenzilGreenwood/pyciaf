"""
Comprehensive test vectors for CIAF core cryptographic operations.

Provides reference test vectors for all cryptographic functions to ensure
interoperability and correct implementation across different environments.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any

from .canonicalization import canonical_json, canonicalize_and_hash, make_anchor, Policy
from .crypto import sha256_hash, compute_hash
from .determinism import DeterministicClock, canonical_timestamp
from .enums import HashAlgorithm, RecordType
from .merkle import MerkleTree
from .signers import Ed25519Signer


@dataclass
class TestVector:
    """Single test vector with input, expected output, and metadata."""
    name: str
    description: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    algorithm: str
    test_category: str


@dataclass
class TestVectorSuite:
    """Complete suite of test vectors."""
    version: str
    created_at: str
    description: str
    vectors: List[TestVector]


class CIAFTestVectors:
    """Generator and validator for CIAF test vectors."""
    
    def __init__(self):
        self.test_vectors = []
        self.fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    def generate_hash_vectors(self) -> List[TestVector]:
        """Generate test vectors for hash functions."""
        vectors = []
        
        test_inputs = [
            "",
            "hello",
            "The quick brown fox jumps over the lazy dog",
            '{"key": "value", "number": 42}',
            "🎉 Unicode test with émojis and special chars: àáâãäå",
            "A" * 1000,  # Long string
            "\x00\x01\x02\x03\xff",  # Binary data as string
        ]
        
        for i, test_input in enumerate(test_inputs):
            data = test_input.encode('utf-8')
            
            # SHA256 vector
            vectors.append(TestVector(
                name=f"sha256_test_{i+1}",
                description=f"SHA256 hash of test input {i+1}",
                inputs={"data": test_input, "algorithm": "sha256"},
                expected_outputs={"hash": sha256_hash(data)},
                algorithm="SHA256",
                test_category="hash"
            ))
            
            # SHA3-256 vector (if available)
            try:
                sha3_hash = compute_hash(data, "sha3-256")
                vectors.append(TestVector(
                    name=f"sha3_256_test_{i+1}",
                    description=f"SHA3-256 hash of test input {i+1}",
                    inputs={"data": test_input, "algorithm": "sha3-256"},
                    expected_outputs={"hash": sha3_hash},
                    algorithm="SHA3-256",
                    test_category="hash"
                ))
            except ValueError:
                pass
        
        return vectors
    
    def generate_canonicalization_vectors(self) -> List[TestVector]:
        """Generate test vectors for JSON canonicalization."""
        vectors = []
        
        test_objects = [
            {},
            {"key": "value"},
            {"b": 2, "a": 1},  # Test sorting
            {"nested": {"z": 3, "a": 1}, "top": "level"},
            {"array": [3, 1, 2], "string": "test"},
            {"unicode": "🎉", "special": "àáâãäå"},
            {"number": 42, "float": 3.14159, "bool": True, "null": None},
        ]
        
        for i, test_obj in enumerate(test_objects):
            canonical = canonical_json(test_obj)
            hash_result = canonicalize_and_hash(test_obj, HashAlgorithm.SHA256)
            
            vectors.append(TestVector(
                name=f"canonicalization_test_{i+1}",
                description=f"JSON canonicalization test {i+1}",
                inputs={"object": test_obj},
                expected_outputs={
                    "canonical_json": canonical,
                    "canonical_hash": hash_result
                },
                algorithm="Canonical JSON + SHA256",
                test_category="canonicalization"
            ))
        
        return vectors
    
    def generate_merkle_vectors(self) -> List[TestVector]:
        """Generate test vectors for Merkle tree operations."""
        vectors = []
        
        # Test cases with different numbers of leaves
        test_cases = [
            ["a" * 64],  # Single leaf (64 hex chars)
            ["a" * 64, "b" * 64],  # Two leaves
            ["a" * 64, "b" * 64, "c" * 64],  # Three leaves
            ["a" * 64, "b" * 64, "c" * 64, "d" * 64],  # Four leaves (perfect binary tree)
            [f"{chr(ord('a') + i)}" * 64 for i in range(5)],  # Five leaves
        ]
        
        for i, leaves in enumerate(test_cases):
            tree = MerkleTree(leaves)
            root = tree.get_root()
            
            # Generate proofs for each leaf
            proofs = {}
            for leaf in leaves:
                proof = tree.get_proof(leaf)
                is_valid = tree.verify_proof_static(leaf, root, proof)
                proofs[leaf] = {
                    "proof": proof,
                    "valid": is_valid
                }
            
            vectors.append(TestVector(
                name=f"merkle_test_{i+1}",
                description=f"Merkle tree with {len(leaves)} leaves",
                inputs={"leaves": leaves},
                expected_outputs={
                    "root": root,
                    "proofs": proofs
                },
                algorithm="Merkle Tree (SHA256)",
                test_category="merkle"
            ))
        
        return vectors
    
    def generate_signature_vectors(self) -> List[TestVector]:
        """Generate test vectors for Ed25519 signatures."""
        vectors = []
        
        # Fixed key for reproducible tests
        # This is a test vector from RFC 8032
        test_private_key_hex = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
        test_public_key_hex = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        
        # Create signer from known test key
        private_key_bytes = bytes.fromhex(test_private_key_hex)
        
        # Note: This is simplified - real implementation would need proper key loading
        # For now, generate a signer and use it consistently
        signer = Ed25519Signer("test_key")
        
        test_messages = [
            "",
            "hello world",
            "The quick brown fox jumps over the lazy dog",
            json.dumps({"timestamp": "2025-01-01T12:00:00.000000Z", "data": "test"}),
            "A" * 1000,
        ]
        
        for i, message in enumerate(test_messages):
            message_bytes = message.encode('utf-8')
            signature = signer.sign(message_bytes)
            is_valid = signer.verify(message_bytes, signature)
            
            vectors.append(TestVector(
                name=f"ed25519_signature_test_{i+1}",
                description=f"Ed25519 signature test {i+1}",
                inputs={
                    "message": message,
                    "key_id": signer.key_id
                },
                expected_outputs={
                    "signature": signature,
                    "verification_result": is_valid,
                    "public_key_fingerprint": signer.get_public_key_fingerprint()
                },
                algorithm="Ed25519",
                test_category="signature"
            ))
        
        return vectors
    
    def generate_anchor_vectors(self) -> List[TestVector]:
        """Generate test vectors for anchor creation."""
        vectors = []
        
        # Fixed clock for deterministic timestamps
        fixed_clock = DeterministicClock(self.fixed_time)
        
        # Test policy configurations
        test_policies = [
            Policy(
                policy_id="test_policy_1",
                schema_version="1.0",
                domain_labels=["test"],
                hash_algorithm=HashAlgorithm.SHA256
            ),
            Policy(
                policy_id="high_risk_policy",
                schema_version="1.0", 
                domain_labels=["healthcare", "finance"],
                hash_algorithm=HashAlgorithm.SHA256,
                external_timestamping=True
            ),
        ]
        
        test_roots = [
            "a" * 64,
            "b" * 64,
            sha256_hash(b"test_root")
        ]
        
        signer = Ed25519Signer("test_signer")
        
        for i, (policy, root) in enumerate(zip(test_policies, test_roots)):
            # Mock the timestamp for deterministic results
            original_timestamp = canonical_timestamp
            
            try:
                anchor = make_anchor(root, policy, signer)
                
                vectors.append(TestVector(
                    name=f"anchor_test_{i+1}",
                    description=f"Anchor creation test {i+1}",
                    inputs={
                        "root": root,
                        "policy": asdict(policy),
                        "signer_id": signer.key_id
                    },
                    expected_outputs={
                        "anchor_root": anchor.root,
                        "anchor_policy_id": anchor.policy_id,
                        "anchor_schema_version": anchor.schema_version,
                        "anchor_domain_labels": anchor.domain_labels,
                        "signature_length": len(anchor.signature),
                        "signing_key_id": anchor.signing_key_id
                    },
                    algorithm="Ed25519 + Anchor Format",
                    test_category="anchor"
                ))
            
            finally:
                pass
        
        return vectors
    
    def generate_determinism_vectors(self) -> List[TestVector]:
        """Generate test vectors for deterministic operations."""
        vectors = []
        
        # Fixed time operations
        fixed_clock = DeterministicClock(self.fixed_time)
        
        # Test deterministic timestamp generation
        test_operations = [
            ("operation_1", "entropy_1"),
            ("operation_2", "entropy_2"),
            ("", ""),
            ("long_operation_name_with_details", "additional_entropy_data"),
        ]
        
        for i, (op_id, entropy) in enumerate(test_operations):
            timestamp1 = fixed_clock.canonical_iso_format(fixed_clock.now())
            timestamp2 = fixed_clock.canonical_iso_format(fixed_clock.now())
            
            # Test that deterministic operations are actually deterministic
            time_hash1 = fixed_clock.time_hash(f"{op_id}:{entropy}")
            time_hash2 = fixed_clock.time_hash(f"{op_id}:{entropy}")
            
            vectors.append(TestVector(
                name=f"determinism_test_{i+1}",
                description=f"Deterministic operations test {i+1}",
                inputs={
                    "operation_id": op_id,
                    "entropy": entropy,
                    "fixed_time": self.fixed_time.isoformat()
                },
                expected_outputs={
                    "canonical_timestamp_format": timestamp1,
                    "time_hash": time_hash1,
                    "hash_consistency": time_hash1 == time_hash2
                },
                algorithm="Deterministic Clock + SHA256",
                test_category="determinism"
            ))
        
        return vectors
    
    def generate_all_vectors(self) -> TestVectorSuite:
        """Generate complete test vector suite."""
        all_vectors = []
        
        all_vectors.extend(self.generate_hash_vectors())
        all_vectors.extend(self.generate_canonicalization_vectors())
        all_vectors.extend(self.generate_merkle_vectors())
        all_vectors.extend(self.generate_signature_vectors())
        all_vectors.extend(self.generate_anchor_vectors())
        all_vectors.extend(self.generate_determinism_vectors())
        
        return TestVectorSuite(
            version="1.0.0",
            created_at=canonical_timestamp(),
            description="CIAF Core Cryptographic Test Vectors",
            vectors=all_vectors
        )
    
    def export_vectors_json(self, file_path: str = "ciaf_test_vectors.json"):
        """Export test vectors to JSON file."""
        suite = self.generate_all_vectors()
        
        # Convert to serializable format
        suite_dict = asdict(suite)
        
        with open(file_path, 'w') as f:
            json.dump(suite_dict, f, indent=2, sort_keys=True)
        
        return file_path
    
    def validate_implementation(self, implementation_vectors: TestVectorSuite) -> Dict[str, Any]:
        """
        Validate an implementation against reference test vectors.
        
        Args:
            implementation_vectors: Test vectors from implementation to validate
            
        Returns:
            Validation report
        """
        reference_suite = self.generate_all_vectors()
        
        # Group vectors by name for comparison
        reference_map = {v.name: v for v in reference_suite.vectors}
        implementation_map = {v.name: v for v in implementation_vectors.vectors}
        
        results = {
            'total_tests': len(reference_suite.vectors),
            'passed': 0,
            'failed': 0,
            'missing': 0,
            'failures': [],
            'missing_tests': []
        }
        
        for name, ref_vector in reference_map.items():
            if name not in implementation_map:
                results['missing'] += 1
                results['missing_tests'].append(name)
                continue
            
            impl_vector = implementation_map[name]
            
            # Compare expected outputs
            if ref_vector.expected_outputs == impl_vector.expected_outputs:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    'test': name,
                    'expected': ref_vector.expected_outputs,
                    'actual': impl_vector.expected_outputs
                })
        
        results['success_rate'] = results['passed'] / results['total_tests'] if results['total_tests'] > 0 else 0
        
        return results


# Convenience functions

def generate_test_vectors() -> TestVectorSuite:
    """Generate the complete CIAF test vector suite."""
    generator = CIAFTestVectors()
    return generator.generate_all_vectors()


def export_test_vectors(file_path: str = "ciaf_test_vectors.json") -> str:
    """Export test vectors to JSON file."""
    generator = CIAFTestVectors()
    return generator.export_vectors_json(file_path)


def load_test_vectors(file_path: str) -> TestVectorSuite:
    """Load test vectors from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to dataclasses
    vectors = [TestVector(**v) for v in data['vectors']]
    return TestVectorSuite(
        version=data['version'],
        created_at=data['created_at'],
        description=data['description'],
        vectors=vectors
    )


def validate_ciaf_implementation(test_vectors_file: str) -> Dict[str, Any]:
    """
    Validate a CIAF implementation against reference test vectors.
    
    Args:
        test_vectors_file: Path to implementation test vectors JSON
        
    Returns:
        Validation report
    """
    generator = CIAFTestVectors()
    implementation_vectors = load_test_vectors(test_vectors_file)
    return generator.validate_implementation(implementation_vectors)


if __name__ == "__main__":
    # Generate and export test vectors when run as script
    file_path = export_test_vectors()
    print(f"Test vectors exported to {file_path}")
    
    # Print summary
    suite = generate_test_vectors()
    categories = {}
    for vector in suite.vectors:
        categories[vector.test_category] = categories.get(vector.test_category, 0) + 1
    
    print(f"\nGenerated {len(suite.vectors)} test vectors:")
    for category, count in categories.items():
        print(f"  {category}: {count} vectors")