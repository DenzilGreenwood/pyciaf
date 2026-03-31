#!/usr/bin/env python3
"""
CIAF Watermarks - Signature Envelope Tests

Tests for the signature envelope pattern implementation, validating:
- SignatureEnvelope and SignatureMetadata dataclasses
- Serialization/deserialization (to_dict/from_dict)
- Integration with ArtifactEvidence model
- Backward compatibility handling

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks.signature_envelope import (
    KeyBackend,
    SignatureEncoding,
    SignatureMetadata,
    SignatureEnvelope,
    create_signature_envelope,
)
from ciaf.watermarks.models import (
    ArtifactEvidence,
    ArtifactType,
    WatermarkDescriptor,
    WatermarkType,
    ArtifactHashSet,
)
import json


def test_signature_metadata_creation():
    """Test SignatureMetadata dataclass creation and serialization."""
    print("\n[TEST] SignatureMetadata Creation")
    print("=" * 60)

    metadata = SignatureMetadata(
        signature_algorithm="Ed25519",
        key_id="aws-kms:alias/ciaf-prod",
        canonicalization_version="RFC8785-like/1.0",
        key_backend=KeyBackend.KMS,
        signing_service="ciaf-vault-signer",
        public_key_ref="jwks://cognitiveinsight.ai/keys/abc123",
        verification_method="ciaf-verify/v1",
    )

    assert metadata.signature_algorithm == "Ed25519"
    assert metadata.key_id == "aws-kms:alias/ciaf-prod"
    assert metadata.key_backend == KeyBackend.KMS
    assert metadata.signing_service == "ciaf-vault-signer"

    print("  ✅ SignatureMetadata created successfully")

    # Test serialization
    metadata_dict = metadata.to_dict()
    assert metadata_dict["signature_algorithm"] == "Ed25519"
    assert metadata_dict["key_backend"] == "kms"  # Enum converted to string

    print("  ✅ to_dict() serialization successful")

    # Test deserialization
    metadata_restored = SignatureMetadata.from_dict(metadata_dict)
    assert (
        metadata_restored.key_backend == KeyBackend.KMS
    )  # String converted back to enum
    assert metadata_restored.key_id == "aws-kms:alias/ciaf-prod"

    print("  ✅ from_dict() deserialization successful")
    print("[OK] SignatureMetadata test passed\n")


def test_signature_envelope_creation():
    """Test SignatureEnvelope dataclass creation and serialization."""
    print("\n[TEST] SignatureEnvelope Creation")
    print("=" * 60)

    metadata = SignatureMetadata(
        signature_algorithm="Ed25519",
        key_id="ciaf-test-key",
        canonicalization_version="RFC8785-like/1.0",
        key_backend=KeyBackend.LOCAL,
    )

    envelope = SignatureEnvelope(
        payload_hash="abc123def456" + "0" * 52,  # 64 hex chars
        hash_algorithm="SHA-256",
        signature_value="SGVsbG8gV29ybGQh",  # Base64 encoded
        signature_encoding=SignatureEncoding.BASE64,
        signed_at="2026-03-30T18:00:00Z",
        metadata=metadata,
    )

    assert envelope.payload_hash == "abc123def456" + "0" * 52
    assert envelope.hash_algorithm == "SHA-256"
    assert envelope.signature_encoding == SignatureEncoding.BASE64

    print("  ✅ SignatureEnvelope created successfully")

    # Test serialization
    envelope_dict = envelope.to_dict()
    assert envelope_dict["payload_hash"] == "abc123def456" + "0" * 52
    assert envelope_dict["signature_encoding"] == "base64"  # Enum to string
    assert envelope_dict["metadata"]["key_backend"] == "local"

    print("  ✅ to_dict() serialization successful")

    # Test JSON serialization
    json_str = json.dumps(envelope_dict, indent=2)
    assert "payload_hash" in json_str
    assert "metadata" in json_str

    print("  ✅ JSON serialization successful")

    # Test deserialization
    envelope_restored = SignatureEnvelope.from_dict(envelope_dict)
    assert envelope_restored.signature_encoding == SignatureEncoding.BASE64
    assert envelope_restored.metadata.key_backend == KeyBackend.LOCAL

    print("  ✅ from_dict() deserialization successful")
    print("[OK] SignatureEnvelope test passed\n")


def test_create_signature_envelope_factory():
    """Test the factory function for creating signature envelopes."""
    print("\n[TEST] create_signature_envelope_factory()")
    print("=" * 60)

    envelope = create_signature_envelope(
        payload_hash="fedcba987654" + "0" * 52,
        signature_value="U2lnbmVkIURhdGEh",
        key_id="ciaf-watermark-key",
        key_backend=KeyBackend.KMS,
        signing_service="ciaf-vault",
    )

    assert envelope.payload_hash == "fedcba987654" + "0" * 52
    assert envelope.signature_value == "U2lnbmVkIURhdGEh"
    assert envelope.hash_algorithm == "SHA-256"
    assert envelope.signature_encoding == SignatureEncoding.BASE64
    assert envelope.metadata.signature_algorithm == "Ed25519"
    assert envelope.metadata.key_id == "ciaf-watermark-key"
    assert envelope.metadata.key_backend == KeyBackend.KMS
    assert envelope.metadata.signing_service == "ciaf-vault"
    assert envelope.metadata.canonicalization_version == "RFC8785-like/1.0"

    print("  ✅ Factory function creates complete envelope")
    print(f"  ✅ Generated signed_at: {envelope.signed_at}")
    print("[OK] Factory function test passed\n")


def test_unsigned_placeholder():
    """Test creating an unsigned placeholder envelope."""
    print("\n[TEST] Unsigned Placeholder Envelope")
    print("=" * 60)

    placeholder = SignatureEnvelope.create_unsigned_placeholder()

    assert placeholder.payload_hash == "0" * 64
    assert placeholder.signature_value == ""
    assert placeholder.metadata.key_id == "unsigned"
    assert placeholder.metadata.key_backend == KeyBackend.LOCAL

    print("  ✅ Unsigned placeholder created")
    print(f"  ✅ Timestamp: {placeholder.signed_at}")
    print("[OK] Unsigned placeholder test passed\n")


def test_artifact_evidence_with_signature_envelope():
    """Test ArtifactEvidence integration with SignatureEnvelope."""
    print("\n[TEST] ArtifactEvidence + SignatureEnvelope Integration")
    print("=" * 60)

    # Create artifact evidence
    evidence = ArtifactEvidence(
        artifact_id="artifact-12345",
        artifact_type=ArtifactType.TEXT,
        mime_type="text/plain",
        created_at="2026-03-30T18:00:00Z",
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-17",
        prompt_hash="a" * 64,
        output_hash_raw="b" * 64,
        output_hash_distributed="c" * 64,
        watermark=WatermarkDescriptor(
            watermark_id="wmk-123",
            watermark_type=WatermarkType.VISIBLE,
            verification_url="https://vault.example.com/verify/wmk-123",
            tag_text="CIAF Watermark",
        ),
        hashes=ArtifactHashSet(
            content_hash_before_watermark="b" * 64,
            content_hash_after_watermark="c" * 64,
        ),
    )

    print("  ✅ ArtifactEvidence created")

    # Add signature envelope
    envelope = create_signature_envelope(
        payload_hash=evidence.compute_receipt_hash(),
        signature_value="U2lnbmVkIEFydGlmYWN0",
        key_id="ciaf-artifact-key",
        key_backend=KeyBackend.KMS,
    )

    evidence.signature = envelope

    assert evidence.signature is not None
    assert isinstance(evidence.signature, SignatureEnvelope)
    assert evidence.signature.payload_hash == evidence.compute_receipt_hash()

    print("  ✅ Signature envelope attached to evidence")

    # Test to_dict() includes signature
    evidence_dict = evidence.to_dict()
    assert "signature" in evidence_dict
    assert evidence_dict["signature"]["payload_hash"] == evidence.compute_receipt_hash()
    assert evidence_dict["signature"]["metadata"]["key_backend"] == "kms"

    print("  ✅ to_dict() includes signature envelope")

    # Test canonical_dict excludes signature
    canonical_dict = evidence.to_canonical_dict()
    assert "signature" not in canonical_dict
    print("  ✅ to_canonical_dict() correctly excludes signature")

    # Test JSON serialization of full evidence
    json_str = json.dumps(evidence_dict, indent=2)
    assert "signature" in json_str
    assert "payload_hash" in json_str
    assert "metadata" in json_str

    print("  ✅ Full evidence JSON serialization successful")
    print("[OK] ArtifactEvidence integration test passed\n")


def test_enum_serialization_consistency():
    """Test that enums are consistently serialized as strings."""
    print("\n[TEST] Enum Serialization Consistency")
    print("=" * 60)

    # Test KeyBackend enum
    backends = [
        KeyBackend.LOCAL,
        KeyBackend.KMS,
        KeyBackend.HSM,
        KeyBackend.CLOUDHSM,
        KeyBackend.EXTERNAL_KMS,
    ]

    for backend in backends:
        metadata = SignatureMetadata(
            signature_algorithm="Ed25519",
            key_id="test-key",
            canonicalization_version="RFC8785-like/1.0",
            key_backend=backend,
        )

        # Serialize
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict["key_backend"], str)

        # Deserialize
        metadata_restored = SignatureMetadata.from_dict(metadata_dict)
        assert metadata_restored.key_backend == backend

        print(
            f"  ✅ {backend.value:15} -> '{metadata_dict['key_backend']}' -> {metadata_restored.key_backend.value}"
        )

    # Test SignatureEncoding enum
    encodings = [
        SignatureEncoding.BASE64,
        SignatureEncoding.BASE64URL,
        SignatureEncoding.HEX,
    ]

    for encoding in encodings:
        envelope = SignatureEnvelope(
            payload_hash="0" * 64,
            hash_algorithm="SHA-256",
            signature_value="test",
            signature_encoding=encoding,
            signed_at="2026-03-30T18:00:00Z",
            metadata=SignatureMetadata(
                signature_algorithm="Ed25519",
                key_id="test",
                canonicalization_version="RFC8785-like/1.0",
                key_backend=KeyBackend.LOCAL,
            ),
        )

        # Serialize
        envelope_dict = envelope.to_dict()
        assert isinstance(envelope_dict["signature_encoding"], str)

        # Deserialize
        envelope_restored = SignatureEnvelope.from_dict(envelope_dict)
        assert envelope_restored.signature_encoding == encoding

        print(
            f"  ✅ {encoding.value:10} -> '{envelope_dict['signature_encoding']}' -> {envelope_restored.signature_encoding.value}"
        )

    print("[OK] Enum serialization test passed\n")


def run_all_tests():
    """Run all signature envelope tests."""
    print("\n" + "=" * 60)
    print("CIAF WATERMARKS - SIGNATURE ENVELOPE TEST SUITE")
    print("Testing SignatureEnvelope pattern implementation")
    print("=" * 60)

    try:
        # Test 1: SignatureMetadata
        test_signature_metadata_creation()

        # Test 2: SignatureEnvelope
        test_signature_envelope_creation()

        # Test 3: Factory function
        test_create_signature_envelope_factory()

        # Test 4: Unsigned placeholder
        test_unsigned_placeholder()

        # Test 5: Integration with ArtifactEvidence
        test_artifact_evidence_with_signature_envelope()

        # Test 6: Enum serialization
        test_enum_serialization_consistency()

        # Summary
        print("\n" + "=" * 60)
        print("✅ ALL SIGNATURE ENVELOPE TESTS PASSED")
        print("=" * 60)
        print("\nSignatureEnvelope Pattern Verified:")
        print("  ✅ SignatureMetadata dataclass working")
        print("  ✅ SignatureEnvelope dataclass working")
        print("  ✅ Serialization/deserialization (to_dict/from_dict)")
        print("  ✅ JSON serialization successful")
        print("  ✅ Factory function working")
        print("  ✅ Unsigned placeholder creation")
        print("  ✅ Integration with ArtifactEvidence")
        print("  ✅ Enum handling (KeyBackend, SignatureEncoding)")
        print("  ✅ Canonical dict excludes signature (correct)")
        print("\n✅ Week 3 Task 1 COMPLETE: Signature Standardization")
        print("\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
