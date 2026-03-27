"""
Quick test to verify ProvenanceCapsule bug fix.

Tests that the derive_master_anchor function is correctly used
instead of the undefined derive_key function.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_provenance_capsule_creation():
    """Test creating and using a ProvenanceCapsule."""
    print("[TEST] ProvenanceCapsule creation and decryption... ", end="", flush=True)

    try:
        from ciaf.provenance import ProvenanceCapsule

        # Create a capsule
        test_data = "Sensitive patient data: John Doe, Age 45"
        metadata = {
            "source": "Hospital A",
            "consent_status": "granted",
            "data_type": "PHI",
        }
        secret = "my-secret-key-12345"

        capsule = ProvenanceCapsule(test_data, metadata, secret)

        # Verify it was created
        assert capsule.hash_proof is not None
        assert capsule.encrypted_data is not None
        assert capsule.salt is not None
        assert capsule.derived_key is not None
        assert len(capsule.derived_key) == 32  # Should be 32 bytes from PBKDF2

        # Test decryption
        decrypted = capsule.decrypt_data()
        assert decrypted == test_data

        # Test serialization
        json_data = capsule.to_json()
        assert "metadata" in json_data
        assert "encrypted_data" in json_data
        assert "salt" in json_data

        # Test deserialization
        capsule2 = ProvenanceCapsule.from_json(json_data, secret)
        decrypted2 = capsule2.decrypt_data()
        assert decrypted2 == test_data

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_provenance_capsule_with_numbers():
    """Test ProvenanceCapsule with numerical data."""
    print("[TEST] ProvenanceCapsule with numerical data... ", end="", flush=True)

    try:
        from ciaf.provenance import ProvenanceCapsule

        # Test with numeric data
        test_data = 42.5
        metadata = {"source": "sensor-123", "unit": "celsius"}
        secret = "temperature-secret"

        capsule = ProvenanceCapsule(test_data, metadata, secret)

        # Decrypt and verify
        decrypted = capsule.decrypt_data()
        assert decrypted == "42.5"  # Should be string representation

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("ProvenanceCapsule Bug Fix Verification")
    print("=" * 60)
    print()

    results = []
    results.append(test_provenance_capsule_creation())
    results.append(test_provenance_capsule_with_numbers())

    print()
    print("=" * 60)
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"Results: {passed} passed, {failed} failed (of {len(results)})")
    print("=" * 60)

    if failed == 0:
        print("\n[OK] All tests passed! Bug fix verified.")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
