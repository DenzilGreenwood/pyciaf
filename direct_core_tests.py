#!/usr/bin/env python3
"""
Direct core module tests without package imports.
Tests core functionality by importing modules directly.
"""

import sys
import os

# Add PYPI to path so we can import ciaf modules
sys.path.insert(0, os.path.dirname(__file__))

# Add ciaf to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ciaf'))

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ciaf', 'core'))

print("Testing core modules directly...")


def test_crypto_functions():
    """Test crypto functions."""
    print("\n🔍 Testing crypto functions...")
    
    try:
        # Import directly from the core crypto module
        from crypto import sha256_hash, blake3_hash, derive_master_anchor
        
        # Test SHA256 with bytes
        test_bytes = b"test data"
        sha256_result = sha256_hash(test_bytes)
        assert isinstance(sha256_result, str), "SHA256 should return string"
        print(f"✓ SHA256 hash (bytes): {sha256_result[:20]}...")
        
        # Test SHA256 with string (should auto-encode)
        test_string = "test data"
        sha256_result_str = sha256_hash(test_string)
        assert isinstance(sha256_result_str, str), "SHA256 should return string"
        print(f"✓ SHA256 hash (string): {sha256_result_str[:20]}...")
        
        # Test BLAKE3 with bytes
        blake3_result = blake3_hash(test_bytes)
        assert isinstance(blake3_result, str), "BLAKE3 should return string"  
        print(f"✓ BLAKE3 hash: {blake3_result[:20]}...")
        
        # Test derivation function
        master_result = derive_master_anchor("password", b"salt12345678")
        assert isinstance(master_result, bytes), "derive_master_anchor should return bytes"
        print("✓ Master anchor derivation works")
        
        return True
    except Exception as e:
        print(f"❌ Crypto functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ed25519_signer():
    """Test Ed25519 signer functionality."""
    print("\n🔍 Testing Ed25519 signer...")
    
    try:
        from signers import Ed25519Signer
        
        # Create signer
        signer = Ed25519Signer("test_key")
        
        # Test signing and verification
        test_data = b"test message for signing"
        signature = signer.sign(test_data)
        
        print(f"✓ Signature created: {signature[:20]}...")
        
        # Verify signature
        is_valid = signer.verify(test_data, signature)
        assert is_valid, "Signature verification should succeed"
        print("✓ Signature verification successful")
        
        # Test PEM export/import
        public_pem = signer.export_public_key_pem()
        print(f"✓ Public key exported: {len(public_pem)} bytes")
        
        private_pem = signer.export_private_key_pem()
        print(f"✓ Private key exported: {len(private_pem)} bytes")
        
        # Test key loading
        loaded_signer = Ed25519Signer.from_private_key_pem(private_pem)
        loaded_signature = loaded_signer.sign(test_data)
        is_loaded_valid = loaded_signer.verify(test_data, loaded_signature)
        assert is_loaded_valid, "Loaded signer should work"
        print("✓ Key loading and reuse works")
        
        return True
    except Exception as e:
        print(f"❌ Ed25519 signer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constants_and_enums():
    """Test constants and enums."""
    print("\n🔍 Testing constants and enums...")
    
    try:
        from constants import SALT_LENGTH, DEFAULT_SIGNATURE_ALGORITHM, PBKDF2_ITERATIONS
        from enums import HashAlgorithm, SignatureAlgorithm, RecordType
        
        # Test constants exist and are reasonable
        assert SALT_LENGTH > 0, "SALT_LENGTH should be positive"
        assert DEFAULT_SIGNATURE_ALGORITHM == "ed25519", "Default should be ed25519"
        assert PBKDF2_ITERATIONS >= 10000, "PBKDF2 iterations should be strong"
        print("✓ Constants are defined and reasonable")
        
        # Test enums
        assert HashAlgorithm.SHA256.value == "sha256"
        assert HashAlgorithm.BLAKE3.value == "blake3" 
        assert SignatureAlgorithm.ED25519.value == "ed25519"
        print("✓ Enums are properly defined")
        
        return True
    except Exception as e:
        print(f"❌ Constants and enums test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_merkle_basic():
    """Test basic Merkle tree functionality."""
    print("\n🔍 Testing basic Merkle tree...")
    
    try:
        from merkle import MerkleTree
        from enums import HashAlgorithm
        
        # Create basic Merkle tree
        tree = MerkleTree(HashAlgorithm.SHA256)
        
        # Add some leaves
        leaves = [
            "leaf1_hash",
            "leaf2_hash", 
            "leaf3_hash"
        ]
        
        for leaf in leaves:
            tree.add_leaf(leaf)
        
        # Get root
        root = tree.get_root()
        assert root is not None, "Root should not be None"
        print(f"✓ Merkle root computed: {root[:20]}...")
        
        # Test proof generation and verification
        proof = tree.get_proof(0)  # Get proof for first leaf
        is_valid = tree.verify_proof("leaf1_hash", proof, root)
        assert is_valid, "Proof should be valid"
        print("✓ Merkle proof generation and verification works")
        
        return True
    except Exception as e:
        print(f"❌ Basic Merkle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run direct core module tests."""
    print("=" * 60)
    print("CIAF Core Direct Module Tests") 
    print("=" * 60)
    
    tests = [
        ("Crypto Functions", test_crypto_functions),
        ("Ed25519 Signer", test_ed25519_signer),
        ("Constants and Enums", test_constants_and_enums),
        ("Basic Merkle Tree", test_merkle_basic)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All core module tests passed!")
        print("\nCore functionality confirmed:")
        print("  ✅ Crypto functions work with proper encoding") 
        print("  ✅ Ed25519 signing and verification functional")
        print("  ✅ Key export/import working")
        print("  ✅ Constants and enums properly defined")
        print("  ✅ Basic Merkle tree operations work")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed.")
        
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)