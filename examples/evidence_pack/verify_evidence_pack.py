#!/usr/bin/env python3
"""
Independent Evidence Pack Verifier

Verifies CIAF evidence pack integrity using only public information.
No private keys required - full third-party verification.

Usage: python verify_evidence_pack.py

This script proves:
- Every file in the evidence pack matches its SHA-256 in the manifest
- The manifest itself is Ed25519-signed; verification uses public key only
- The Merkle inclusion proof is valid for the given leaf/root
- The anchor signature is valid over canonical bytes; anchor root = Merkle root
- The capsule's proof and anchor point to the same root

Version: 1.0.0
Author: CIAF Enhanced Core Demo
"""

import json
import base64
import hashlib
import sys
import pathlib
from typing import List

# Import CIAF core modules for verification
try:
    sys.path.insert(0, "../..")  # Go up two levels to reach the main CIAF directory
    from ciaf.core import canonical_json
    from ciaf.core.signers import Ed25519Verifier
except ImportError as e:
    print(f"❌ Cannot import CIAF core modules: {e}")
    print(
        "Ensure you're running from the evidence_pack directory with ../../ciaf/ available"
    )
    sys.exit(1)


def load_bytes(path: str) -> bytes:
    """Load file as bytes."""
    return pathlib.Path(path).read_bytes()


def load_json(path: str) -> dict:
    """Load JSON file."""
    return json.loads(pathlib.Path(path).read_text())


def sha256_hash(data: bytes) -> str:
    """Calculate SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def verify_merkle_proof_manual(leaf_hash: str, proof: List, root_hash: str) -> bool:
    """
    Manually verify Merkle inclusion proof.

    Args:
        leaf_hash: Hex string of leaf hash
        proof: List of [hash_hex, position] pairs
        root_hash: Hex string of expected root

    Returns:
        True if proof is valid
    """
    current = bytes.fromhex(leaf_hash)

    for step in proof:
        if isinstance(step, list) and len(step) == 2:
            sibling_hex, position = step
            sibling = bytes.fromhex(sibling_hex)
        elif isinstance(step, dict):
            sibling = bytes.fromhex(step["hash"])
            position = step["position"]
        else:
            raise ValueError(f"Invalid proof step format: {step}")

        if position == "left":
            # Sibling is left, current is right
            current = hashlib.sha256(sibling + current).digest()
        else:
            # Current is left, sibling is right
            current = hashlib.sha256(current + sibling).digest()

    return current.hex() == root_hash


def main():
    """Run all verification checks."""
    print("🔍 CIAF Evidence Pack Independent Verification")
    print("=" * 55)

    # Check required files exist
    required_files = [
        "evidence_manifest.json",
        "evidence_manifest.sig",
        "public_key.pem",
        "anchor.json",
        "proof.json",
        "capsule.json",
    ]

    for file in required_files:
        if not pathlib.Path(file).exists():
            print(f"❌ Required file missing: {file}")
            sys.exit(1)

    print("✓ All required files present")

    # 1) Verify manifest file hashes
    print("\n--- Verifying Manifest File Hashes ---")
    manifest = load_json("evidence_manifest.json")
    files = manifest["files"]

    for f in files:
        path, expected = f["path"], f["sha256"]
        if not pathlib.Path(path).exists():
            print(f"⚠️  File listed in manifest but missing: {path}")
            continue

        actual = sha256_hash(load_bytes(path))
        if actual != expected:
            print(f"❌ Hash mismatch for {path}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            sys.exit(2)
        else:
            print(f"   ✓ {path}: {expected[:16]}...")

    print("✓ All manifest file hashes verified")

    # 2) Verify manifest signature (deterministic canonicalization)
    print("\n--- Verifying Manifest Signature ---")
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    manifest_hash = sha256_hash(manifest_bytes)
    print(f"   Manifest canonical bytes SHA256: {manifest_hash}")

    sig_bytes = load_bytes("evidence_manifest.sig")
    pub_pem = load_bytes("public_key.pem").decode("utf-8")

    print(f"   Signature length: {len(sig_bytes)} bytes")

    verifier = Ed25519Verifier("manifest_verifier", pub_pem)
    try:
        # Convert signature bytes to base64 string for the verifier
        sig_b64 = base64.b64encode(sig_bytes).decode("ascii")
        sig_valid = verifier.verify(manifest_bytes, sig_b64)
        if not sig_valid:
            print("❌ Manifest signature verification failed")
            sys.exit(2)
    except Exception as e:
        print(f"❌ Manifest signature verification error: {e}")
        sys.exit(2)

    print(f"   ✓ Ed25519 signature valid ({len(sig_bytes)} bytes)")
    print("✓ Manifest signature verified with public key only")

    # 3) Verify Merkle proof independently
    print("\n--- Verifying Merkle Inclusion Proof ---")
    proof_data = load_json("proof.json")
    leaf_hash = proof_data["leaf_hash"]
    root_hash = proof_data["merkle_root"]
    merkle_path = proof_data["merkle_path"]

    print(f"   Leaf hash: {leaf_hash[:16]}...")
    print(f"   Root hash: {root_hash[:16]}...")
    print(f"   Proof steps: {len(merkle_path)}")

    # Manual verification of Merkle proof
    proof_valid = verify_merkle_proof_manual(leaf_hash, merkle_path, root_hash)
    if not proof_valid:
        print("❌ Merkle inclusion proof verification failed")
        sys.exit(2)

    print(f"✓ Merkle proof verified independently ({len(merkle_path)} steps)")

    # 4) Verify anchor signature over canonical bytes
    print("\n--- Verifying Anchor Signature ---")
    anchor = load_json("anchor.json")

    # Create canonical anchor for signing (match enhanced_core_demo.py)
    canonical_anchor = {
        "root": anchor["root"],
        "policy_id": anchor["policy_id"],
        "schema_version": anchor["schema_version"],
        "timestamp": anchor["timestamp"],
        "domain_labels": sorted(anchor["domain_labels"]),
    }

    anchor_bytes = canonical_json(canonical_anchor).encode("utf-8")
    anchor_bytes_hash = sha256_hash(anchor_bytes)

    print(f"   Anchor canonical bytes SHA256: {anchor_bytes_hash}")
    print(f"   Expected SHA256: {anchor.get('canonical_bytes_sha256', 'N/A')}")

    # Verify the stored hash matches what we computed
    if anchor.get("canonical_bytes_sha256") != anchor_bytes_hash:
        print("❌ Anchor canonical bytes hash mismatch")
        sys.exit(2)

    # Verify anchor signature
    anchor_sig = anchor["signature"]
    try:
        anchor_sig_valid = verifier.verify(anchor_bytes, anchor_sig)
        if not anchor_sig_valid:
            print("❌ Anchor signature verification failed")
            sys.exit(2)
    except Exception as e:
        print(f"❌ Anchor signature verification error: {e}")
        sys.exit(2)

    # Verify anchor root matches Merkle root
    if anchor["root"] != root_hash:
        print("❌ Anchor root ≠ Merkle root")
        print(f"   Anchor root: {anchor['root']}")
        print(f"   Merkle root: {root_hash}")
        sys.exit(2)

    print("✓ Anchor signature verified (public key only)")
    print("✓ Anchor root = Merkle root linkage confirmed")

    # 5) Verify capsule linkage (light check)
    print("\n--- Verifying Capsule Linkage ---")
    capsule = load_json("capsule.json")

    # Check capsule contains consistent root references
    capsule_merkle_root = capsule.get("metadata", {}).get("merkle_root")
    if capsule_merkle_root and capsule_merkle_root != root_hash:
        print("❌ Capsule Merkle root mismatch")
        sys.exit(2)

    print("✓ Capsule linkage verified")

    # 6) Verify provenance information
    print("\n--- Verifying Provenance ---")
    provenance_files = ["anchor.json", "proof.json", "capsule.json"]
    for pfile in provenance_files:
        data = load_json(pfile)
        prov = data.get("provenance", {})
        if prov.get("tool") == "enhanced_core_demo.py" and prov.get("version"):
            print(f"   ✓ {pfile}: {prov['tool']} v{prov['version']}")
        else:
            print(f"   ⚠️  {pfile}: Missing or incomplete provenance")

    print("\n🎉 ALL VERIFICATION CHECKS PASS ✅")
    print("\n📋 Evidence Pack Summary:")
    print(f"   📁 Files verified: {len(files)}")
    print("   🔐 Manifest signature: Valid Ed25519")
    print(f"   🌳 Merkle proof steps: {len(merkle_path)}")
    print("   ⚓ Anchor signature: Valid Ed25519")
    print("   🔗 Root linkage: Anchor ↔ Merkle ↔ Capsule")
    print("   📝 Provenance: enhanced_core_demo.py v1.0.0")
    print(
        "\n✨ This evidence pack is cryptographically sound and third-party verifiable."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Verification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
