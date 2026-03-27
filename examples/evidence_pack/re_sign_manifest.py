#!/usr/bin/env python3
"""
Re-sign the evidence manifest after fixing the file hashes.
"""

import sys
import json

sys.path.insert(0, "../..")

from ciaf.core import canonical_json
from ciaf.core.signers import Ed25519Signer
import base64

# Load the production key from the examples directory
with open("../demo_keys/demo_signing_key_2025.json", "r") as f:
    key_data = json.load(f)

# Create signer
signer = Ed25519Signer.from_private_key_pem(
    key_id="production_anchor_key", pem_data=key_data["private_key_pem"]
)

# Load and sign the manifest
with open("evidence_manifest.json", "r") as f:
    manifest = json.load(f)

# Create canonical bytes and sign
canonical_bytes = canonical_json(manifest).encode()
signature_b64 = signer.sign(canonical_bytes)
signature_bytes = base64.b64decode(signature_b64)

# Write signature to file
with open("evidence_manifest.sig", "wb") as f:
    f.write(signature_bytes)

print("✓ Manifest re-signed")
print(f"  Canonical bytes length: {len(canonical_bytes)}")
print(f"  Signature length: {len(signature_bytes)} bytes")
print(f"  Signature (b64): {signature_b64}")
