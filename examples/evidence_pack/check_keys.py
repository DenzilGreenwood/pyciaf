#!/usr/bin/env python3
"""
Check the public key fingerprint from the demo key.
"""

import sys
import json
sys.path.insert(0, '../..')

from ciaf.core.signers import Ed25519Signer

# Load the demo key 
with open('../demo_keys/demo_signing_key_2025.json', 'r') as f:
    key_data = json.load(f)

# Create signer
signer = Ed25519Signer.from_private_key_pem(
    key_id="demo_key",
    pem_data=key_data["private_key_pem"]
)

print(f"Demo key fingerprint: {signer.get_public_key_fingerprint()}")
print(f"Demo public key PEM:")
print(signer.get_public_key_pem())

# Also check what's in the current public_key.pem
with open('public_key.pem', 'r') as f:
    current_pem = f.read()

print(f"\nCurrent public_key.pem:")
print(current_pem)

# Check if they match
print(f"\nKeys match: {signer.get_public_key_pem().strip() == current_pem.strip()}")