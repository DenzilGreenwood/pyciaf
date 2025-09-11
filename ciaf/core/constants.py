"""
CIAF Core Constants

Centralized constants for anchor schema and merkle policy versions
to keep all scripts in lockstep.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

# Anchor and Policy Versions
ANCHOR_SCHEMA_VERSION = "1.0"
MERKLE_POLICY_VERSION = "1.0"

# Hash Functions
DEFAULT_HASH_FUNCTION = "sha256"
HASH_OUTPUT_LENGTH = 64  # SHA-256 produces 64 hex chars

# Event ID Format
EVENT_ID_PREFIX = "evt"

# RNG Sources (extensible list)
SUPPORTED_RNG_SOURCES = ["numpy", "torch", "tensorflow", "jax", "random"]

# Capsule Signature
DEFAULT_SIGNATURE_ALGORITHM = "ed25519"
DEFAULT_PUBKEY_ID = "ciaf_demo_key_001"
