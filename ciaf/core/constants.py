"""
CIAF Core Constants

Centralized constants for anchor schema and merkle policy versions
to keep all scripts in lockstep. Single source of truth for all
cryptographic parameters, schema versions, and configuration.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.1.0
"""

# SALT_LENGTH moved here from crypto.py for centralization
SALT_LENGTH = 16  # bytes

# Schema and Policy Versions
ANCHOR_SCHEMA_VERSION = "1.0"
MERKLE_POLICY_VERSION = "1.0"

# Crypto & KDF Parameters
PBKDF2_ITERATIONS = 100_000
KDF_DKLEN = 32

# Hash / Signature Algorithm Defaults
DEFAULT_HASH_FUNCTION = "sha256"
DEFAULT_SIGNATURE_ALGORITHM = "ed25519"  # Production default
DEFAULT_PUBKEY_ID = "ciaf_production_key_001"

# Hash Algorithm Properties
HASH_OUTPUT_LENGTH = 64  # SHA-256 produces 64 hex chars

# IDs / Misc
EVENT_ID_PREFIX = "evt"

# RNG Sources (extensible list)
SUPPORTED_RNG_SOURCES = ["numpy", "torch", "tensorflow", "jax", "random"]
