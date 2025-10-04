"""
Core cryptographic and foundational components for CIAF.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.2.1
"""

from .constants import (
    ANCHOR_SCHEMA_VERSION,
    MERKLE_POLICY_VERSION,
    PBKDF2_ITERATIONS,
    KDF_DKLEN,
    SALT_LENGTH,  # single source of truth
    DEFAULT_HASH_FUNCTION,
    DEFAULT_SIGNATURE_ALGORITHM,
    DEFAULT_PUBKEY_ID,
    HASH_OUTPUT_LENGTH,
    EVENT_ID_PREFIX,
    SUPPORTED_RNG_SOURCES
)

from .enums import RecordType, HashAlgorithm, SignatureAlgorithm
from .interfaces import Signer, RNG, Merkle, AnchorDeriver, AnchorStore

from .crypto import (
    CryptoUtils,
    decrypt_aes_gcm,
    encrypt_aes_gcm,
    hmac_sha256,
    secure_random_bytes,
    sha256_hash,
    blake3_hash,
    sha3_256_hash,
    compute_hash,
    derive_anchor_from_master,
    derive_master_anchor,
    derive_dataset_anchor,
    derive_model_anchor,
    derive_capsule_anchor,
    to_hex,
    from_hex,
    make_aad,
    generate_master_password,
)

from .signers import Ed25519Signer, Ed25519Verifier, ProductionSigner
from .canonicalization import (
    create_production_signer,
    Policy,
    AnchorRecord,
    Receipt,
    MockSigner,
    WORMMerkleTree,
    CapsuleBuilder,
    canonical_json,
    canonicalize_and_hash,
    validate_required_fields,
    enrich_metadata_with_defaults,
    make_anchor,
    REQUIRED_FIELDS
)
from .merkle import MerkleTree

# New enhanced modules
from .policy_enforcement import (
    RiskLevel,
    ComplianceResult,
    PolicyViolation,
    RiskAssessment,
    PolicyRule,
    PolicyEnforcer,
    HighRiskDomainRule,
    PiiDetectionRule,
    TimestampValidationRule,
    RequiredFieldsRule,
    create_healthcare_policy_enforcer,
    create_financial_policy_enforcer,
    create_gdpr_policy_enforcer
)

from .determinism import (
    DeterministicClock,
    LocaleIndependentOps,
    DeterministicTimestampGenerator,
    default_clock,
    default_timestamp_generator,
    now_iso,
    canonical_timestamp,
    deterministic_timestamp,
    timestamped_id,
    normalize_for_determinism,
    compare_deterministic,
    sort_deterministic,
    FixedTimeContext,
    DeterministicContext
)

from .key_management import (
    KeyStatus,
    KeyType,
    KeyMetadata,
    KeyBundle,
    KeyStore,
    FileSystemKeyStore,
    KeyManager,
    create_filesystem_key_manager,
    create_default_ciaf_key_manager,
    generate_ciaf_signing_key,
    get_ciaf_signer
)

from .worm_store import (
    WORMRecord,
    WORMStore,
    SQLiteWORMStore,
    LMDBWORMStore,
    DurableWORMMerkleTree,
    create_sqlite_worm_store,
    create_lmdb_worm_store
)

from .test_vectors import (
    TestVector,
    TestVectorSuite,
    CIAFTestVectors,
    generate_test_vectors,
    export_test_vectors,
    load_test_vectors,
    validate_ciaf_implementation
)

# Legacy anchor managers removed - using LCM system instead
BaseAnchorManager = None
AnchorManager = None

__all__ = [
    # Constants
    "ANCHOR_SCHEMA_VERSION",
    "MERKLE_POLICY_VERSION",
    "PBKDF2_ITERATIONS",
    "KDF_DKLEN",
    "SALT_LENGTH",
    "DEFAULT_HASH_FUNCTION",
    "DEFAULT_SIGNATURE_ALGORITHM",
    "DEFAULT_PUBKEY_ID",
    "HASH_OUTPUT_LENGTH",
    "EVENT_ID_PREFIX", 
    "SUPPORTED_RNG_SOURCES",
    # Enums
    "RecordType",
    "HashAlgorithm",
    "SignatureAlgorithm",
    "RiskLevel",
    "ComplianceResult",
    "KeyStatus",
    "KeyType",
    # Interfaces
    "Signer",
    "RNG",
    "Merkle",
    "AnchorDeriver",
    "AnchorStore",
    # Crypto
    "encrypt_aes_gcm",
    "decrypt_aes_gcm",
    "sha256_hash",
    "blake3_hash",
    "sha3_256_hash",
    "compute_hash",
    "hmac_sha256",
    "secure_random_bytes",
    "generate_master_password",
    "CryptoUtils",
    # Signers
    "Ed25519Signer",
    "Ed25519Verifier",
    "ProductionSigner",
    "create_production_signer",
    # Anchors
    "derive_anchor_from_master",
    "derive_master_anchor",
    "derive_dataset_anchor",
    "derive_model_anchor",
    "derive_capsule_anchor",
    "to_hex",
    "from_hex",
    "make_aad",
    # Canonicalization and anchoring
    "Policy",
    "AnchorRecord",
    "Receipt",
    "MockSigner",
    "WORMMerkleTree",
    "CapsuleBuilder",
    "canonical_json",
    "canonicalize_and_hash",
    "validate_required_fields",
    "enrich_metadata_with_defaults",
    "make_anchor",
    "REQUIRED_FIELDS",
    # Merkle
    "MerkleTree",
    "DurableWORMMerkleTree",
    # Policy enforcement
    "PolicyViolation",
    "RiskAssessment",
    "PolicyRule",
    "PolicyEnforcer",
    "HighRiskDomainRule",
    "PiiDetectionRule",
    "TimestampValidationRule",
    "RequiredFieldsRule",
    "create_healthcare_policy_enforcer",
    "create_financial_policy_enforcer",
    "create_gdpr_policy_enforcer",
    # Deterministic operations
    "DeterministicClock",
    "LocaleIndependentOps",
    "DeterministicTimestampGenerator",
    "default_clock",
    "default_timestamp_generator",
    "now_iso",
    "canonical_timestamp",
    "deterministic_timestamp",
    "timestamped_id",
    "normalize_for_determinism",
    "compare_deterministic",
    "sort_deterministic",
    "FixedTimeContext",
    "DeterministicContext",
    # Key management
    "KeyMetadata",
    "KeyBundle",
    "KeyStore",
    "FileSystemKeyStore",
    "KeyManager",
    "create_filesystem_key_manager",
    "create_default_ciaf_key_manager",
    "generate_ciaf_signing_key",
    "get_ciaf_signer",
    # WORM storage
    "WORMRecord",
    "WORMStore",
    "SQLiteWORMStore",
    "LMDBWORMStore",
    "create_sqlite_worm_store",
    "create_lmdb_worm_store",
    # Test vectors
    "TestVector",
    "TestVectorSuite",
    "CIAFTestVectors",
    "generate_test_vectors",
    "export_test_vectors",
    "load_test_vectors",
    "validate_ciaf_implementation"
]
