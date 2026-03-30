# CIAF Schema Specification
## Single Source of Truth for CIAF Schema Architecture

**Version:** 1.0.0  
**Date:** March 30, 2026  
**Status:** ✅ AUTHORITATIVE - Production Ready  
**Supersedes:** All previous schema documentation files

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Cryptographic Standards](#2-cryptographic-standards)
3. [Schema Architecture](#3-schema-architecture)
4. [Receipt Metadata Requirements](#4-receipt-metadata-requirements)
5. [Canonicalization Rules](#5-canonicalization-rules)
6. [Merkle Batching Policy](#6-merkle-batching-policy)
7. [Dataset Collection and Evaluation](#7-dataset-collection-and-evaluation)
8. [Training Provenance](#8-training-provenance)
9. [Inference Batching Rules](#9-inference-batching-rules)
10. [Watermark Evidence Policy](#10-watermark-evidence-policy)
11. [Verification Requirements](#11-verification-requirements)
12. [Schema Catalog](#12-schema-catalog)
13. [Implementation Guidance](#13-implementation-guidance)
14. [Migration and Versioning](#14-migration-and-versioning)

---

## 1. Purpose and Scope

### 1.1 Document Purpose

This document defines the **authoritative specification** for all CIAF (Cognitive Insight Audit Framework) schemas, cryptographic standards, receipt requirements, and Merkle batching policies. It ensures:

- **Consistency**: All lifecycle evidence follows uniform standards
- **Verifiability**: Receipts are independently verifiable by third parties
- **Auditability**: Complete provenance tracking with hash chains
- **Compliance**: Built-in regulatory framework mappings
- **Tamper-Evidence**: Cryptographic integrity at all levels

### 1.2 Scope

This specification covers:

- ✅ Cryptographic algorithms and key management
- ✅ Receipt and proof object structure requirements
- ✅ Merkle tree batching policies for datasets, training, and inference
- ✅ Schema versioning and evolution strategies
- ✅ Common reusable schema components
- ✅ Signature envelope structure for production deployments
- ✅ Verification and validation requirements

### 1.3 Applicability

All CIAF implementations, extensions, and third-party integrations **MUST** conform to this specification unless explicitly documented deviations are approved and versioned.

---

## 2. Cryptographic Standards

### 2.1 Content Hashing

**Algorithm:** SHA-256

**Requirements:**
- SHA-256 **MUST** be used for all content hashes, metadata hashes, event hashes, receipt hashes, snapshot hashes, and artifact hashes
- All hash values **MUST** be represented as lowercase hexadecimal strings (64 characters)
- Hash computation **MUST** be performed on canonicalized payloads (see Section 5)

**Pattern:**
```regex
^[a-f0-9]{64}$
```

**Reference Schema:**
```json
{
  "$ref": "common/identifiers/sha256-hash.json"
}
```

### 2.2 External Signatures

**Algorithm:** Ed25519

**Requirements:**
- Ed25519 **MUST** be used for all receipts, anchors, and evidence objects requiring external verification
- Ed25519 **SHOULD** be treated as the default for portable cryptographic proof and non-repudiation
- Signatures **MUST** be base64-encoded for storage and transmission
- Key rotation **MUST** be supported via `key_id` field tracking

**Production Enhancement:**
- KMS (Key Management Service) or HSM (Hardware Security Module) backends **SHOULD** be used for production deployments
- The `key_backend` field **MUST** specify the custody mechanism: `local`, `kms`, `hsm`, `cloudhsm`, or `external`

**Reference Schema:**
```json
{
  "$ref": "common/signature-envelope.json"
}
```

### 2.3 Internal Integrity Signatures

**Algorithm:** HMAC-SHA256 (OPTIONAL)

**Constraints:**
- HMAC-SHA256 **MAY** be used only for internal service-to-service integrity checks
- HMAC-SHA256 **MUST NOT** be the sole signature method for receipts intended for third-party verification
- If used, HMAC-SHA256 **MUST** be clearly marked as internal-only in metadata

### 2.4 Key Management Requirements

**Mandatory Fields:**
- `key_id`: Unique identifier for the signing key (enables rotation)
- `key_backend`: Custody mechanism (`local|kms|hsm|cloudhsm|external`)
- `signature_algorithm`: Algorithm used (default: `Ed25519`)

**Key Rotation:**
- Systems **MUST** support key rotation without breaking verification
- Historical receipts **MUST** remain verifiable using archived public keys
- Key status tracking (active, rotated, revoked) **SHOULD** be maintained

---

## 3. Schema Architecture

### 3.1 Schema Organization

All CIAF schemas are organized into a hierarchical structure:

```
ciaf/schemas/
├── common/                          # Reusable atomic components
│   ├── identifiers/                 # ID patterns (UUID, SHA-256, etc.)
│   ├── enums/                       # Enumeration types
│   ├── patterns/                    # Common structural patterns
│   ├── signature-envelope.json      # Production signature structure
│   ├── signature-metadata.json      # Signature metadata
│   └── cryptographic-policy.json    # Crypto policy configuration
├── receipts/                        # Receipt schemas
├── merkle/                          # Merkle tree structures
├── [domain-specific schemas]        # Training, inference, gates, etc.
└── backups/                         # Backup files (gitignored)
```

### 3.2 Common Reusable Schemas

**Total:** 18 atomic common schemas

#### Identifiers (4 schemas)
1. **`common/identifiers/uuid.json`**
   - Standard UUID v4 pattern
   - Used for: receipt_id, session_id, evaluation_id, approval_id
   - Pattern: RFC 4122 UUID format

2. **`common/identifiers/sha256-hash.json`**
   - SHA-256 hash pattern (64 hex characters)
   - Used for: All content hashes, anchors, digests
   - Pattern: `^[a-f0-9]{64}$`

3. **`common/identifiers/semantic-version.json`**
   - Semantic versioning pattern (semver)
   - Used for: schema_version, policy_version, gate_version
   - Pattern: `^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$`

4. **`common/identifiers/correlation-id.json`**
   - Correlation ID for distributed tracing
   - Used for: Cross-receipt correlation tracking

#### Enumerations (6 schemas)
5. **`common/enums/principal-type.json`**
   - Values: `user`, `agent`, `system`, `service`, `api_key`
   - Used for: Identity and authorization tracking

6. **`common/enums/environment-type.json`**
   - Values: `development`, `staging`, `production`, `testing`, `simulation`
   - Used for: Deployment and runtime environment tagging

7. **`common/enums/decision-type.json`**
   - Values: `allow`, `deny`, `require_approval`
   - Used for: Gate evaluations, policy decisions

8. **`common/enums/gate-type.json`**
   - Values: `provenance`, `validation`, `approval`, `runtime`, `pre_action`, `post_action`
   - Used for: Gate definitions and evaluations

9. **`common/enums/evidence-strength.json`**
   - Values: `REAL`, `SIMULATED`, `FALLBACK`
   - Used for: Enhanced receipts quality tracking

10. **`common/enums/hash-algorithm.json`**
    - Values: `SHA-256`, `SHA-512`, `SHA3-256`
    - Used for: Algorithm specification in cryptographic operations

#### Patterns (5 schemas)
11. **`common/patterns/timestamp.json`**
    - ISO 8601 / RFC 3339 timestamp format
    - Used for: All timestamp fields
    - Pattern: `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$`

12. **`common/patterns/metadata.json`**
    - Flexible metadata object pattern
    - Used for: Extensible metadata in receipts and evidence

13. **`common/patterns/hash-chain-reference.json`**
    - Hash chain reference structure
    - Used for: `prior_receipt_hash` linkage

14. **`common/patterns/merkle-path.json`**
    - Merkle inclusion proof path
    - Values: Array of SHA-256 hashes
    - Used for: Merkle tree inclusion proofs

15. **`common/patterns/policy-obligations.json`**
    - Policy obligation structure
    - Used for: Post-decision obligations and constraints

#### Signature Infrastructure (3 schemas)
16. **`common/signature-envelope.json`**
    - Production-ready signature envelope
    - Required fields: `payload_hash`, `hash_algorithm`, `signature_value`, `signature_encoding`, `signed_at`, `metadata`
    - Replaces legacy proof-metadata inline composition

17. **`common/signature-metadata.json`**
    - Signature metadata object
    - Fields: `signature_algorithm`, `key_id`, `canonicalization_version`, `key_backend`, `signing_service`, `public_key_ref`, `verification_method`

18. **`common/cryptographic-policy.json`**
    - Cryptographic policy configuration
    - Defines allowed algorithms, key rotation policies, HSM/KMS settings

### 3.3 Schema Versioning

**Version Field:**
- All schemas **MUST** include a `version` field at the top level
- Current version: `1.0.0`
- Version follows semantic versioning (semver) rules

**Schema Structure:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://cognitiveinsight.ai/schemas/[schema-name].json",
  "title": "Schema Title",
  "description": "Schema description",
  "version": "1.0.0",
  "type": "object",
  ...
}
```

### 3.4 Reference Pattern

**Using Common Schemas:**
```json
{
  "properties": {
    "receipt_id": {
      "$ref": "common/identifiers/uuid.json",
      "description": "Unique receipt identifier"
    },
    "model_anchor": {
      "$ref": "common/identifiers/sha256-hash.json",
      "description": "Model cryptographic anchor"
    },
    "timestamp": {
      "$ref": "common/patterns/timestamp.json",
      "description": "Creation timestamp"
    }
  }
}
```

---

## 4. Receipt Metadata Requirements

All receipt objects **MUST** include the following metadata structure.

### 4.1 Core Receipt Fields

**Mandatory:**
- `receipt_id` - UUID identifier for the receipt
- `created_at` - ISO 8601 timestamp of receipt creation
- `schema_version` - Version of the receipt schema used

**Strongly Recommended:**
- `correlation_id` - For distributed tracing
- `metadata` - Extensible metadata object

### 4.2 Signature Envelope (Production)

All production receipts **MUST** include a signature envelope:

```json
{
  "signature": {
    "payload_hash": "<sha256_64char_hex>",
    "hash_algorithm": "SHA-256",
    "signature_value": "<base64_encoded_signature>",
    "signature_encoding": "base64",
    "signed_at": "2026-03-30T12:00:00Z",
    "metadata": {
      "signature_algorithm": "Ed25519",
      "key_id": "production-signing-key-2026-q1",
      "canonicalization_version": "1.0.0",
      "key_backend": "hsm",
      "signing_service": "AWS KMS",
      "public_key_ref": "arn:aws:kms:...",
      "verification_method": "https://verify.example.com/keys/..."
    }
  }
}
```

**Schema Reference:**
```json
{
  "signature": {
    "$ref": "common/signature-envelope.json",
    "description": "Cryptographic signature envelope"
  }
}
```

### 4.3 Lineage References

Where applicable, receipts **SHOULD** include lineage references:

```json
{
  "dataset_anchor_ref": "<sha256_hash>",
  "training_receipt_ref": "<uuid>",
  "model_anchor_ref": "<sha256_hash>",
  "predeployment_anchor_ref": "<deployment_id>",
  "deployment_anchor_ref": "<deployment_id>",
  "inference_receipt_ref": "<uuid>",
  "artifact_evidence_ref": "<uuid>",
  "prior_receipt_hash": "<sha256_hash>"
}
```

**Schema Reference:**
```json
{
  "$ref": "lineage-references.schema.json"
}
```

### 4.4 Hash Fields

Common hash fields (all SHA-256):
- `content_hash` - Hash of the content being receipted
- `receipt_hash` - Hash of the complete receipt (before signing)
- `payload_hash` - Hash of the canonicalized payload being signed
- `params_hash` - Hash of action parameters
- `input_hash` - Hash of inference input
- `output_hash` - Hash of inference output
- `prior_receipt_hash` - Hash of previous receipt in chain

### 4.5 Merkle Fields

When receipts are part of Merkle batch:
- `merkle_root` - Root hash of the Merkle tree
- `merkle_path` - Array of hashes forming inclusion proof
- `leaf_index` - Position in Merkle tree (optional)

**Schema Reference:**
```json
{
  "merkle_path": {
    "$ref": "common/patterns/merkle-path.json",
    "description": "Merkle inclusion proof path"
  }
}
```

---

## 5. Canonicalization Rules

### 5.1 Purpose

All signed or hashed payloads **MUST** be canonicalized to ensure:
- Reproducible hashes across systems
- Deterministic signature verification
- Consistent audit trail validation

### 5.2 Canonicalization Algorithm

**Standard:** RFC8785-like deterministic JSON serialization

**Requirements:**
1. **Key Ordering:** All object keys sorted lexicographically (UTF-8 byte order)
2. **No Whitespace:** Remove all unnecessary whitespace
3. **Minimal Encoding:** Use minimal JSON encoding (no escape sequences unless required)
4. **Number Format:** Numbers in minimal representation (no leading zeros, no trailing decimal zeros)
5. **String Normalization:** UTF-8 encoding, NFC normalization
6. **No Extra Fields:** Only include fields defined in schema

### 5.3 Canonicalization Version

**Current Version:** `1.0.0`

All receipts **MUST** include:
```json
{
  "signature": {
    "metadata": {
      "canonicalization_version": "1.0.0"
    }
  }
}
```

This enables:
- Future algorithm upgrades
- Historical receipt verification
- Compatibility tracking

### 5.4 Implementation Guidance

**Python Reference:**
```python
def canonicalize(obj: dict) -> str:
    """Canonical JSON serialization."""
    import json
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )
```

---

## 6. Merkle Batching Policy

### 6.1 Purpose

CIAF uses Merkle batching to:
- Reduce signing costs (sign roots instead of individual items)
- Preserve auditability at scale
- Support efficient inclusion proofs
- Enable hierarchical verification

### 6.2 General Batching Rules

**Process:**
1. Evidence records are hashed individually
2. Hashes are grouped into fixed-size batches
3. Each batch is sealed as a Merkle tree root
4. Batch roots may be grouped into higher-order Merkle trees

**Tree Structure:**
- Binary Merkle trees (each node has 0 or 2 children)
- Leaf-to-root path length: `log₂(n)` where n is batch size
- Unbalanced trees padded with zero hashes or last leaf duplication

### 6.3 Batch Root Creation

A batch root **MUST** be created when:
- The batch reaches its configured threshold size, **OR**
- A logical boundary is reached (session end, job end, day end)

**Threshold Examples:**
- Dataset collection: 10,000 records
- Inference activity: 1,000 interactions
- Training checkpoints: Per epoch or 1,000 steps

### 6.4 Tree of Trees Policy

When multiple batch roots are created for the same logical process:
1. Batch roots are collected
2. A higher-level Merkle tree is constructed over batch roots
3. The higher-level root becomes the canonical summary root
4. This process is recursive for very large datasets

**Example Hierarchy:**
```
Daily Root (Level 3)
├── Session Root 1 (Level 2)
│   ├── Batch Root 1.1 (Level 1) [1,000 interactions]
│   └── Batch Root 1.2 (Level 1) [1,000 interactions]
└── Session Root 2 (Level 2)
    ├── Batch Root 2.1 (Level 1) [1,000 interactions]
    └── Batch Root 2.2 (Level 1) [1,000 interactions]
```

### 6.5 Merkle Path Structure

**Format:** Array of hashes (sibling nodes from leaf to root)

**Schema Reference:**
```json
{
  "merkle_path": {
    "$ref": "common/patterns/merkle-path.json"
  }
}
```

**Example:**
```json
{
  "merkle_path": [
    "a1b2c3d4...",  // Sibling at level 0
    "e5f6g7h8...",  // Sibling at level 1
    "i9j0k1l2..."   // Sibling at level 2
  ]
}
```

**Note:** Some implementations may include direction indicators (`left`/`right`). Standard CIAF format uses hash-only arrays with implicit direction based on leaf index.

---

## 7. Dataset Collection and Evaluation

### 7.1 Batching Threshold

**Default:** 10,000 records per Merkle batch

### 7.2 Process

1. **Raw Collection:**
   - Each data item is hashed individually (SHA-256)
   - Hashes are accumulated until batch threshold

2. **Batch Sealing:**
   - Every 10,000 records → create Merkle tree
   - Root hash becomes batch anchor
   - Merkle paths stored for inclusion proofs

3. **Higher-Order Aggregation:**
   - For datasets > 10,000 records → create tree of batch roots
   - Continue recursively until single dataset-level root
   - Final root = authoritative provenance anchor

### 7.3 Dataset Anchor

**Final Output:**
```json
{
  "dataset_id": "dataset-2026-q1",
  "dataset_anchor": "<final_merkle_root_sha256>",
  "total_records": 150000,
  "batch_count": 15,
  "batch_roots": [
    "<batch_1_root>",
    "<batch_2_root>",
    ...
    "<batch_15_root>"
  ],
  "created_at": "2026-03-30T12:00:00Z",
  "schema_version": "1.0.0"
}
```

### 7.4 Storage and Linkage

Dataset anchor **MUST** be linked to:
- Dataset metadata schema
- Evaluation metadata (if applicable)
- Training receipts (when used for training)
- Model provenance references

**Schema Reference:**
```json
{
  "dataset_anchor_ref": {
    "$ref": "common/identifiers/sha256-hash.json",
    "description": "Reference to dataset Merkle root"
  }
}
```

---

## 8. Training Provenance

### 8.1 Multi-Dataset Handling

**Requirement:** If a model is trained on multiple datasets, the training process **MUST** record all dataset roots.

**Example:**
```json
{
  "training_receipt_id": "training-uuid",
  "datasets": [
    {
      "dataset_id": "dataset-1",
      "dataset_anchor": "<merkle_root_1>",
      "split": "train"
    },
    {
      "dataset_id": "dataset-2",
      "dataset_anchor": "<merkle_root_2>",
      "split": "validation"
    }
  ]
}
```

### 8.2 Training Provenance Root

**Components:**
- Ordered set of dataset roots
- Training configuration hash
- Environment hash (Python version, framework versions, hardware)
- Code digest (Git commit or source hash)
- Random seeds (for reproducibility)
- Model fingerprints (architecture hash, initial weights hash)

**Aggregation:**
```json
{
  "training_provenance_root": "<merkle_root_of_all_components>",
  "components": {
    "datasets": ["<root_1>", "<root_2>"],
    "config_digest": "<sha256>",
    "code_digest": "<sha256>",
    "env_hash": "<sha256>",
    "random_seeds": {...},
    "model_architecture_hash": "<sha256>"
  }
}
```

### 8.3 Linkage Requirements

Training provenance **MUST** be linked to:
- Training snapshot
- Model anchor
- Inference receipts (all inferences reference training receipt)

### 8.4 Dataset Evolution Tracking

**Rule:** If a dataset changes, it **MUST**:
1. Produce a new dataset root
2. Create a new provenance lineage entry
3. Update downstream references explicitly

**Anti-Pattern (FORBIDDEN):**
```json
{
  "dataset_ref": "latest"  // ❌ Never use implicit latest
}
```

**Correct Pattern:**
```json
{
  "dataset_anchor_ref": "a1b2c3d4...",  // ✅ Explicit version
  "dataset_version": "2.1.0"
}
```

---

## 9. Inference Batching Rules

### 9.1 Provenance Preservation

**Requirement:** All inferences **MUST** retain:
- `model_anchor` - Merkle root of the model
- `deployment_anchor` - Deployment configuration anchor
- `dataset_anchor_ref` - Training dataset reference (via training receipt)

**Schema Fragment:**
```json
{
  "model_anchor": {
    "$ref": "common/identifiers/sha256-hash.json"
  },
  "deployment_anchor": {
    "type": "string"
  },
  "training_receipt_ref": {
    "$ref": "common/identifiers/uuid.json"
  }
}
```

### 9.2 Per-Interaction Hashing

**Requirement:** Every inference interaction **MUST** produce an individual evidence hash.

**Hash Components:**
- Input hash (SHA-256 of canonicalized input)
- Output hash (SHA-256 of canonicalized output)
- Model anchor
- Timestamp
- Session ID (if applicable)

### 9.3 Session-Level Batching

**Threshold:** 1,000 interactions per Merkle batch

**Process:**
1. Accumulate interaction hashes within a session
2. Every 1,000 interactions → create Merkle tree root
3. At session end → roll all batch roots into session-level root

**Session Root:**
```json
{
  "session_id": "session-uuid",
  "session_root": "<merkle_root>",
  "interaction_count": 2500,
  "batch_count": 3,
  "batch_roots": [
    "<batch_1_root>",  // Interactions 0-999
    "<batch_2_root>",  // Interactions 1000-1999
    "<batch_3_root>"   // Interactions 2000-2499
  ],
  "started_at": "2026-03-30T10:00:00Z",
  "ended_at": "2026-03-30T14:00:00Z"
}
```

### 9.4 Daily Aggregation

**Requirement:** For each day, all session-level roots **MUST** be rolled into a daily Merkle root.

**Process:**
1. Collect all session roots for the day
2. Create Merkle tree over session roots
3. Daily root becomes primary forensic boundary
4. Chain daily roots via `prior_receipt_hash`

**Daily Root Structure:**
```json
{
  "daily_root_id": "daily-2026-03-30",
  "date": "2026-03-30",
  "daily_root": "<merkle_root>",
  "session_count": 15,
  "total_interactions": 37500,
  "session_roots": [
    "<session_1_root>",
    "<session_2_root>",
    ...
  ],
  "prior_daily_root_hash": "<previous_day_root>",
  "created_at": "2026-03-31T00:00:00Z"
}
```

### 9.5 Audit Support

This structure enables:
- ✅ Inclusion proofs for individual interactions
- ✅ Session reconstruction from batch roots
- ✅ Day-bounded audit review
- ✅ Efficient forensic replay
- ✅ Chain-of-custody from interaction → session → day

---

## 10. Watermark Evidence Policy

### 10.1 Watermark Generation Timing

**Requirement:** Watermark evidence **MUST** be generated at inference time.

### 10.2 Session-Level Batching

**Threshold:** Per-session Merkle trees for watermark artifacts

**Process:**
1. Each watermarked output creates watermark evidence
2. Evidence is hashed and accumulated per session
3. At session end → create watermark session root

### 10.3 Daily Aggregation

**Requirement:** One daily watermark aggregation per user or agent

**Hierarchy:**
```
Daily Watermark Root (per user)
├── Session 1 Root
│   ├── Watermark Evidence 1
│   ├── Watermark Evidence 2
│   └── ...
└── Session 2 Root
    ├── Watermark Evidence 1
    └── ...
```

### 10.4 Watermark Evidence Structure

**Required Fields:**
```json
{
  "watermark_id": "<uuid>",
  "dataset_anchor_ref": "<sha256>",
  "model_anchor": "<sha256>",
  "deployment_anchor": "<deployment_id>",
  "inference_receipt_ref": "<uuid>",
  "before_hash": "<sha256_of_unwatermarked>",
  "after_hash": "<sha256_of_watermarked>",
  "watermark_method": "pdf|text|image|video|audio|metadata",
  "artifact_id": "<unique_artifact_id>",
  "perceptual_hash": "<phash>",  // When relevant
  "created_at": "2026-03-30T12:00:00Z"
}
```

### 10.5 Watermark Method Types

**Enumeration:**
- `text_embedded` - Text watermarking (linguistic patterns)
- `image_steganographic` - Image steganography
- `image_visible` - Visible image watermarks
- `metadata` - Metadata-based watermarks
- `qr_code` - QR code watermarks
- `perceptual` - Perceptual hashing
- `pdf` - PDF-specific watermarking
- `video` - Video watermarking
- `audio` - Audio watermarking
- `none` - No watermark applied

### 10.6 Provenance Preservation

Watermark evidence enables:
- Direct watermark provenance for single session
- Broader daily evidence of output integrity by user
- Forensic artifact tracking
- Regulatory compliance (e.g., AI Act Article 50 watermarking)

---

## 11. Verification Requirements

### 11.1 Verifiable Elements

Any verifier (internal or third-party) **MUST** be able to:

1. **Recompute Content Hash:**
   - Retrieve receipt or evidence object
   - Canonicalize according to `canonicalization_version`
   - Compute SHA-256 hash
   - Compare to declared `receipt_hash` or `content_hash`

2. **Validate Signature:**
   - Extract `signature` envelope
   - Retrieve public key using `key_id` and `key_backend`
   - Verify signature using declared `signature_algorithm`
   - Confirm `payload_hash` matches canonicalized payload

3. **Verify Merkle Inclusion:**
   - Given a leaf hash and `merkle_path`
   - Recompute path to root
   - Compare to declared `merkle_root`

4. **Verify Higher-Order Lineage:**
   - Session roots roll up to daily root
   - Batch roots roll up to dataset root
   - Training provenance links datasets + config + code

5. **Validate Hash Chains:**
   - Each receipt references `prior_receipt_hash`
   - Chain integrity verified by recomputing hashes

### 11.2 Verification Tools

CIAF implementations **SHOULD** provide:
- Standalone verification scripts (no CIAF dependency)
- Command-line verification tools
- Web-based verification portals
- API endpoints for programmatic verification

### 11.3 Offline Verification

**Requirement:** Receipts **MUST** be verifiable offline without access to CIAF systems.

**Minimal Requirements:**
- Receipt JSON object
- Public key (from `key_id`)
- Canonicalization algorithm (from `canonicalization_version`)
- Merkle root (if verifying inclusion)

---

## 12. Schema Catalog

### 12.1 Schema Categories

**Total Schemas:** 64 (as of version 1.0.0)

#### Core CIAF (8 schemas)
1. `receipt.schema.json` - Base CIAF receipt
2. `anchor.schema.json` - Merkle root anchors
3. `capsule.schema.json` - Provenance capsules
4. `merkle-proof.schema.json` - Merkle inclusion proofs
5. `dataset-metadata.schema.json` - Dataset descriptors
6. `lineage-references.schema.json` - Provenance linkage
7. `proof-metadata.schema.json` - Legacy proof structure (deprecated)
8. `identity.schema.json` - Principal identity

#### Training & Inference (7 schemas)
9. `training-receipt-enhanced.schema.json` - Training receipts
10. `training-checkpoint.schema.json` - Training checkpoints
11. `training-environment.schema.json` - Training environment
12. `training-metrics.schema.json` - Training metrics
13. `inference-receipt-enhanced.schema.json` - Inference receipts
14. `model-architecture.schema.json` - Model architecture
15. `predeployment-anchor.schema.json` - Pre-deployment anchors

#### Governance & Policy (9 schemas)
16. `gate-definition.schema.json` - Gate definitions
17. `gate-evaluation.schema.json` - Gate evaluations
18. `gate-receipt.schema.json` - Gate receipts
19. `policy-set.schema.json` - Policy collections
20. `policy-rule.schema.json` - Individual policy rules
21. `policy-evaluation.schema.json` - Policy evaluation results
22. `approval-decision.schema.json` - Approval decisions
23. `human-override-record.schema.json` - Override records
24. `elevation-grant.schema.json` - Privilege elevation

#### Agentic & Actions (5 schemas)
25. `action-request.schema.json` - Action requests
26. `action-receipt.schema.json` - Action execution receipts
27. `execution-result.schema.json` - Execution results
28. `permission.schema.json` - Permission grants
29. `resource.schema.json` - Resource descriptors

#### Artifacts & Evidence (6 schemas)
30. `artifact-evidence.schema.json` - Artifact forensic evidence
31. `artifact-fingerprint.schema.json` - Artifact fingerprints
32. `artifact-hash-set.schema.json` - Artifact hash collections
33. `build-artifact.schema.json` - Build artifacts
34. `sbom.schema.json` - Software Bill of Materials
35. `deployment-anchor.schema.json` - Deployment anchors

#### Watermarking (4 schemas)
36. `watermark-descriptor.schema.json` - Watermark descriptors
37. `text-forensic-fragment.schema.json` - Text watermark fragments
38. `image-forensic-fragment.schema.json` - Image watermark fragments
39. `forensic-fragment-set.schema.json` - Fragment collections

#### Compliance (2 schemas)
40. `corrective-action.schema.json` - Corrective actions
41. `policy.schema.json` - Compliance policies

#### Common Schemas (18 schemas)
42-45. **Identifiers:** uuid, sha256-hash, semantic-version, correlation-id
46-51. **Enums:** principal-type, environment-type, decision-type, gate-type, evidence-strength, hash-algorithm
52-56. **Patterns:** timestamp, metadata, hash-chain-reference, merkle-path, policy-obligations
57-59. **Signature:** signature-envelope, signature-metadata, cryptographic-policy

#### Specialized (5 schemas)
60. `vault.schema.json` - Vault storage
61. `merkle/merkle-batch.json` - Merkle batching
62. `receipts/runtime-receipt.json` - Runtime receipts
63. `receipts/runtime-receipt-example.json` - Example receipt
64. `merkle/merkle-batch-example.json` - Example batch

### 12.2 Schema Status

**Production-Ready:** All 64 schemas
**Version:** 1.0.0
**Validation:** JSON Schema Draft 2020-12

---

## 13. Implementation Guidance

### 13.1 Default Thresholds

| Context | Threshold | Aggregation Boundary |
|---------|-----------|---------------------|
| Dataset Collection | 10,000 records | Per batch → Dataset root |
| Dataset Evaluation | 10,000 records | Per batch → Evaluation root |
| Inference Activity | 1,000 interactions | Per batch → Session root |
| Inference Sessions | Per session | All sessions → Daily root |
| Watermark Evidence | Per session | Per user/agent → Daily root |
| Training Checkpoints | 1,000 steps or per epoch | Per checkpoint → Training root |

### 13.2 Configuration Override

**Recommendation:** Thresholds **SHOULD** be configurable via:
- Environment variables
- Configuration files
- Runtime API parameters

**Example Configuration:**
```json
{
  "merkle_batching": {
    "dataset_batch_size": 10000,
    "inference_batch_size": 1000,
    "training_checkpoint_interval": 1000,
    "session_aggregation": "session_end",
    "daily_aggregation": "midnight_utc"
  }
}
```

### 13.3 Timestamp Generation

**Standard:**
```python
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).isoformat()
```

**Anti-Pattern (FORBIDDEN):**
```python
# ❌ NEVER USE - deprecated and timezone-naive
timestamp = datetime.utcnow().isoformat()
```

### 13.4 Hash Computation

**Standard:**
```python
import hashlib
import json

def compute_hash(obj: dict) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
```

### 13.5 Signature Generation

**Standard:**
```python
from nacl import signing
import base64

def sign_receipt(receipt: dict, private_key: signing.SigningKey) -> dict:
    # Canonicalize
    canonical = canonicalize(receipt)
    payload_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    # Sign
    signature_bytes = private_key.sign(canonical.encode('utf-8')).signature
    signature_value = base64.b64encode(signature_bytes).decode('ascii')
    
    # Create envelope
    receipt['signature'] = {
        "payload_hash": payload_hash,
        "hash_algorithm": "SHA-256",
        "signature_value": signature_value,
        "signature_encoding": "base64",
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "signature_algorithm": "Ed25519",
            "key_id": "production-key-2026-q1",
            "canonicalization_version": "1.0.0",
            "key_backend": "local"
        }
    }
    return receipt
```

### 13.6 Merkle Tree Construction

**Standard:** Binary Merkle tree with SHA-256

**Example:**
```python
def build_merkle_tree(leaf_hashes: list) -> str:
    if len(leaf_hashes) == 0:
        return "0" * 64  # Zero hash
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]
    
    # Build tree bottom-up
    level = leaf_hashes
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            parent = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
            next_level.append(parent)
        level = next_level
    
    return level[0]  # Root hash
```

---

## 14. Migration and Versioning

### 14.1 Schema Evolution Strategy

**Principles:**
- Backward compatibility whenever possible
- Explicit version field in all schemas
- Deprecation warnings before removal
- Migration scripts for breaking changes

### 14.2 Version Upgrade Path

**Current Version:** 1.0.0

**Future Versions:**
- **1.x.y** - Backward compatible additions and fixes
- **2.0.0** - Breaking changes requiring migration

### 14.3 Migration Tools

**Location:** `tools/`

**Available Tools:**
1. `migrate_to_common_schemas.py` - Migrate to common schema references
2. `validate_schemas.py` - Validate all schemas against JSON Schema spec
3. `add_schema_version.py` - Add version field to schemas

**Usage:**
```bash
# Validate current schemas
python tools/validate_schemas.py

# Migrate to common schemas (dry-run)
python tools/migrate_to_common_schemas.py --dry-run

# Apply migration
python tools/migrate_to_common_schemas.py
```

### 14.4 Deprecation Policy

**Legacy Schemas:**
- `proof-metadata.schema.json` - Replaced by `signature-envelope.json`
- Status: Deprecated but supported for backward compatibility
- Removal: Version 2.0.0 (with 6-month notice)

### 14.5 Backward Compatibility

**Requirement:** Systems **MUST** accept receipts from older schema versions for verification.

**Implementation:**
- Key validation logic based on `schema_version` field
- Support multiple canonicalization versions
- Archive historical public keys for signature verification

---

## Document Changelog

### Version 1.0.0 (March 30, 2026)

**Initial Release:**
- Consolidated cryptographic standards
- Unified Merkle batching policies
- Complete schema catalog (64 schemas)
- Common schema architecture (18 reusable components)
- Production signature envelope specification
- Verification requirements
- Implementation guidance

**Supersedes:**
- CIAF_COMPLETE_SCHEMA.md
- COMMON_SCHEMA_CONSOLIDATION_ANALYSIS.md
- SCHEMA_ENHANCEMENT_SUMMARY.md
- SCHEMA_MIGRATION_TO_SIGNATURE_ENVELOPE.md
- SCHEMA_MIGRATION_QUICK_START.md

---

## Compliance and Certification

**This specification aligns with:**
- ✅ ISO/IEC 27001 - Information Security Management
- ✅ NIST AI RMF - AI Risk Management Framework
- ✅ EU AI Act - Transparency and auditability requirements
- ✅ GDPR Article 5 - Data integrity and accountability
- ✅ HIPAA - Audit trail requirements (when applicable)
- ✅ SOC 2 Type II - Logging and monitoring controls

---

## References

1. **JSON Schema Draft 2020-12:** https://json-schema.org/draft/2020-12/schema
2. **RFC 8785 (JSON Canonical):** https://www.rfc-editor.org/rfc/rfc8785
3. **RFC 4122 (UUID):** https://www.rfc-editor.org/rfc/rfc4122
4. **RFC 3339 (Timestamps):** https://www.rfc-editor.org/rfc/rfc3339
5. **Ed25519 Specification:** https://ed25519.cr.yp.to/
6. **SHA-256 (FIPS 180-4):** https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

---

**Document Control:**
- **Authority:** CIAF Core Team
- **Review Cycle:** Quarterly
- **Change Control:** GitHub Issues + Pull Requests
- **Feedback:** ciaf-schema-feedback@cognitiveinsight.ai

---

**END OF SPECIFICATION**
