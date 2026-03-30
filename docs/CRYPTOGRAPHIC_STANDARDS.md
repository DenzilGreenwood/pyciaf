# CIAF Cryptographic and Governance Standards

## Version: 1.0.0
## Last Updated: March 30, 2026

---

## Executive Summary

This document defines the mandatory cryptographic standards, canonicalization rules, Merkle batching policies, and governance patterns for the CIAF (Compliant Interpretable AI Framework) system. All implementations must adhere to these standards to ensure:

* **Cryptographic integrity**: reproducible verification across systems
* **Interoperability**: portable evidence and audit trails
* **Long-term verifiability**: future-proof provenance
* **Compliance readiness**: audit-friendly evidence structures

---

## 1. Cryptographic Standards

### 1.1 Hash Algorithm Standard

**SHA-256 is the mandatory content hashing algorithm** for all CIAF evidence objects.

* **Primary:** SHA-256 (32-byte/256-bit output)
* **Acceptable alternatives:** SHA3-256, BLAKE3 (must be explicitly declared)
* **Deprecated:** MD5, SHA-1 (never use for security-critical operations)

**Rationale:**
* Universally supported across platforms and languages
* NIST-approved and cryptographically secure
* Optimal balance of security, performance, and adoption
* Compatible with blockchain and external timestamp services

**Implementation requirement:**
All hash values must be represented as lowercase hexadecimal strings (64 characters).

### 1.2 Signature Algorithm Standard

**Ed25519 is the default signature algorithm** for externally verifiable receipts.

* **Primary:** Ed25519 (Edwards-curve Digital Signature Algorithm)
* **Internal-only alternative:** HMAC-SHA256 (must not be the sole proof for third-party verification)
* **Deprecated:** RSA-2048, ECDSA-secp256k1 (unless required for external system interop)

**Rationale:**
* Small signature size (64 bytes)
* Fast verification
* Strong security guarantees
* Deterministic (no random nonce required)
* Wide library support (libsodium, NaCl, ed25519-dalek)

**Usage guidelines:**
* Use **Ed25519** for:
  * Anchor records
  * Capsules for external verification
  * Training and inference receipts
  * Gate receipts
  * Action receipts
  * Artifact evidence
  
* Use **HMAC-SHA256** only for:
  * Internal service-to-service integrity
  * Performance-critical high-volume operations
  * Must be explicitly labeled as `"signature_algorithm": "hmac-sha256"`

### 1.3 Required Proof Metadata Fields

All proof-bearing objects (receipts, anchors, capsules, gate evaluations, policy evaluations) **must** include the following standardized fields:

```json
{
  "schema_version": "1.0.0",
  "hash_algorithm": "sha256",
  "signature_algorithm": "ed25519",
  "key_id": "signing-key-production-2026-03",
  "canonicalization_version": "1.0.0",
  "signature": "...",
  "receipt_hash": "...",
  "created_at": "2026-03-30T12:00:00Z"
}
```

**Field definitions:**

* `schema_version`: Semantic version of schema (enables forward/backward compatibility)
* `hash_algorithm`: Hash algorithm used (default: `"sha256"`)
* `signature_algorithm`: Signature algorithm used (default: `"ed25519"`)
* `key_id`: Identifier for the signing key (enables key rotation and lookup)
* `canonicalization_version`: Version of canonicalization rules applied
* `signature`: Base64 or hex-encoded cryptographic signature
* `receipt_hash`: SHA-256 hash of the canonical object representation
* `created_at`: RFC3339 timestamp of object creation

---

## 2. Canonicalization Rules

### 2.1 Canonical JSON Standard (Version 1.0.0)

All signed or hashed payloads **must** be canonicalized deterministically before hashing or signing.

**Canonicalization rules:**
1. **Sort keys alphabetically** at all nesting levels
2. **No whitespace** between keys, values, or structural characters
3. **Use compact separators:** `,` (comma) and `:` (colon) only
4. **Preserve Unicode** (do not escape non-ASCII characters unless required by JSON spec)
5. **Serialize null as `null`**, not as empty string or omitted
6. **Numbers:** use standard JSON number representation (no leading zeros, no trailing decimals)

**Python implementation:**
```python
import json

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
```

**JavaScript implementation:**
```javascript
function canonicalJSON(obj) {
    return JSON.stringify(obj, Object.keys(obj).sort(), null, 0);
}
```

### 2.2 Versioning and Evolution

Canonicalization rules are versioned using semantic versioning.

* Breaking changes require a major version increment
* All receipts must declare the `canonicalization_version` used
* Verifiers must support at least the last 2 major versions

---

## 3. Merkle Batching Policies

CIAF uses Merkle tree batching to scale provenance tracking while maintaining cryptographic integrity and enabling efficient inclusion proofs.

### 3.1 Dataset Collection and Evaluation

**Batching policy:**
* Create a Merkle root every **10,000 records**
* Create higher-order Merkle trees over batch roots
* Continue until a single dataset-level root is produced
* Store the final root as the authoritative dataset provenance reference

**Ordering requirement:**
Records must be canonically ordered before tree construction:
* Primary sort: record ID (lexicographic)
* Secondary sort: timestamp (chronological)
* Tertiary sort: content hash (ensures determinism even with duplicate IDs)

**Implementation notes:**
* Leaf nodes contain SHA-256(canonical_json(record))
* Internal nodes contain SHA-256(left_hash || right_hash)
* Unbalanced trees: duplicate the last node to complete the level

### 3.2 Dataset Ingest into Training

**Batching policy:**
* Each dataset used in training carries its own final dataset root
* Training references the ordered set of dataset roots
* Create a training provenance root across:
  * Dataset roots (in deterministic order)
  * Config hash
  * Environment hash
  * Model architecture fingerprint
  * Code commit hash

**Training root computation:**
```
training_root = SHA-256(
    dataset_root_1 || 
    dataset_root_2 || 
    ... || 
    config_hash || 
    env_hash || 
    model_arch_hash || 
    code_commit_hash
)
```

### 3.3 Inference Interaction Batching

**Batching policy:**
* Hash every interaction individually
* Create a Merkle root every **1,000 interactions**
* Roll interaction roots into session-level roots
* Roll session roots into daily roots

**Hierarchy:**
```
Daily Root
├── Session 1 Root
│   ├── Batch 1 Root (1,000 interactions)
│   ├── Batch 2 Root (1,000 interactions)
│   └── ...
├── Session 2 Root
│   └── ...
└── Session N Root
```

**Rollover behavior:**
* Sessions ending before threshold: create root with actual interaction count
* Days with partial sessions: aggregate all session roots regardless of count
* Maintain deterministic ordering: sort sessions by session_id, interactions by timestamp

### 3.4 Watermark Evidence Batching

**Batching policy:**
* Create watermark evidence per user session
* Create a daily per-user watermark aggregation root
* Aggregate daily roots into monthly audit roots

**Purpose:**
* Privacy-preserving: individual artifacts not exposed
* Audit-ready: can prove watermark evidence exists without revealing content
* Forensically sound: inclusion proofs can be materialized on demand

---

## 4. Governance Enforcement Standards

### 4.1 Gate-Before-Action Pattern

All governed actions must follow the gate-before-action pattern:

1. **Action Request** → 2. **Gate Evaluation** → 3. **Policy Evaluation(s)** → 4. **Decision** → 5. **Action Execution** → 6. **Gate Receipt**

**Mandatory requirements:**
* Gate evaluation must complete before action execution
* All policy versions must be recorded in gate receipt
* Denied actions must generate denial receipts (not just log entries)
* Override actions must be explicitly recorded with justification

### 4.2 Policy Evaluation Timing

Policy evaluations must be timestamped and recorded **before** the action proceeds.

**Anti-pattern (forbidden):**
```
// BAD: Log evaluation after action completes
execute_action()
log_policy_evaluation()
```

**Correct pattern (mandatory):**
```
// GOOD: Evaluate policy, record decision, then act
evaluation = evaluate_policy()
if evaluation.decision == "allow":
    result = execute_action()
    create_gate_receipt(evaluation, result)
```

### 4.3 Denial Receipt Requirement

All denied actions must generate a cryptographically signed denial receipt containing:
* Gate ID and evaluation ID
* Principal ID and request context
* Policy rule that triggered denial
* Timestamp of denial
* Signature proving denial occurred

**Purpose:**
* Prevents "absence of evidence" attacks
* Proves governance was enforced
* Provides audit trail of denied attempts
* Enables incident reconstruction

---

## 5. Lineage Reference Standards

### 5.1 Required Lineage Fields

All provenance-bearing objects must include explicit lineage references:

* `dataset_anchor_ref`: Reference to dataset anchor hash
* `training_snapshot_ref`: Reference to training snapshot identifier
* `training_receipt_ref`: Reference to training receipt UUID
* `model_anchor_ref`: Reference to model anchor hash
* `predeployment_anchor_ref`: Reference to predeployment anchor
* `deployment_anchor_ref`: Reference to deployment anchor
* `inference_receipt_ref`: Reference to inference receipt UUID
* `artifact_evidence_ref`: Reference to artifact evidence identifier
* `prior_receipt_hash`: Hash of prior receipt for chain linking

### 5.2 Chain Validation Pattern

Receipts that form audit chains (e.g., action receipts, inference receipts) must include:

```json
{
  "receipt_id": "current-receipt-uuid",
  "prior_receipt_hash": "sha256_hash_of_previous_receipt",
  "receipt_hash": "sha256_hash_of_this_receipt"
}
```

**Validation algorithm:**
1. Verify `receipt_hash` matches actual hash of current receipt
2. Retrieve previous receipt using `prior_receipt_hash`
3. Verify previous receipt's `receipt_hash` matches `prior_receipt_hash`
4. Continue backward until genesis receipt (prior_receipt_hash = "0" * 64)

---

## 6. Key Management Requirements

### 6.1 Key Rotation Policy

* Production signing keys must be rotated annually (maximum)
* Key rotation must not break verification of historical receipts
* All receipts must include `key_id` to support key lookup

**Key ID format:**
```
<environment>-<purpose>-<year>-<month>[-<counter>]

Examples:
- production-signing-2026-03
- staging-signing-2026-q1
- test-signing-2026-dev
```

### 6.2 Key Storage

* Private keys must be stored in HSM or secure key management service
* Key material must never be logged or transmitted in plaintext
* Key access must be audited and restricted to authorized services

---

## 7. Schema Versioning

All schemas use semantic versioning (`major.minor.patch`):

* **Major:** Breaking changes (field removal, type changes)
* **Minor:** Backward-compatible additions (new optional fields)
* **Patch:** Non-functional changes (description updates, examples)

**Compatibility requirement:**
Verifiers must support at least N-1 major versions to allow graceful migration.

---

## 8. Compliance Mapping

CIAF standards map to regulatory requirements:

| Standard | GDPR | EU AI Act | SOC 2 | ISO 27001 |
|----------|------|-----------|-------|-----------|
| Cryptographic integrity | Art. 32 | Art. 17 | CC6.1 | A.10.1.1 |
| Audit trails | Art. 30 | Art. 12 | CC7.2 | A.12.4.1 |
| Access control (gates) | Art. 25 | Art. 14 | CC6.2 | A.9.4.1 |
| Data lineage | Art. 5(1)(a) | Art. 10 | CC6.8 | A.8.2.3 |

---

## 9. Implementation Checklist

- [ ] All proof-bearing objects include `proof-metadata` fields
- [ ] All provenance objects include `lineage-references` fields
- [ ] SHA-256 used for all content hashing
- [ ] Ed25519 used for external receipts
- [ ] Canonicalization applied before hashing/signing
- [ ] Merkle batching follows defined thresholds
- [ ] Gate-before-action pattern enforced
- [ ] Denial receipts generated for all denied actions
- [ ] Key rotation policy implemented
- [ ] Schema versions declared in all objects

---

## 10. References

* NIST FIPS 180-4 (SHA-256)
* RFC 8032 (Ed25519)
* RFC 3339 (Timestamps)
* RFC 6902 (JSON Patch for schema evolution)

---

**Document Owner:** CIAF Architecture Team  
**Review Cycle:** Quarterly  
**Next Review:** June 30, 2026
