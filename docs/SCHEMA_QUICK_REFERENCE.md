# CIAF Schema Quick Reference

## Date: March 30, 2026

---

## Proof-Bearing Object Checklist

When creating any CIAF receipt, anchor, or evidence object, include:

### Mandatory Proof Metadata (allOf: proof-metadata.schema.json)

```json
{
  "schema_version": "1.0.0",
  "hash_algorithm": "sha256",
  "signature_algorithm": "ed25519",
  "key_id": "production-signing-2026-03",
  "canonicalization_version": "1.0.0",
  "signature": "<ed25519_signature>",
  "receipt_hash": "<sha256_64char_hex>",
  "created_at": "2026-03-30T12:00:00Z"
}
```

### Lineage References (allOf: lineage-references.schema.json)

Include relevant references for provenance:

```json
{
  "dataset_anchor_ref": "<64char_hex>",
  "training_receipt_ref": "<uuid>",
  "model_anchor_ref": "<64char_hex>",
  "deployment_anchor_ref": "<deployment_id>",
  "prior_receipt_hash": "<64char_hex_or_zeros>"
}
```

---

## Common Patterns

### Pattern 1: Creating a Gate Receipt

```json
{
  "receipt_id": "550e8400-e29b-41d4-a716-446655440000",
  "gate_id": "provenance-gate-1",
  "gate_type": "provenance",
  "evaluation_id": "eval-uuid",
  "action_ref": "action-123",
  "decision": "allow",
  "timestamp": "2026-03-30T12:00:00Z",
  "principal_id": "agent-ml-pipeline",
  "resource_id": "model-v2.1",
  "policy_versions": [
    {"policy_id": "training-policy", "version": "2.0.0"}
  ],
  
  // Include proof-metadata fields
  "schema_version": "1.0.0",
  "hash_algorithm": "sha256",
  "signature_algorithm": "ed25519",
  "key_id": "production-signing-2026-03",
  "signature": "...",
  "receipt_hash": "...",
  "created_at": "2026-03-30T12:00:00Z",
  
  // Include lineage references
  "model_anchor_ref": "...",
  "training_receipt_ref": "...",
  "prior_receipt_hash": "..."
}
```

### Pattern 2: Creating a Training Receipt

```json
{
  "receipt_id": "training-receipt-uuid",
  "session_id": "session-uuid",
  "dataset_anchor": "<64char_hex>",
  "model_anchor": "<64char_hex>",
  "code_digest": "sha256:<64char_hex>",
  "config_digest": "sha256:<64char_hex>",
  "random_seeds": {
    "python": 42,
    "numpy": 42,
    "torch": 42
  },
  "env": {
    "python": "3.11.0",
    "frameworks": {"torch": "2.0.0"},
    "hardware": "NVIDIA A100"
  },
  "metrics": {...},
  "merkle_path": [...],
  
  // Proof metadata
  "schema_version": "1.0.0",
  "signature_algorithm": "ed25519",
  "key_id": "production-signing-2026-03",
  "signature": "...",
  "receipt_hash": "...",
  "created_at": "2026-03-30T12:00:00Z",
  
  // Lineage
  "dataset_anchor_ref": "<64char_hex>",
  "model_anchor_ref": "<64char_hex>",
  "prior_receipt_hash": "..."
}
```

### Pattern 3: Creating an Artifact Evidence Record

```json
{
  "artifact_id": "artifact-12345",
  "artifact_type": "text",
  "mime_type": "text/plain",
  "created_at": "2026-03-30T12:00:00Z",
  "model_id": "gpt-4",
  "model_version": "2026.03",
  "actor_id": "user-alice",
  "prompt_hash": "<64char_hex>",
  "output_hash_raw": "<64char_hex>",
  "output_hash_distributed": "<64char_hex>",
  "watermark_descriptor": {...},
  "hash_set": {
    "content_hash_before_watermark": "<64char_hex>",
    "content_hash_after_watermark": "<64char_hex>",
    "forensic_fragments": {...}
  },
  
  // Proof metadata
  "schema_version": "1.0.0",
  "signature_algorithm": "ed25519",
  "key_id": "production-signing-2026-03",
  "signature": "...",
  "receipt_hash": "...",
  "created_at": "2026-03-30T12:00:00Z",
  
  // Lineage
  "model_anchor_ref": "<64char_hex>",
  "inference_receipt_ref": "<uuid>"
}
```

---

## Canonicalization Quick Reference

### Python
```python
import json

def canonical_json(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )

# Usage
canonical = canonical_json(receipt_data)
receipt_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
```

### JavaScript
```javascript
function canonicalJSON(obj) {
    return JSON.stringify(obj, Object.keys(obj).sort());
}

// Usage
const canonical = canonicalJSON(receiptData);
const receiptHash = crypto.createHash('sha256')
    .update(canonical)
    .digest('hex');
```

---

## Merkle Batching Reference

### Dataset Batching
- **Threshold:** 10,000 records
- **Ordering:** record_id (lex) → timestamp (chrono) → content_hash
- **Tree construction:** SHA-256(left || right) for internal nodes

### Inference Batching
- **Threshold:** 1,000 interactions
- **Hierarchy:** Daily → Session → Batch → Interaction
- **Rollover:** Create root with actual count if threshold not met

### Training Batching
```python
training_root = SHA256(
    dataset_root_1 +
    dataset_root_2 +
    config_hash +
    env_hash +
    model_arch_hash +
    code_commit_hash
)
```

---

## Gate Evaluation Flow

```
1. Action Request
   ↓
2. Gate Lookup (gate_id, gate_type)
   ↓
3. Policy Evaluations (for each rule in policy_set)
   ↓
4. Decision (allow / deny / require_approval)
   ↓
5. If approval required: wait for ApprovalDecision
   ↓
6. If override: record HumanOverrideRecord
   ↓
7. Execute Action (if allowed)
   ↓
8. Generate GateReceipt with signatures
```

---

## Hash Pattern Reference

### SHA-256 Hash (64 hex chars)
```regex
^[a-f0-9]{64}$
```

### UUID v4
```regex
^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$
```

### Semantic Version
```regex
^\\d+\\.\\d+\\.\\d+$
```

### Digest with Algorithm Prefix
```regex
^(sha256|sha3-256|blake3):[a-f0-9]{64}$
```

---

## Schema Composition (allOf)

When a schema uses `allOf`, the object must satisfy **all** referenced schemas:

```json
{
  "type": "object",
  "properties": {
    "gate_id": {"type": "string"},
    "decision": {"type": "string"}
  },
  "allOf": [
    {"$ref": "proof-metadata.schema.json"},
    {"$ref": "lineage-references.schema.json"}
  ]
}
```

This means the object must include:
- Its own properties (gate_id, decision)
- All proof-metadata properties
- All lineage-references properties

---

## Signature Algorithms

### Ed25519 (Recommended)
- **Use for:** External receipts, anchors, capsules
- **Signature size:** 64 bytes
- **Public key size:** 32 bytes
- **Library:** libsodium, NaCl, ed25519-dalek

```python
from nacl.signing import SigningKey

signing_key = SigningKey.generate()
signature = signing_key.sign(canonical_bytes)
```

### HMAC-SHA256 (Internal Only)
- **Use for:** High-volume internal integrity
- **Must not be:** Sole proof for external verification
- **Label clearly:** `"signature_algorithm": "hmac-sha256"`

```python
import hmac
import hashlib

signature = hmac.new(
    key=secret_key,
    msg=canonical_bytes,
    digestmod=hashlib.sha256
).hexdigest()
```

---

## Key ID Format

```
<environment>-<purpose>-<year>-<month>[-<counter>]
```

Examples:
- `production-signing-2026-03`
- `staging-signing-2026-q1`
- `test-signing-2026-dev-01`

---

## Common Enums

### Gate Types
```
provenance | validation | approval | runtime | pre_action | post_action
```

### Enforcement Modes
```
enforcing | permissive | audit_only | disabled
```

### Decisions
```
allow | deny | require_approval | override
```

### Principal Types
```
agent | human | service | system
```

### Record Types
```
dataset | model | inference | anchor | monitoring | compliance | artifact | action
```

### Artifact Types
```
text | image | video | audio | code | mixed
```

---

## Error Handling

### Invalid Schema Version
```json
{
  "error": "schema_version_mismatch",
  "expected": "1.0.0",
  "received": "0.9.0",
  "action": "upgrade_required"
}
```

### Missing Proof Metadata
```json
{
  "error": "missing_required_fields",
  "missing": ["signature", "key_id", "receipt_hash"],
  "schema": "proof-metadata.schema.json"
}
```

### Invalid Signature
```json
{
  "error": "signature_verification_failed",
  "key_id": "production-signing-2026-03",
  "algorithm": "ed25519",
  "action": "reject_receipt"
}
```

---

## Testing Checklist

- [ ] All proof metadata fields present
- [ ] Canonicalization produces deterministic output
- [ ] Signature verifies with public key
- [ ] Receipt hash matches computed hash
- [ ] Lineage references resolve to valid objects
- [ ] Gate evaluation runs before action
- [ ] Denial receipts generated for denied actions
- [ ] Override records created for overrides
- [ ] Merkle paths validate to declared root
- [ ] Timestamps are RFC3339 format

---

**Quick Reference Version:** 1.0.0  
**Last Updated:** March 30, 2026  
**Full Documentation:** See CRYPTOGRAPHIC_STANDARDS.md
