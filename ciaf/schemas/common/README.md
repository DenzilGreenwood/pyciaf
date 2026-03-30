# CIAF Common Reusable Schemas

## Overview

This directory contains reusable JSON Schema components that standardize common patterns across all CIAF schemas. These atomic schemas can be referenced via `$ref` to ensure consistency, reduce duplication, and simplify maintenance.

---

## Directory Structure

```
common/
├── identifiers/          # Common identifier patterns
│   ├── uuid.json
│   ├── sha256-hash.json
│   ├── semantic-version.json
│   └── correlation-id.json
├── enums/               # Common enumeration types
│   ├── principal-type.json
│   ├── environment-type.json
│   ├── decision-type.json
│   ├── gate-type.json
│   ├── evidence-strength.json
│   └── hash-algorithm.json
├── patterns/            # Common structural patterns
│   ├── timestamp.json
│   ├── metadata.json
│   ├── hash-chain-reference.json
│   ├── merkle-path.json
│   └── policy-obligations.json
├── signature-metadata.json       # Signature metadata
├── signature-envelope.json       # Complete signature payload
└── cryptographic-policy.json     # Cryptographic policy profile
```

---

## Schema Categories

### Identifiers (`identifiers/`)

Atomic identifier patterns used throughout CIAF.

#### `uuid.json`
**Purpose:** Standard UUID v4 identifier  
**Pattern:** `format: uuid`  
**Used in:** receipt_id, session_id, evaluation_id, approval_id  
**Example:**
```json
{
  "receipt_id": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/identifiers/uuid.json"
  }
}
```

#### `sha256-hash.json`
**Purpose:** SHA-256 hash (64 character hex)  
**Pattern:** `^[a-f0-9]{64}$`  
**Used in:** All content hashes, anchors, integrity checks  
**Example:**
```json
{
  "payload_hash": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/identifiers/sha256-hash.json"
  }
}
```

#### `semantic-version.json`
**Purpose:** Semantic versioning (major.minor.patch)  
**Pattern:** `^\d+\.\d+\.\d+$`  
**Used in:** schema_version, policy_version, canonicalization_version  
**Example:**
```json
{
  "schema_version": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/identifiers/semantic-version.json"
  }
}
```

#### `correlation-id.json`
**Purpose:** Cross-system event correlation  
**Pattern:** String (1-256 chars)  
**Used in:** Request tracing, distributed transactions  
**Example:**
```json
{
  "correlation_id": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/identifiers/correlation-id.json"
  }
}
```

---

### Enums (`enums/`)

Standardized enumeration types for consistency.

#### `principal-type.json`
**Values:** `agent`, `human`, `service`, `system`  
**Used in:** Identity, action receipts, authorization  
**Example:**
```json
{
  "principal_type": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/principal-type.json"
  }
}
```

#### `environment-type.json`
**Values:** `production`, `staging`, `development`, `test`  
**Used in:** Deployment anchors, identity context  
**Example:**
```json
{
  "environment": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/environment-type.json"
  }
}
```

#### `decision-type.json`
**Values:** `allow`, `deny`, `require_approval`, `not_applicable`  
**Used in:** Policy evaluation, gate decisions, authorization  
**Example:**
```json
{
  "decision": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/decision-type.json"
  }
}
```

#### `gate-type.json`
**Values:** `provenance`, `validation`, `approval`, `runtime`, `pre_action`, `post_action`  
**Used in:** Gate definitions, gate evaluations  
**Example:**
```json
{
  "gate_type": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/gate-type.json"
  }
}
```

#### `evidence-strength.json`
**Values:** `real`, `simulated`, `fallback`  
**Used in:** Receipt validation, evidence quality tracking  
**Example:**
```json
{
  "evidence_strength": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/evidence-strength.json"
  }
}
```

#### `hash-algorithm.json`
**Values:** `sha256` (default), `sha3-256`, `blake3`  
**Used in:** Cryptographic policies, proof metadata  
**Example:**
```json
{
  "hash_algorithm": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/enums/hash-algorithm.json"
  }
}
```

---

### Patterns (`patterns/`)

Common structural patterns and data types.

#### `timestamp.json`
**Format:** ISO 8601 / RFC3339  
**Pattern:** `format: date-time`  
**Used in:** All temporal fields  
**Example:**
```json
{
  "timestamp": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/patterns/timestamp.json"
  }
}
```

#### `metadata.json`
**Type:** Generic object  
**Pattern:** `additionalProperties: true`  
**Used in:** Extensible metadata fields  
**Example:**
```json
{
  "metadata": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/patterns/metadata.json"
  }
}
```

#### `hash-chain-reference.json`
**Purpose:** Link to previous receipt (tamper-evident)  
**Pattern:** SHA-256 hash (64 hex chars, genesis = 64 zeros)  
**Used in:** Receipt chaining, audit trails  
**Example:**
```json
{
  "prior_receipt_hash": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/patterns/hash-chain-reference.json"
  }
}
```

#### `merkle-path.json`
**Type:** Array of SHA-256 hashes  
**Used in:** Merkle inclusion proofs  
**Example:**
```json
{
  "merkle_path": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/patterns/merkle-path.json"
  }
}
```

#### `policy-obligations.json`
**Type:** Array of strings  
**Used in:** Post-decision obligations  
**Example:**
```json
{
  "policy_obligations": {
    "$ref": "https://cognitiveinsight.ai/schemas/common/patterns/policy-obligations.json"
  }
}
```

---

## Usage Examples

### Before (Duplicated Pattern)

Multiple schemas define SHA-256 hash independently:

**inference-receipt-enhanced.schema.json:**
```json
{
  "input_hash": {
    "type": "string",
    "pattern": "^[a-f0-9]{64}$",
    "description": "SHA-256 hash of input"
  },
  "output_hash": {
    "type": "string",
    "pattern": "^[a-f0-9]{64}$",
    "description": "SHA-256 hash of output"
  }
}
```

**artifact-evidence.schema.json:**
```json
{
  "artifact_hash": {
    "type": "string",
    "pattern": "^[a-f0-9]{64}$",
    "description": "SHA-256 hash of artifact"
  },
  "content_hash": {
    "type": "string",
    "pattern": "^[a-f0-9]{64}$",
    "description": "SHA-256 hash of content"
  }
}
```

### After (Reusable Component)

All schemas reference the common sha256-hash:

**inference-receipt-enhanced.schema.json:**
```json
{
  "input_hash": {
    "$ref": "common/identifiers/sha256-hash.json",
    "description": "SHA-256 hash of input"
  },
  "output_hash": {
    "$ref": "common/identifiers/sha256-hash.json",
    "description": "SHA-256 hash of output"
  }
}
```

**artifact-evidence.schema.json:**
```json
{
  "artifact_hash": {
    "$ref": "common/identifiers/sha256-hash.json",
    "description": "SHA-256 hash of artifact"
  },
  "content_hash": {
    "$ref": "common/identifiers/sha256-hash.json",
    "description": "SHA-256 hash of content"
  }
}
```

---

## Benefits

### Consistency
✅ **Single source of truth** - Update pattern once, applies everywhere  
✅ **Eliminates drift** - No schema-specific variations  
✅ **Clear standards** - Explicit naming conventions  

### Maintainability
✅ **Easy updates** - Change pattern in one place  
✅ **Reduced duplication** - Smaller schema files  
✅ **Clear intent** - Descriptive schema names  

### Validation
✅ **Shared validation rules** - Consistent across all schemas  
✅ **Type safety** - Enforced patterns  
✅ **Better errors** - Clear validation messages  

### Documentation
✅ **Self-documenting** - Schema names explain purpose  
✅ **Centralized docs** - This README covers all patterns  
✅ **Examples included** - Every schema has examples  

---

## Migration Guide

### Step 1: Identify Repeated Patterns

Search for duplicate patterns in your schema:
```bash
# Find all SHA-256 hash patterns
grep -r "pattern.*\[a-f0-9\]{64}" ciaf/schemas/*.json

# Find all UUID patterns
grep -r "format.*uuid" ciaf/schemas/*.json

# Find all date-time patterns
grep -r "format.*date-time" ciaf/schemas/*.json
```

### Step 2: Replace with $ref

**Before:**
```json
{
  "receipt_id": {
    "type": "string",
    "format": "uuid",
    "description": "Unique receipt identifier"
  }
}
```

**After:**
```json
{
  "receipt_id": {
    "$ref": "common/identifiers/uuid.json",
    "description": "Unique receipt identifier"
  }
}
```

### Step 3: Validate

Use ajv or another JSON Schema validator:
```bash
ajv validate -s your-schema.json -d test-data.json
```

---

## Coverage Statistics

### Schemas Using Common Components (Current)

| Component | Potential Locations | Migrated | Coverage |
|-----------|-------------------|----------|----------|
| SHA-256 Hash | ~50 | 0 | 0% |
| UUID | ~21 | 0 | 0% |
| Timestamp | ~48 | 0 | 0% |
| Principal Type | ~5 | 0 | 0% |
| Environment Type | ~3 | 0 | 0% |
| Decision Type | ~8 | 0 | 0% |
| Gate Type | ~4 | 0 | 0% |
| Evidence Strength | ~3 | 0 | 0% |
| Semantic Version | ~15 | 0 | 0% |

**Target:** 100% migration for consistency and maintainability.

---

## Best Practices

### When to Use Common Schemas

✅ **DO use** for:
- Identifiers (UUIDs, hashes, versions)
- Standard enums (principal types, environments)
- Timestamps and dates
- Metadata objects
- Hash chains and Merkle paths

❌ **DON'T use** for:
- Domain-specific enums (model types, artifact types)
- Highly contextual patterns
- One-off fields unique to a single schema

### Naming Conventions

- **Identifiers:** Noun describing what it identifies (uuid, sha256-hash)
- **Enums:** Singular noun + -type suffix (principal-type, decision-type)
- **Patterns:** Descriptive noun (timestamp, metadata, merkle-path)

### Documentation Requirements

Every common schema must include:
1. Clear `title` and `description`
2. Practical `examples`
3. `$comment` explaining usage context
4. Reference to which schemas use it

---

## Roadmap

### Phase 1: Foundation (✅ Complete)
- [x] Create directory structure
- [x] Define 15 common schemas
- [x] Document usage patterns
- [x] Create migration guide

### Phase 2: Migration (Next)
- [ ] Migrate receipt schemas to use common components
- [ ] Migrate gate schemas
- [ ] Migrate policy schemas
- [ ] Migrate lifecycle schemas

### Phase 3: Extension (Future)
- [ ] Add domain-specific common patterns
- [ ] Create composite patterns (receipt-base, anchor-base)
- [ ] Add validation utilities
- [ ] Generate TypeScript types from common schemas

---

## References

- **Signature Infrastructure:** [signature-envelope.json](signature-envelope.json), [signature-metadata.json](signature-metadata.json)
- **JSON Schema Specification:** [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/schema)
- **Schema Best Practices:** [docs/CODING_STANDARDS.md](../../../docs/CODING_STANDARDS.md)

---

**Created:** March 30, 2026  
**Status:** Foundation complete, migration pending  
**Maintainer:** CIAF Schema Team
