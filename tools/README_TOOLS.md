# CIAF Schema Tools

This directory contains utility scripts for managing and validating CIAF JSON schemas.

---

## Available Tools

### 1. `migrate_to_common_schemas.py`

Automatically migrates existing schemas to use common reusable components.

**Purpose:**
- Refactors inline schema patterns to use `$ref` to common schemas
- Reduces duplication across schema files
- Ensures consistency in pattern usage

**Usage:**

```bash
# Dry run (show what would change)
python tools/migrate_to_common_schemas.py --dry-run

# Live migration with backup
python tools/migrate_to_common_schemas.py

# Live migration without backup
python tools/migrate_to_common_schemas.py --no-backup

# Generate migration report
python tools/migrate_to_common_schemas.py --report migration-report.txt
```

**What It Migrates:**

| Pattern | Replaces | With |
|---------|----------|------|
| UUID identifiers | `"type": "string", "format": "uuid"` | `"$ref": "common/identifiers/uuid.json"` |
| SHA-256 hashes | `"pattern": "^[a-f0-9]{64}$"` | `"$ref": "common/identifiers/sha256-hash.json"` |
| Timestamps | `"format": "date-time"` | `"$ref": "common/patterns/timestamp.json"` |
| Semantic versions | `"pattern": "^\d+\.\d+\.\d+$"` | `"$ref": "common/identifiers/semantic-version.json"` |
| Principal types | `"enum": ["agent", "human", ...]` | `"$ref": "common/enums/principal-type.json"` |
| Environment types | `"enum": ["production", "staging", ...]` | `"$ref": "common/enums/environment-type.json"` |
| Decision types | `"enum": ["allow", "deny", ...]` | `"$ref": "common/enums/decision-type.json"` |
| Gate types | `"enum": ["provenance", "validation", ...]` | `"$ref": "common/enums/gate-type.json"` |
| Evidence strength | `"enum": ["real", "simulated", "fallback"]` | `"$ref": "common/enums/evidence-strength.json"` |
| Hash algorithms | `"enum": ["sha256", "sha3-256", ...]` | `"$ref": "common/enums/hash-algorithm.json"` |
| Metadata objects | `"type": "object"` (field name: metadata) | `"$ref": "common/patterns/metadata.json"` |
| Merkle paths | Array of SHA-256 hashes | `"$ref": "common/patterns/merkle-path.json"` |
| Policy obligations | Array of strings | `"$ref": "common/patterns/policy-obligations.json"` |

**Example Output:**

```
CIAF Schema Migration Tool
Schemas Directory: d:/Github/UsefulStuf/Resume/base/pyciaf/ciaf/schemas

Found 46 schema files to process
Mode: LIVE MIGRATION
Backup: Enabled
------------------------------------------------------------
✓ ciaf/schemas/action-receipt.schema.json
✓ ciaf/schemas/inference-receipt-enhanced.schema.json
  ciaf/schemas/anchor.schema.json (no changes)
...

============================================================
MIGRATION REPORT
============================================================
Date: 2026-03-30 14:32:45
Mode: LIVE

Summary:
  Files Processed: 46
  Files Modified: 23
  Total Migrations: 23

Migrations Applied:
------------------------------------------------------------

ciaf/schemas/action-receipt.schema.json:
  • timestamp → Timestamp
  • principal_type → Principal Type
  • params_hash → SHA-256 Hash
  • prior_receipt_hash → SHA-256 Hash

ciaf/schemas/inference-receipt-enhanced.schema.json:
  • receipt_id → UUID
  • model_anchor → SHA-256 Hash
  • input_hash → SHA-256 Hash
  • output_hash → SHA-256 Hash
  • timestamp → Timestamp
  • evidence_strength → Evidence Strength
  • committed_at → Timestamp
  • merkle_path → Merkle Path

============================================================
```

**Safety Features:**
- ✅ Creates `.backup` files by default
- ✅ Preserves descriptions and defaults
- ✅ Skips files in `common/` directory
- ✅ Skips example files
- ✅ Validates JSON before writing

---

### 2. `validate_schemas.py`

Validates all CIAF schemas and checks for common issues.

**Purpose:**
- Validates JSON syntax
- Checks for required schema fields
- Detects anti-patterns
- Suggests improvements

**Usage:**

```bash
# Basic validation
python tools/validate_schemas.py

# Verbose output
python tools/validate_schemas.py --verbose
```

**What It Checks:**

**Errors (Critical):**
- ✗ Invalid JSON syntax
- ✗ File reading errors

**Warnings (Improvements):**
- ⚠ Missing `$schema` field
- ⚠ Missing `$id` field
- ⚠ Missing `title` or `description`
- ⚠ Wrong schema version (not 2020-12)
- ⚠ Properties missing descriptions
- ⚠ Inline patterns that should use common schemas

**Example Output:**

```
CIAF Schema Validation Tool
Schemas Directory: d:/Github/UsefulStuf/Resume/base/pyciaf/ciaf/schemas

Validating 64 schema files...
------------------------------------------------------------
✓ ciaf/schemas/action-receipt.schema.json
✓ ciaf/schemas/anchor.schema.json
✗ ciaf/schemas/old-schema.json

============================================================
SCHEMA VALIDATION REPORT
============================================================

Summary:
  Total Files: 64
  Valid: 63
  Invalid: 1
  Files with Warnings: 12

ERRORS:
------------------------------------------------------------

ciaf/schemas/old-schema.json:
  ✗ Invalid JSON: Expecting ',' delimiter: line 45 column 3

WARNINGS:
------------------------------------------------------------

ciaf/schemas/inference-receipt-enhanced.schema.json:
  ⚠ Property 'input_hash' uses inline SHA-256 pattern. Consider using common/identifiers/sha256-hash.json
  ⚠ Property 'timestamp' uses inline timestamp format. Consider using common/patterns/timestamp.json

ciaf/schemas/gate-definition.schema.json:
  ⚠ Property 'metadata' missing description

============================================================
✓ All schemas are valid!
============================================================
```

---

## Workflow

### Standard Schema Migration Workflow

```bash
# Step 1: Validate current schemas
python tools/validate_schemas.py

# Step 2: Dry run migration to see what will change
python tools/migrate_to_common_schemas.py --dry-run

# Step 3: Review the dry run output

# Step 4: Run actual migration (with backup)
python tools/migrate_to_common_schemas.py --report migration-report.txt

# Step 5: Validate migrated schemas
python tools/validate_schemas.py

# Step 6: Review and test
# - Check git diff to review changes
# - Run schema validation tests
# - Test with actual data
```

### Rollback Procedure

If migration causes issues:

```bash
# Restore from backups
find ciaf/schemas -name "*.json.backup" | while read backup; do
    original="${backup%.backup}"
    mv "$backup" "$original"
done
```

---

## Requirements

**Python 3.7+** with standard library only (no external dependencies)

Both tools use only built-in Python modules:
- `json` - JSON parsing
- `pathlib` - File path handling
- `re` - Regular expressions (optional)
- `shutil` - File operations
- `argparse` - Command-line arguments
- `datetime` - Timestamps

---

## Best Practices

### Before Migration

1. **Commit current state** to version control
2. **Run validation** to ensure schemas are valid
3. **Review migration plan** with dry run
4. **Backup critical schemas** manually if needed

### During Migration

1. **Use dry run first** to preview changes
2. **Keep backups enabled** (default)
3. **Generate migration report** for documentation
4. **Monitor output** for unexpected changes

### After Migration

1. **Validate migrated schemas** immediately
2. **Review git diff** to understand changes
3. **Test with actual data** to ensure compatibility
4. **Update documentation** if patterns changed
5. **Delete backup files** only after verification

---

## Common Issues

### Issue: Migration doesn't detect pattern

**Cause:** Pattern matching is exact. Variations in enum order or additional fields prevent matching.

**Solution:**
1. Check pattern definition in `migrate_to_common_schemas.py`
2. Manually verify the inline pattern matches exactly
3. If needed, add a new migration pattern

### Issue: Migrated schema validation fails

**Cause:** `$ref` resolution may fail if common schemas aren't accessible.

**Solution:**
1. Ensure common schemas exist in `ciaf/schemas/common/`
2. Check `$ref` paths are correct (relative to schema location)
3. Validate common schemas first

### Issue: Descriptions lost after migration

**Cause:** Bug in migration script (should not happen - descriptions are preserved).

**Solution:**
1. Check backup files
2. Restore original
3. Report bug with example

---

## Future Enhancements

Planned improvements:

- [ ] Add JSON Schema meta-schema validation using `jsonschema` library
- [ ] Generate TypeScript types from migrated schemas
- [ ] Create visualization of schema dependencies
- [ ] Add schema diff tool to compare versions
- [ ] Support batch rollback with single command
- [ ] Add pre-commit hooks for automatic validation
- [ ] Generate schema documentation from JSON Schema

---

## Contributing

When adding new common schemas:

1. Add the pattern to `migration_patterns` list in `migrate_to_common_schemas.py`
2. Update this README with the new pattern
3. Test migration with dry run
4. Validate results

---

## Support

For issues or questions:
- Check this README first
- Review error messages carefully
- Check backup files if migration failed
- Consult CIAF Schema Team

---

**Last Updated:** March 30, 2026  
**Maintainer:** CIAF Schema Team
