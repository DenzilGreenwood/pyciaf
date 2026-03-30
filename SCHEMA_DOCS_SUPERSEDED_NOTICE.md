# ⚠️ SUPERSEDED DOCUMENTATION

**Date:** March 30, 2026

---

## Notice

This documentation has been **SUPERSEDED** by the consolidated single source of truth:

### 📘 [CIAF_SCHEMA_SPECIFICATION.md](CIAF_SCHEMA_SPECIFICATION.md)

The new specification consolidates:
- ✅ Cryptographic standards (SHA-256, Ed25519)
- ✅ Merkle batching policies
- ✅ Receipt metadata requirements
- ✅ Common schema architecture (18 reusable components)
- ✅ Production signature envelope structure
- ✅ Complete schema catalog (64 schemas)
- ✅ Implementation guidance
- ✅ Verification requirements

---

## Superseded Documents

The following files are **retained for historical reference only**:

1. **CIAF_COMPLETE_SCHEMA.md** - Schema catalog and structure
2. **COMMON_SCHEMA_CONSOLIDATION_ANALYSIS.md** - Common schema analysis
3. **SCHEMA_ENHANCEMENT_SUMMARY.md** - Enhancement history
4. **SCHEMA_MIGRATION_TO_SIGNATURE_ENVELOPE.md** - Signature migration
5. **SCHEMA_MIGRATION_QUICK_START.md** - Migration tools guide

---

## Migration Path

**For all new development:**
- ❌ Do NOT reference the superseded documents above
- ✅ Use **CIAF_SCHEMA_SPECIFICATION.md** as the authoritative source

**For schema questions:**
- Refer to Section 12 (Schema Catalog) in the specification
- Review Section 4 (Receipt Metadata Requirements)
- Check Section 13 (Implementation Guidance)

**For migration:**
- Tool documentation remains in `tools/README_TOOLS.md`
- Migration scripts remain functional in `tools/` directory

---

## Why Consolidation?

**Problem:** Multiple overlapping documentation files created confusion about:
- Which cryptographic algorithms to use
- How to structure receipts
- When to create Merkle batches
- Which schemas are authoritative

**Solution:** Single authoritative specification combining:
- Cryptographic standards policy
- Schema architecture documentation
- Merkle batching rules
- Implementation guidance

---

## Quick Links

- **Specification:** [CIAF_SCHEMA_SPECIFICATION.md](CIAF_SCHEMA_SPECIFICATION.md)
- **Schemas Directory:** [ciaf/schemas/](ciaf/schemas/)
- **Migration Tools:** [tools/](tools/)
- **Common Schemas:** [ciaf/schemas/common/](ciaf/schemas/common/)

---

**For questions or clarifications:**
- Open an issue on GitHub
- Review the specification changelog (Section 14)
- Check the compliance section for regulatory alignment

---

**Status:** 🔒 Read-Only Archive
**Authoritative Document:** CIAF_SCHEMA_SPECIFICATION.md v1.0.0
