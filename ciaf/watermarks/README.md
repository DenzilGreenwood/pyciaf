# CIAF Watermarking - Forensic Provenance for AI Artifacts

**Version**: 1.0.0
**Created**: 2026-03-24
**Status**: Production-Ready (Text, Images, PDF Metadata) - ✅ Phase 1 Complete

## Overview

The CIAF Watermarking module implements a **dual-state artifact integrity model** for forensic provenance of AI-generated content. It enables detection of:
- Watermark removal attempts
- Content tampering
- Unauthorized modifications
- AI-generated content verification

### Key Innovation: Before/After Hashing

Unlike traditional watermarking, CIAF stores **two cryptographic hashes**:
1. **Before watermark**: Hash of original AI output
2. **After watermark**: Hash of distributed version with provenance tag

This enables forensic detection even when watermarks are removed!

## Architecture

```
ciaf/watermarks/
├── __init__.py              # Main exports
├── models.py                # Data models (ArtifactEvidence, etc.)
├── hashing.py               # Hashing strategies (exact, normalized, SimHash)
├── text.py                  # Text watermarking implementation
├── verify.py                # Verification and forensic matching
├── vault_adapter.py         # Storage integration with CIAF vault
├── images/                  # Image watermarking package ✅ Phase 1
│   ├── __init__.py
│   ├── visual.py            # Visual watermarks (text + QR overlays)
│   ├── fingerprints.py      # Perceptual hashing (pHash, aHash, dHash, wHash)
│   └── qr.py                # QR code generation
└── pdf/                     # PDF watermarking package ✅ Phase 1
    ├── __init__.py
    └── metadata.py          # PDF metadata watermarking
```

## Features

### ✅ Implemented - Phase 1 Complete

**Text Watermarking**:
- Dual-state hashing (pre/post watermark)
- Multiple verification strategies:
  - Exact hash matching (cryptographic proof)
  - Normalized hash matching (format-resilient)
  - SimHash similarity (content-resilient)
- Watermark styles: Footer, Header, Inline
- Watermark removal detection
- Content modification detection
- Vault integration (persistent storage)

**Image Watermarking**:
- Visual text overlays (9 positions, opacity control)
- QR code overlays (verification URLs)
- Combined text + QR watermarks
- Perceptual hashing (pHash, aHash, dHash, wHash)
- Image similarity detection (hamming distance)
- Dual-state hashing for removal detection
- Support for PNG, JPEG, and other PIL formats

**PDF Watermarking**:
- Metadata embedding (invisible watermarks)
- Custom CIAF fields in PDF metadata
- Watermark extraction and verification
- Dual-state hashing for removal detection
- Preserves original content exactly

**QR Code Generation**:
- Verification URL QR codes
- Compact CIAF token QR codes
- Customizable styling (size, colors)

### 🚧 Planned - Phase 2

- PDF visible stamps (header/footer)
- QR code placement in PDF pages
- Hybrid image watermarking (visible + steganographic)
- Image verification helpers
- Batch watermarking operations

### 🔮 Future - Phase 3

- Steganographic image watermarking (LSB embedding)
- Video watermarking (frame-based)
- Audio watermarking


## Quick Start

### Basic Text Watermarking

```python
from ciaf.watermarks import build_text_artifact_evidence

# Generate watermarked artifact with full evidence
evidence, watermarked_text = build_text_artifact_evidence(
    raw_text="The quarterly risk summary indicates elevated model drift.",
    model_id="gpt-governed-prod",
    model_version="2026.03",
    actor_id="user:analyst-17",
    prompt="Summarize the model risk findings",
    verification_base_url="https://vault.cognitiveinsight.ai"
)

print(f"Artifact ID: {evidence.artifact_id}")
print(f"Watermark ID: {evidence.watermark.watermark_id}")
print(f"\nWatermarked:\n{watermarked_text}")
```

Output:
```
The quarterly risk summary indicates elevated model drift.

---
AI Provenance Tag: wmk-a1b2c3d4-e5f6-7890-ab12-cd34ef567890
Verify: https://vault.cognitiveinsight.ai/verify/7d6d5c0b-...
Generated with CIAF (Cognitive Insight Audit Framework)
```

### Quick Watermarking (Simplified)

```python
from ciaf.watermarks import quick_watermark_text

watermarked, artifact_id = quick_watermark_text(
    text="AI generated content",
    model_id="my-model"
)
```

### Image Watermarking

```python
from ciaf.watermarks import build_image_artifact_evidence, ImageWatermarkSpec

# Read image
with open("generated_image.png", "rb") as f:
    image_bytes = f.read()

# Configure watermark
spec = ImageWatermarkSpec(
    mode="visual",
    text="AI Generated",
    opacity=0.4,
    position="bottom_right",
    include_qr=True,  # Add QR code
    qr_position="top_right",
)

# Create watermarked image
evidence, watermarked = build_image_artifact_evidence(
    image_bytes=image_bytes,
    model_id="stable-diffusion-v3",
    model_version="2026.03",
    actor_id="user:artist-42",
    prompt="A futuristic cityscape at sunset",
    verification_base_url="https://vault.example.com",
    watermark_spec=spec,
    include_perceptual_hashes=True,  # Enable similarity detection
)

# Save watermarked image
with open("watermarked_image.png", "wb") as f:
    f.write(watermarked)

print(f"Artifact ID: {evidence.artifact_id}")
print(f"Perceptual hash: {evidence.hashes.perceptual_hash_before}")
```

### PDF Metadata Watermarking

```python
from ciaf.watermarks import build_pdf_artifact_evidence

# Read PDF
with open("report.pdf", "rb") as f:
    pdf_bytes = f.read()

# Create watermarked PDF (invisible metadata)
evidence, watermarked_pdf = build_pdf_artifact_evidence(
    pdf_bytes=pdf_bytes,
    model_id="gpt-report-gen",
    model_version="4.0",
    actor_id="system:report-bot",
    prompt="Generate Q4 risk analysis report",
    verification_base_url="https://vault.example.com",
    additional_metadata={
        "Department": "Risk Management",
        "Classification": "Confidential",
    }
)

# Save watermarked PDF
with open("watermarked_report.pdf", "wb") as f:
    f.write(watermarked_pdf)

print(f"Watermark in metadata: {evidence.watermark.watermark_id}")
```

### Verifying Suspect Artifacts

```python
from ciaf.watermarks import verify_text_artifact, format_verification_report

# Suspect text received from unknown source
suspect_text = "..."  # Text to verify

# Verify against stored evidence
result = verify_text_artifact(suspect_text, evidence)

# Check results
if result.is_authentic():
    print(f"✓ AUTHENTIC (confidence: {result.confidence:.1%})")
else:
    print("✗ NOT VERIFIED")

# Detailed report
print(format_verification_report(result))
```

## Forensic Detection Capabilities

### Scenario 1: Exact Match (Watermarked)

```python
# Someone shares the exact watermarked version
result = verify_text_artifact(watermarked_text, evidence)

assert result.exact_match_after_watermark == True
assert result.confidence == 1.0
# ✓ Perfect match - this is the distributed version
```

### Scenario 2: Watermark Removed

```python
# Someone removed the watermark but kept the content
from ciaf.watermarks import remove_watermark

stripped_text = remove_watermark(watermarked_text)
result = verify_text_artifact(stripped_text, evidence)

assert result.exact_match_before_watermark == True
assert result.likely_tag_removed == True
# ⚠ Watermark removed - but content is authentic!
```

### Scenario 3: Content Modified

```python
# Someone changed the content slightly
modified_text = watermarked_text.replace("elevated", "reduced")
result = verify_text_artifact(modified_text, evidence)

# May still detect via SimHash similarity
if result.perceptual_similarity_score and result.perceptual_similarity_score > 0.8:
    print("⚠ Content appears modified from original")
    # Content modified but likely derived from our output
```

### Scenario 4: Completely Unrelated

```python
# Random text not from our system
fake_text = "This is completely unrelated content."
result = verify_text_artifact(fake_text, evidence)

assert result.exact_match_before_watermark == False
assert result.exact_match_after_watermark == False
assert result.confidence < 0.5
# ✗ Not from our system
```

## Hashing Strategies

### 1. Exact Hashing (SHA-256)

**Use**: Cryptographic proof of identity

```python
from ciaf.watermarks import sha256_text

hash1 = sha256_text("Exact text")
hash2 = sha256_text("Exact text")
hash3 = sha256_text("Different text")

assert hash1 == hash2  # Exact match
assert hash1 != hash3  # Different content
```

### 2. Normalized Hashing

**Use**: Resilient to formatting changes (whitespace, casing)

```python
from ciaf.watermarks import normalized_text_hash

text1 = "This is some text."
text2 = "  THIS   IS    SOME   TEXT.  "  # Different formatting

hash1 = normalized_text_hash(text1)
hash2 = normalized_text_hash(text2)

assert hash1 == hash2  # Same content despite formatting
```

### 3. SimHash (Similarity)

**Use**: Detect minor rewording or edits

```python
from ciaf.watermarks import simhash_text, simhash_distance

text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The fast brown fox leaps over the sleepy dog."

hash1 = simhash_text(text1)
hash2 = simhash_text(text2)
distance = simhash_distance(hash1, hash2)

print(f"Distance: {distance}/64")  # Low distance = similar content
```

**SimHash Distance Thresholds**:
- `0-3`: Near duplicates
- `4-10`: Similar content
- `11-20`: Somewhat related
- `>20`: Different content

## Vault Integration

### Storing Watermark Evidence

```python
from ciaf.watermarks import create_watermark_vault

# Create vault adapter
vault = create_watermark_vault(storage_path="./watermark_vault")

# Store evidence
success = vault.store_evidence(evidence)

# Retrieve by artifact ID
retrieved = vault.retrieve_evidence(evidence.artifact_id)

# Search by model
artifacts = vault.search_by_model("gpt-governed-prod")

# Search by watermark ID
found = vault.search_by_watermark(watermark_id)
```

### Integration with CIAF Vault

```python
from ciaf.vault import MetadataStorage
from ciaf.watermarks import create_watermark_vault

# Use existing CIAF vault
vault_storage = MetadataStorage(backend="postgresql", postgresql_config={...})
watermark_vault = create_watermark_vault(vault_storage=vault_storage)

# Now watermark evidence is stored in PostgreSQL
watermark_vault.store_evidence(evidence)
```

## Data Models

### ArtifactEvidence

Complete provenance record for AI artifact:

```python
{
    "artifact_id": "7d6d5c0b-...",
    "artifact_type": "text",  # text, image, pdf, json, binary
    "model_id": "gpt-governed-prod",
    "model_version": "2026.03",
    "actor_id": "user:analyst-17",
    "prompt_hash": "abc123...",  # SHA-256 of input
    "output_hash_raw": "def456...",  # Pre-watermark hash
    "output_hash_distributed": "ghi789...",  # Post-watermark hash
    "watermark": {
        "watermark_id": "wmk-...",
        "watermark_type": "visible",
        "verification_url": "https://...",
        "removal_resistance": "low"
    },
    "hashes": {
        "content_hash_before_watermark": "...",
        "content_hash_after_watermark": "...",
        "normalized_hash_before": "...",
        "simhash_before": "..."
    },
    "fingerprints": [
        {"algorithm": "simhash", "value": "...", "role": "similarity"}
    ]
}
```

### VerificationResult

Forensic analysis results:

```python
{
    "artifact_id": "...",
    "exact_match_after_watermark": True,  # Matches distributed version
    "exact_match_before_watermark": False,
    "likely_tag_removed": False,
    "watermark_present": True,
    "watermark_intact": True,
    "confidence": 1.0,  # 0.0-1.0
    "notes": [
        "✓ Exact match to distributed watermarked version.",
        "✓ Original watermark present and intact."
    ]
}
```

## Verification Workflow

### Complete Verification Pipeline

```python
from ciaf.watermarks import (
    build_text_artifact_evidence,
    verify_text_artifact,
    analyze_suspect_text,
    create_watermark_vault
)

# Step 1: Generate and store watermarked content
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI output",
    model_id="production-model",
    model_version="1.0",
    actor_id="system",
    prompt="Generate content",
    verification_base_url="https://vault.example.com"
)

# Store in vault
vault = create_watermark_vault()
vault.store_evidence(evidence)

# Step 2: Someone sends you suspect text
suspect_text = "..."  # Received from external source

# Step 3: Quick analysis (no evidence needed)
analysis = analyze_suspect_text(suspect_text)
print(f"Has CIAF watermark: {analysis['has_ciaf_watermark']}")
print(f"Watermark ID: {analysis['watermark_id']}")

# Step 4: Full verification (with evidence)
if analysis['watermark_id']:
    # Find evidence by watermark
    evidence = vault.search_by_watermark(analysis['watermark_id'])

if evidence:
    result = verify_text_artifact(suspect_text, evidence)

    print(f"Authentic: {result.is_authentic()}")
    print(f"Confidence: {result.confidence:.1%}")

    if result.likely_tag_removed:
        print("⚠ WARNING: Watermark was removed!")

    if result.content_modified:
        print("⚠ WARNING: Content was modified!")
```

## Advanced Features

### Batch Verification

```python
from ciaf.watermarks import verify_against_multiple_evidence

# Check suspect text against all stored artifacts
results = verify_against_multiple_evidence(
    suspect_text=suspect_text,
    evidence_records=[evidence1, evidence2, evidence3, ...],
    min_confidence=0.8
)

# Results sorted by confidence (highest first)
for result in results:
    print(f"Match: {result.artifact_id} (confidence: {result.confidence:.1%})")
```

### Watermark Extraction

```python
from ciaf.watermarks import (
    extract_watermark_id,
    extract_verification_url,
    has_watermark
)

# Check if text has watermark
if has_watermark(text):
    watermark_id = extract_watermark_id(text)
    verify_url = extract_verification_url(text)

    print(f"Watermark: {watermark_id}")
    print(f"Verify at: {verify_url}")
```

### Custom Watermark Styles

```python
from ciaf.watermarks import apply_text_watermark

# Footer style (default)
watermarked = apply_text_watermark(
    raw_text=text,
    watermark_id=watermark_id,
    verification_url=url,
    style="footer"
)

# Header style
watermarked = apply_text_watermark(
    raw_text=text,
    watermark_id=watermark_id,
    verification_url=url,
    style="header"
)

# Inline style (end of first paragraph)
watermarked = apply_text_watermark(
    raw_text=text,
    watermark_id=watermark_id,
    verification_url=url,
    style="inline"
)
```

## Best Practices

### 1. Always Store Evidence

```python
# GOOD: Store evidence immediately
evidence, watermarked = build_text_artifact_evidence(...)
vault.store_evidence(evidence)

# BAD: Lose the evidence
watermarked, artifact_id = quick_watermark_text(...)
# Evidence not stored - can't verify later!
```

### 2. Use Multiple Verification Strategies

```python
result = verify_text_artifact(suspect, evidence)

# Check all indicators
if result.exact_match_after_watermark:
    # Perfect cryptographic match
elif result.exact_match_before_watermark:
    # Watermark removed
elif result.perceptual_similarity_score > 0.9:
    # Content modified but likely derived
else:
    # Probably not from our system
```

### 3. Include SimHash for Robustness

```python
evidence, watermarked = build_text_artifact_evidence(
    ...,
    include_simhash=True  # Enables similarity detection
)
```

### 4. Verification URL Configuration

```python
# Use your actual vault URL
verification_base_url = "https://vault.yourcompany.com"

# URLs will be:
# https://vault.yourcompany.com/verify/{artifact_id}
```

## Schema Validation

Watermark artifacts follow the JSON schema at `ciaf/schemas/vault.schema.json`:

```python
import json
from pathlib import Path

# Load schema
schema_path = Path("ciaf/schemas/vault.schema.json")
with open(schema_path) as f:
    vault_schema = json.load(f)

# Validate artifact
import jsonschema
jsonschema.validate(
    instance=evidence.to_dict(),
    schema=vault_schema['definitions']['watermarked_artifact']
)
```

## Testing

Run comprehensive tests:

```bash
# Text watermarking tests
python tests/test_watermarks.py

# Phase 1 tests (images + PDF)
python tests/test_watermarks_phase1.py
```

**Text Tests** cover:
- Text watermarking
- Watermark removal detection
- Similarity matching (SimHash)
- Normalized hashing
- Vault storage integration
- Verification reporting
- Suspect artifact analysis

**Phase 1 Tests** cover:
- Image visual watermarking (text + QR)
- Image perceptual hashing (pHash, aHash, dHash, wHash)
- QR code generation
- Combined watermarks
- PDF metadata watermarking
- PDF watermark removal detection

**Test Results**:
- Text: 7/7 passing ✅
- Phase 1: 5/5 core tests passing ✅ (2 optional tests require imagehash)

## Performance

### Hashing Performance

```python
# SHA-256: ~1M hashes/sec
# Normalized hash: ~500K/sec
# SimHash: ~100K/sec
# Perceptual hash (pHash): ~40 images/sec
# All 4 perceptual hashes: ~10 images/sec
```

### Watermarking Performance

```python
# Text watermarking: ~10K artifacts/sec
# Text verification: ~5K checks/sec
# Image text overlay: ~20 images/sec
# Image text + QR: ~12 images/sec
# PDF metadata: ~50 PDFs/sec
```

### Vault Performance

See `ciaf/vault/README.md` for vault performance characteristics.

## Security Considerations

### Watermark Limitations

1. **Visual watermarks are easy to remove** (low removal resistance)
   - Solution: Dual-state hashing detects removal
   - Future: Steganographic watermarking (Phase 3)

2. **Text paraphrasing defeats exact hashing**
   - Solution: SimHash provides similarity detection
   - Future: Embedding-based similarity

3. **Image cropping can remove watermarks**
   - Solution: Perceptual hashing survives many modifications
   - Dual-state hashing detects removal

4. **PDF metadata can be stripped**
   - Medium resistance (requires tools)
   - Solution: Dual-state hashing detects stripping
   - Future: Combined metadata + visible stamps (Phase 2)

### Best Security Practices

1. **Store evidence securely** (encrypted PostgreSQL vault)
2. **Use HTTPS for verification URLs**
3. **Log all verification attempts**
4. **Implement rate limiting** on verification endpoints
5. **Sign evidence records** (cryptographic integrity)

## Future Enhancements

### Phase 1 (Complete) ✅
- [x] Image visual watermarking (text + QR overlays)
- [x] Image perceptual hashing (pHash, aHash, dHash, wHash)
- [x] QR code generation
- [x] PDF metadata watermarking
- [x] Dual-state hashing for all artifact types

### Phase 2 (Planned)
- [ ] PDF visible stamps (header/footer)
- [ ] QR code placement in PDF pages
- [ ] Hybrid image watermarking (visible + steganographic)
- [ ] Image verification helpers
- [ ] Batch watermarking operations
- [ ] REST API for verification

### Phase 3 (Future)
- [ ] Steganographic image watermarking (LSB embedding)
- [ ] Video watermarking (frame-based)
- [ ] Audio watermarking
- [ ] Blockchain anchoring (immutable audit trail)
- [ ] Zero-knowledge proofs
- [ ] Embedding-based similarity (neural networks)
- [ ] Federated verification (cross-organization)

## Examples

See `tests/test_watermarks.py` for comprehensive examples.

### Minimal Example

```python
from ciaf.watermarks import build_text_artifact_evidence, verify_text_artifact

# Create watermarked artifact
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI content",
    model_id="model-1",
    model_version="1.0",
    actor_id="user-1",
    prompt="Generate",
    verification_base_url="https://vault.example.com"
)

# Verify
result = verify_text_artifact(watermarked, evidence)
print(f"Authentic: {result.is_authentic()}")
```

## Dependencies

### Required
- **Python 3.8+**
- **Pillow** (PIL) - Image manipulation
  ```bash
  pip install Pillow
  ```
- **qrcode** - QR code generation
  ```bash
  pip install qrcode[pil]
  ```
- **pypdf** or **PyPDF2** - PDF manipulation
  ```bash
  pip install pypdf
  ```

### Optional
- **imagehash** - Perceptual image hashing (highly recommended for image similarity)
  ```bash
  pip install imagehash
  ```
  Without imagehash, perceptual hashing tests will be skipped but all other functionality works.

### Install All Dependencies
```bash
# Required + optional
pip install Pillow qrcode[pil] pypdf imagehash

# Or from project requirements
pip install -r requirements.txt
```

## Troubleshooting

### Issue: PIL/Pillow not available

```bash
pip install Pillow
```

### Issue: imagehash not available

```python
# Check if available
from ciaf.watermarks import IMAGEHASH_AVAILABLE
print(f"Perceptual hashing available: {IMAGEHASH_AVAILABLE}")

# Install if needed
# pip install imagehash
```

### Issue: ModuleNotFoundError

```bash
pip install -e .  # Install CIAF in development mode
```

### Issue: Watermark not detected

```python
# Check watermark presence
from ciaf.watermarks import has_watermark, extract_watermark_id

if not has_watermark(text):
    print("No CIAF watermark found")
else:
    print(f"Watermark: {extract_watermark_id(text)}")
```

### Issue: Verification confidence low

```python
# Enable detailed analysis
result = verify_text_artifact(
    suspect_text,
    evidence,
    check_normalized=True,  # Enable format-resilient matching
    check_simhash=True,  # Enable similarity detection
    simhash_threshold=15  # More lenient threshold
)

# Check individual signals
print(f"Exact match: {result.exact_match_after_watermark}")
print(f"Normalized match: {result.normalized_match_before}")
print(f"Similarity: {result.perceptual_similarity_score}")
```

## API Reference

See docstrings in each module for detailed API documentation:

**Core Modules**:
- `models.py` - Data structures (ArtifactEvidence, WatermarkDescriptor, etc.)
- `hashing.py` - Hashing utilities (SHA-256, normalized, SimHash)
- `text.py` - Text watermarking
- `verify.py` - Verification logic
- `vault_adapter.py` - Storage integration

**Phase 1 Modules**:
- `images/visual.py` - Image visual watermarking (text + QR overlays)
- `images/fingerprints.py` - Perceptual hashing (pHash, aHash, dHash, wHash)
- `images/qr.py` - QR code generation
- `pdf/metadata.py` - PDF metadata watermarking

## Contributing

When adding new watermarking features:
1. Add data models to `models.py`
2. Implement hashing in `hashing.py`
3. Add watermarking logic (e.g., `video.py`)
4. Update verification in `verify.py`
5. Add tests to `tests/test_watermarks.py`
6. Update this README

## License

See main CIAF LICENSE file.

---

**Version**: 1.0.0
**Last Updated**: 2026-03-24
**Author**: Denzil James Greenwood
**Contact**: founder@cognitiveinsight.ai
