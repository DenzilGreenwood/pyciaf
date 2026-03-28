# CIAF v1.2.0 Developer Integration Guide

**Quick Start**: Adding Forensic Fragments to Your Watermarking Workflow  
**Framework**: CIAF (Cognitive Insight Audit Framework)  
**Version**: 1.2.0  

---

## Quick Reference: Text Example

### Before (v1.0)
```python
from ciaf.watermarks import build_text_artifact_evidence, verify_text_artifact

# Create artifact
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI-generated content",
    model_id="gpt-4",
    model_version="2026.Q1",
    actor_id="user:analyst-1"
)

# Verify suspect
result = verify_text_artifact(suspect_text, evidence)
print(f"Authentic: {result.is_authentic()}")
```

### After (v1.2 with Forensic Fragments) ⭐
```python
from ciaf.watermarks import (
    build_text_artifact_evidence,
    verify_text_artifact,
    verify_text_fragments  # ⭐ NEW
)

# Create artifact WITH forensic fragments
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI-generated content",
    model_id="gpt-4",
    model_version="2026.Q1",
    actor_id="user:analyst-1",
    enable_forensic_fragments=True  # ⭐ NEW
)

# Verify with traditional verification
result = verify_text_artifact(suspect_text, evidence)

# ⭐ NEW: Also verify forensic fragments
if evidence.hashes.forensic_fragments:
    frag_result = verify_text_fragments(
        suspect_text,
        evidence.hashes.forensic_fragments.text_fragments
    )
    print(f"Fragments matched: {frag_result.fragments_matched}/3")
    print(f"Legal defensibility: {frag_result.legal_defensibility}")
```

---

## Step-by-Step Integration

### 1. Creating Watermarked Artifacts with Fragments

#### Text
```python
from ciaf.watermarks import build_text_artifact_evidence

evidence, watermarked_text = build_text_artifact_evidence(
    raw_text="The financial analysis reveals strong quarter-over-quarter growth...",
    model_id="financial-analyzer",
    model_version="2.1",
    actor_id="user:analyst-42",
    prompt="Analyze Q1 financial performance",
    verification_base_url="https://vault.cognitiveinsight.ai",
    enable_forensic_fragments=True  # ⭐ Enable DNA sampling
)

# Access forensic fragments
print(f"Fragments: {evidence.hashes.forensic_fragments.fragment_count}")
print(f"Fragments matched: {evidence.hashes.forensic_fragments.text_fragments}")
for frag in evidence.hashes.forensic_fragments.text_fragments:
    print(f"  - {frag.sample_location} ({frag.offset_start}-{frag.offset_end}): "
          f"entropy={frag.entropy_score:.2f}")
```

#### Image
```python
from ciaf.watermarks import build_image_artifact_evidence, ImageWatermarkSpec

with open("generated_landscape.png", "rb") as f:
    image_bytes = f.read()

spec = ImageWatermarkSpec(
    mode="visual",
    text="AI Generated",
    opacity=0.4
)

evidence, watermarked_bytes = build_image_artifact_evidence(
    image_bytes=image_bytes,
    model_id="image-gen-xl",
    model_version="1.5",
    actor_id="user:artist-7",
    prompt="Create landscape painting",
    watermark_spec=spec,
    enable_forensic_fragments=True  # ⭐ Enable DNA sampling
)

# Access fragments
print(f"Image patches: {len(evidence.hashes.forensic_fragments.image_fragments)}")
for patch in evidence.hashes.forensic_fragments.image_fragments:
    x, y, w, h = patch.region_coordinates
    print(f"  - Patch {patch.patch_grid_position} at ({x},{y}): "
          f"entropy={patch.entropy_score:.2f}")
```

### 2. Storing Evidence in Vault

```python
from ciaf.watermarks import create_watermark_vault

vault = create_watermark_vault(storage_path="./watermarks")

# Store evidence (includes forensic fragments automatically)
vault.store_evidence(evidence)
print(f"Stored artifact {evidence.artifact_id}")

# Retrieve later
retrieved = vault.retrieve_evidence(evidence.artifact_id)
print(f"Retrieved with {retrieved.hashes.forensic_fragments.fragment_count} fragments")
```

### 3. Verifying Suspect Content

#### Text Fragment Verification
```python
from ciaf.watermarks import verify_text_fragments

suspect_text = "The financial analysis reveals strong quarter..."  # Possibly edited

# Verify fragments
result = verify_text_fragments(
    suspect_text=suspect_text,
    stored_fragments=evidence.hashes.forensic_fragments.text_fragments
)

print(f"\nForensic Verification:")
print(f"  Fragments matched: {result.fragments_matched}/{result.total_fragments_checked}")
print(f"  Legal defensibility: {result.legal_defensibility}")
print(f"  Match confidence: {result.match_confidence:.1%}")

# Show details
for match in result.forensic_matches:
    status = "✓ FOUND" if match.matched else "✗ NOT FOUND"
    print(f"  {status}: {match.fragment_id} @ {match.match_position}")
    
# Legal threshold check
if result.fragments_matched >= 2:
    print("\n✓ Suitable for legal/regulatory proceedings (P < 10^-15)")
```

#### Image Fragment Verification
```python
from ciaf.watermarks import verify_image_fragments

with open("suspect_image.png", "rb") as f:
    suspect_bytes = f.read()

result = verify_image_fragments(
    suspect_image_bytes=suspect_bytes,
    stored_fragments=evidence.hashes.forensic_fragments.image_fragments
)

print(f"\nImage Forensics:")
print(f"  Patches matched: {result.fragments_matched}/{result.total_fragments_checked}")
print(f"  Spatial integrity: {result.legal_defensibility}")

if result.fragments_matched == result.total_fragments_checked:
    print("✓ Image appears unmodified")
elif result.fragments_matched >= 2:
    print("✓ Majority of image patches authentic")
elif result.fragments_matched == 1:
    print("⚠ Limited forensic evidence (1 patch)")
else:
    print("✗ Image likely heavily modified or fake")
```

### 4. Advanced: Manual Fragment Selection

```python
from ciaf.watermarks import (
    select_text_forensic_fragments,
    select_image_forensic_patches,
    create_forensic_fragment_set
)

# Custom text fragment selection
text_fragments = select_text_forensic_fragments(
    raw_text=raw_text,
    fragment_hash_before="hash_before",  # Computed from raw text
    fragment_hash_after="hash_after",    # Computed from watermarked text
    min_entropy=0.5  # Higher threshold = more selective
)

# Custom image fragment selection
image_patches = select_image_forensic_patches(
    image_bytes=image_bytes,
    num_patches=6,        # Select 6 patches instead of 4
    patch_size=64,        # 64x64 pixel patches
    min_entropy=0.6       # High complexity only
)

# Create combined fragment set
fragment_set = create_forensic_fragment_set(
    artifact=raw_text,
    artifact_type="text",
    enable_fragments=True
)
```

---

## Key API Reference

### Fragment Selection

```python
# Text entropy scoring
entropy = compute_text_entropy(text)  # Returns 0.0-1.0

# Text fragment selection (3-point sampling)
fragments = select_text_forensic_fragments(
    raw_text,
    fragment_hash_before,
    fragment_hash_after,
    min_entropy=0.4
)

# Image patch selection (spatial)
patches = select_image_forensic_patches(
    image_bytes,
    num_patches=4,
    patch_size=64,
    min_entropy=0.5
)

# Unified fragment set creation
fragment_set = create_forensic_fragment_set(
    artifact,
    artifact_type="text"|"image"|"video"|"audio",
    enable_fragments=True
)
```

### Fragment Verification

```python
# Text fragment verification
text_result = verify_text_fragments(
    suspect_text,
    stored_fragments
)
# Returns: FragmentMatchResult[] with search results

# Image fragment verification
image_result = verify_image_fragments(
    suspect_image_bytes,
    stored_fragments
)
# Returns: FragmentMatchResult[] with spatial results

# Access results
text_result.fragments_matched        # Count of matches
text_result.total_fragments_checked  # Total tested
text_result.legal_defensibility      # 'high'|'medium'|'low'
text_result.match_confidence         # 0.0-1.0
text_result.forensic_matches[]       # Detailed results
text_result.notes[]                  # Human-readable findings
```

---

## Common Workflows

### Workflow 1: Audit Trail with Fragments
```python
# Generate and store
evidence, watermarked = build_text_artifact_evidence(
    raw_text=document,
    model_id="governance-model",
    actor_id=user_id,
    enable_forensic_fragments=True
)
vault.store_evidence(evidence)

# Later: Verify for compliance
retrieved = vault.retrieve_evidence(artifact_id)
result = verify_text_fragments(suspect_text, 
                               retrieved.hashes.forensic_fragments.text_fragments)

# Report findings
if result.legal_defensibility == "high":
    print("✓ Compliant: Fragments confirm AI origin")
```

### Workflow 2: Forensic Investigation
```python
# Search vault for any matching fragments
all_artifacts = vault.search_evidence(model_id=model_id)

for evidence in all_artifacts:
    result = verify_text_fragments(suspect_text,
                                   evidence.hashes.forensic_fragments.text_fragments)
    if result.fragments_matched >= 2:
        print(f"✓ Match found: artifact {evidence.artifact_id}")
```

### Workflow 3: Batch Processing
```python
# Create fragments for 1000s of artifacts
artifacts = []
for item in large_dataset:
    evidence, watermarked = build_text_artifact_evidence(
        raw_text=item['content'],
        model_id="batch-model",
        enable_forensic_fragments=True
    )
    artifacts.append((evidence, watermarked))
    
    if len(artifacts) >= 100:
        vault.store_many(artifacts)
        artifacts = []
```

---

## Error Handling

```python
from ciaf.watermarks import (
    verify_text_fragments,
    ForensicVerificationSummary
)

def safe_verify_fragments(suspect_text, evidence):
    """Safely verify fragments with error handling."""
    try:
        if not evidence.hashes.forensic_fragments:
            return None
            
        result = verify_text_fragments(
            suspect_text,
            evidence.hashes.forensic_fragments.text_fragments
        )
        
        return result
        
    except Exception as e:
        print(f"Fragment verification failed: {e}")
        return None

# Usage
result = safe_verify_fragments(suspect_text, evidence)
if result and result.legal_defensibility == "high":
    # Act on high-confidence match
```

---

## Data Model: Quick Reference

```python
# Main containers
ArtifactHashSet
  ├── forensic_fragments: ForensicFragmentSet
      ├── text_fragments: List[TextForensicFragment]
      ├── image_fragments: List[ImageForensicFragment]
      ├── video_snippets: List[VideoForensicSnippet]
      └── audio_segments: List[AudioForensicSegment]

# Text fragment structure
TextForensicFragment
  ├── fragment_id: str
  ├── offset_start: int
  ├── offset_end: int
  ├── entropy_score: float (0.0-1.0)
  ├── sample_location: str ('beginning'|'middle'|'end')
  ├── fragment_hash_before: str
  ├── fragment_hash_after: str
  └── fragment_simhash_before: Optional[str]

# Image fragment structure
ImageForensicFragment
  ├── fragment_id: str
  ├── region_coordinates: (x, y, w, h)
  ├── entropy_score: float (0.0-1.0)
  ├── patch_hash_before: str (pHash)
  ├── patch_hash_after: str (pHash)
  ├── patch_ahash_before: Optional[str]
  ├── patch_dhash_before: Optional[str]
  └── patch_whash_before: Optional[str]
```

---

## Migration Checklist

- [ ] Update `ciaf.watermarks` import statements
- [ ] Set `enable_forensic_fragments=True` in `build_*_artifact_evidence()` calls
- [ ] Update verification to use `verify_*_fragments()` for DNA-level verification
- [ ] Update vault queries to filter by `model_id`, `actor_id`, etc.
- [ ] Add forensic fragment verification to compliance workflows
- [ ] Test with existing artifacts (backward compatible)
- [ ] Deploy to staging environment
- [ ] Train team on new forensic verification workflows
- [ ] Update documentation and runbooks

---

## Troubleshooting

### Issue: No fragments created
```python
# Solution: Check enable_forensic_fragments flag
evidence, _ = build_text_artifact_evidence(
    raw_text=text,
    model_id="model",
    enable_forensic_fragments=True  # Must be True
)
assert evidence.hashes.forensic_fragments is not None
```

### Issue: Fragment entropy too low
```python
# Solution: Lower min_entropy threshold
fragments = select_text_forensic_fragments(
    raw_text=text,
    fragment_hash_before=hash_b,
    fragment_hash_after=hash_a,
    min_entropy=0.3  # Lower threshold
)
```

### Issue: Image patches not found
```python
# Solution: Adjust patch size and count
patches = select_image_forensic_patches(
    image_bytes=img,
    num_patches=6,      # Try more patches
    patch_size=32,      # Try smaller patches
    min_entropy=0.4     # Lower entropy threshold
)
```

---

## Performance Tips

- **Text verification**: Sliding window is fast (~50ms for 100KB)
- **Image verification**: Spatial search can be slow for large images
  - Use smaller patches for faster searching
  - Consider parallel patch matching
- **Batch operations**: Store 100s at a time
- **Fragment selection**: Pre-compute and cache entropy scores

---

## Next Steps

1. **Review** the [Forensic Fragments DNA Sampling Guide](../docs/FORENSIC_FRAGMENTS_DNA_SAMPLING_GUIDE.md)
2. **Test** with sample artifacts
3. **Integrate** into your watermarking pipeline
4. **Deploy** to production
5. **Monitor** forensic verification metrics

---

**Framework**: CIAF v1.2.0  
**Last Updated**: March 28, 2026  
**Support**: See CIAF documentation at https://github.com/DenzilGreenwood/pyciaf
