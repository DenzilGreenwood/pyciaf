# Dual-State Hashing Code Review

## Status: ✅ VERIFIED WORKING CORRECTLY

## Summary

The dual-state hashing implementation is **working correctly**. The before and after hash values are **different** as expected, and the watermarked output includes both the original AI response and the watermark metadata.

---

## Test Results

### 1. Hash Differentiation ✅

**Hash BEFORE watermark (original AI output):**
```
f19f7a23ac4d23e3db0f05d91e13553bc9f4b2a3c9a261fa76e2d14dde11f53c
```

**Hash AFTER watermark (with provenance tags):**
```
e1b5375f6b5dd0b390744f7ced4c98936b9434f969cb1190f3889788325b1233
```

**Result:** Hashes are **DIFFERENT** ✅

---

### 2. Original AI Output (Before Watermark)

```
The capital of France is Paris. Paris is known for its
art, culture, and the Eiffel Tower. It is one of the most visited cities
in the world.
```

**Hash:** `f19f7a23...` (stored in `content_hash_before_watermark`)

---

### 3. Watermarked Output (After Watermark)

```
The capital of France is Paris. Paris is known for its
art, culture, and the Eiffel Tower. It is one of the most visited cities
in the world.

---
AI Provenance Tag: wmk-7de372b3-047b-4419-89f6-1ecc791a1206
Verify: https://verify.example.com/verify/9ebc178a-ec62-49d5-90a7-9f8d73cdcf5b
Generated with CIAF (Cognitive Insight Audit Framework)
```

**Hash:** `e1b5375f...` (stored in `content_hash_after_watermark`)

---

## Code Flow Analysis

### File: `ciaf/watermarks/text.py`

#### Function: `build_text_artifact_evidence()` (Lines 166-294)

**Step 1: Apply Watermark** (Lines 202-207)
```python
watermarked_text = apply_text_watermark(
    raw_text=raw_text,              # ← Original AI output
    watermark_id=watermark_id,
    verification_url=verification_url,
    style=watermark_style,
)
```
✅ Creates watermarked version with provenance tags

**Step 2: Compute Dual Hashes** (Lines 210-212)
```python
prompt_hash = sha256_text(prompt)
hash_before = sha256_text(raw_text)           # ← Hash of ORIGINAL
hash_after = sha256_text(watermarked_text)    # ← Hash of WATERMARKED
```
✅ Two separate hashes computed correctly

**Step 3: Build Hash Set** (Lines 219-224)
```python
hash_set = ArtifactHashSet(
    content_hash_before_watermark=hash_before,   # ← Original hash
    content_hash_after_watermark=hash_after,     # ← Watermarked hash
    normalized_hash_before=norm_before,
    normalized_hash_after=norm_after,
)
```
✅ Hash set properly constructed

**Step 4: Store in Evidence** (Lines 273-288)
```python
evidence = ArtifactEvidence(
    # ...
    output_hash_raw=hash_before,              # ← Before watermark
    output_hash_distributed=hash_after,       # ← After watermark
    hashes=hash_set,                          # ← Complete hash set
    # ...
)
```
✅ Evidence record properly structured

---

## Additional Verification

### Normalized Hashes (Also Different) ✅
- **Before:** `d0fbd0f2a464593e73a349a7fefd0135f65175934ba391d7ee565f8f7ec58ab3`
- **After:** `95ccf15f699b02f5c881281dfbe8b095cf80c8a40462907b9ea71c2de77c8a2c`
- **Different:** ✅ YES

### SimHash Fingerprints (Also Different) ✅
- **Before:** `8ff42525478927d2`
- **After:** `cde00dadc52837fa`
- **Different:** ✅ YES

---

## Watermark Content Verification ✅

| Check | Status |
|-------|--------|
| Original AI response included in watermarked text | ✅ YES |
| Watermark tag present (`AI Provenance Tag:`) | ✅ YES |
| Verification URL present | ✅ YES |
| CIAF attribution present | ✅ YES |
| Model ID stored | ✅ YES |
| Prompt hash computed | ✅ YES |

---

## Key Implementation Details

### 1. Data Structures (ciaf/watermarks/models.py)

**ArtifactHashSet Class** (Lines 127-157)
```python
@dataclass
class ArtifactHashSet:
    """
    Dual-state hashing for forensic provenance.

    Critical feature: stores hashes BEFORE and AFTER watermark application.
    This enables detection of watermark removal attacks.
    """
    content_hash_before_watermark: str  # SHA-256 of original AI output
    content_hash_after_watermark: str   # SHA-256 of watermarked version
    canonical_receipt_hash: Optional[str] = None
    normalized_hash_before: Optional[str] = None
    normalized_hash_after: Optional[str] = None
    perceptual_hash_before: Optional[str] = None
    perceptual_hash_after: Optional[str] = None
    simhash_before: Optional[str] = None
    simhash_after: Optional[str] = None
```

### 2. Watermark Application Methods

**Footer Style** (Lines 68-81):
```python
def _apply_footer_watermark(text: str, watermark_id: str, verification_url: str) -> str:
    footer = (
        "\n\n"
        "---\n"
        f"AI Provenance Tag: {watermark_id}\n"
        f"Verify: {verification_url}\n"
        "Generated with CIAF (Cognitive Insight Audit Framework)\n"
    )
    return text + footer
```

**Result:** Original AI output + provenance footer

---

## Security Analysis

### Tamper Detection Capabilities

1. **Exact Match Detection**
   - If suspect text hashes to `hash_after` → **Exact distributed copy**
   - If suspect text hashes to `hash_before` → **Watermark removed** 🚨
   - If neither → Content modified (use similarity matching)

2. **Watermark Removal Detection**
   ```python
   if hash(suspect) == content_hash_before_watermark:
       # Watermark was removed, but original content intact
       alert("Watermark removal detected!")
   ```

3. **Content Modification Detection**
   ```python
   if hash(suspect) != hash_before and hash(suspect) != hash_after:
       # Content was modified after generation
       use_simhash_similarity()
   ```

---

## Conclusion

### ✅ All Requirements Met

1. ✅ **Different Hashes**: `hash_before ≠ hash_after`
2. ✅ **Before Hash**: Computed on **original AI output only**
3. ✅ **After Hash**: Computed on **AI output + watermark**
4. ✅ **Watermark Includes**: Original AI response + provenance metadata
5. ✅ **Evidence Record**: Properly stores both hashes
6. ✅ **Tamper Detection**: Can detect watermark removal
7. ✅ **Forensic Trail**: Complete audit trail maintained

### Implementation Quality: EXCELLENT

- Clean separation of concerns
- Proper cryptographic hashing (SHA-256)
- Multiple hash types for resilience (exact, normalized, SimHash)
- Comprehensive evidence records
- Well-documented code
- Test coverage included

---

## Recommendations

### Current Implementation: No Changes Needed ✅

The dual-state hashing is implemented correctly and working as designed. The implementation:

1. Properly separates original and watermarked content
2. Uses strong cryptographic hashing (SHA-256)
3. Maintains forensic evidence trail
4. Supports multiple matching strategies
5. Enables tamper detection

### Optional Enhancements (Future Consideration):

1. **Add timestamp hashing** - Include generation timestamp in hash chain
2. **Add perceptual hashing for images** - Already supported in models, needs implementation
3. **Add semantic similarity** - Use embeddings for fuzzy matching on heavily modified text
4. **Hash chaining** - Link sequential artifacts cryptographically

---

## Test Validation

**Test File:** `test_dual_hash.py`

**All Assertions Passed:**
- ✅ Hashes are different
- ✅ hash_before matches manual computation on original
- ✅ hash_after matches manual computation on watermarked
- ✅ Original content preserved in watermark
- ✅ Provenance tags present
- ✅ Evidence record properly structured

**Test Status:** PASSED ✅

---

*Review completed: 2026-03-27*
*Reviewer: Claude Opus 4.6*
*Verdict: Implementation is correct and working as intended*
