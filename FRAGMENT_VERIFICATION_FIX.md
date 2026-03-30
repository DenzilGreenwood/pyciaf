# Fragment Verification Bug - Quick Fix Guide

**Bug ID**: #161  
**Severity**: HIGH (Core Feature Broken)  
**File**: `ciaf/watermarks/fragment_verification.py:161`  
**Impact**: Fragment verification unusable - searches for hash strings instead of text content

---

## Problem

```python
# CURRENT CODE (Line 161 of fragment_verification.py)
match_result = verify_text_fragment_sliding_window(
    suspect_text, 
    fragment.fragment_hash_before  # ❌ BUG: This is a SHA-256 hash, not text!
)

# This passes something like:
# "7d6d5c0b8f4e3a2c9b1d4f5e6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6"
# instead of the actual fragment text like:
# "The quarterly risk summary indicates elevated model drift."
```

**What Happens**:
- `verify_text_fragment_sliding_window()` receives a 64-character hex string
- Tries to find that hex string in the suspect document
- Never matches (suspect contains the original text, not the hash)
- All fragment verifications fail incorrectly

---

## Root Cause

`TextForensicFragment` model stores **hashes** but not **actual fragment text**:

```python
# FILE: ciaf/watermarks/models.py
@dataclass
class TextForensicFragment(ForensicFragment):
    offset_start: int
    offset_end: int
    fragment_length: int
    sample_location: str
    
    # Stores hashes only:
    fragment_hash_before: str   # SHA-256 of fragment
    fragment_hash_after: str    # SHA-256 after watermark
    
    # ❌ MISSING: The actual fragment text!
```

---

## Fix 1: Update Data Model

**File**: `ciaf/watermarks/models.py` (around line 159)

```python
@dataclass
class TextForensicFragment(ForensicFragment):
    """
    High-entropy text fragment for granular verification.

    Stores:
    - Fragment text for sliding window matching
    - Fragment hash before/after watermark for integrity verification
    - Character position in original document
    - Fragment length and entropy score
    """

    offset_start: int  # Character offset in document
    offset_end: int  # End offset
    fragment_length: int  # Length of fragment
    sample_location: str  # 'beginning', 'middle', 'end'
    
    # ✅ ADD THIS FIELD:
    fragment_text: str  # The actual sampled text (for sliding window matching)

    # Dual-state fragment hashing (for integrity verification)
    fragment_hash_before: str  # SHA-256 of fragment before watermark
    fragment_hash_after: str  # SHA-256 of fragment after watermark

    # Optional similarity hashing
    fragment_simhash_before: Optional[str] = None
    fragment_simhash_after: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result.pop("fragment_type", None)  # Avoid duplication
        return result
```

---

## Fix 2: Update Fragment Selection

**File**: `ciaf/watermarks/fragment_selection.py`

Update the `select_text_forensic_fragments()` function to **store the actual text**:

```python
def select_text_forensic_fragments(
    raw_text: str,
    watermarked_text: str,
    num_fragments: int = 3,
    min_entropy: float = 0.6,
) -> List[TextForensicFragment]:
    """Select high-entropy text fragments for verification."""
    
    # ... existing entropy calculation logic ...
    
    # When creating fragments:
    for start, end, entropy, location in selected_segments:
        # ✅ EXTRACT THE ACTUAL TEXT:
        fragment_text = raw_text[start:end]
        fragment_text_watermarked = watermarked_text[start:end]
        
        fragment = TextForensicFragment(
            fragment_id=f"frag_{start}_{location}",
            fragment_type="text",
            entropy_score=entropy,
            sampling_method=location,
            content_position=start,
            offset_start=start,
            offset_end=end,
            fragment_length=end - start,
            sample_location=location,
            
            # ✅ ADD THE TEXT:
            fragment_text=fragment_text,
            
            # Compute hashes of the text:
            fragment_hash_before=sha256_text(fragment_text),
            fragment_hash_after=sha256_text(fragment_text_watermarked),
            fragment_simhash_before=compute_simhash(fragment_text) if compute_simhash else None,
            fragment_simhash_after=compute_simhash(fragment_text_watermarked) if compute_simhash else None,
        )
        
        fragments.append(fragment)
    
    return fragments
```

---

## Fix 3: Update Verification Logic

**File**: `ciaf/watermarks/fragment_verification.py:161`

Update to use `fragment_text` instead of `fragment_hash_before`:

```python
def verify_text_fragments(
    suspect_text: str,
    stored_fragments: List[TextForensicFragment],
) -> ForensicVerificationSummary:
    """
    Verify suspect text against stored text forensic fragments.
    """
    results: List[FragmentMatchResult] = []
    notes: List[str] = []
    matches_found = 0

    for fragment in stored_fragments:
        # ✅ FIX: Use fragment_text instead of fragment_hash_before
        match_result = verify_text_fragment_sliding_window(
            suspect_text, 
            fragment.fragment_text  # ✅ Now passes actual text
        )

        if match_result:
            pos, confidence = match_result
            results.append(
                FragmentMatchResult(
                    fragment_id=fragment.fragment_id,
                    matched=True,
                    confidence=confidence,
                    match_position=pos,
                    match_details=f"Found at character {pos}",
                )
            )
            matches_found += 1
            
            # ✅ OPTIONAL: Verify hash integrity
            matched_text = suspect_text[pos:pos + len(fragment.fragment_text)]
            matched_hash = sha256_text(matched_text)
            if matched_hash == fragment.fragment_hash_before:
                results[-1].match_details += " (hash verified)"
        else:
            results.append(
                FragmentMatchResult(
                    fragment_id=fragment.fragment_id,
                    matched=False,
                    confidence=0.0,
                    match_details="Not found in suspect text",
                )
            )

    # ... rest of function ...
```

---

## Fix 4: Update JSON Schema

**File**: `ciaf/schemas/text-forensic-fragment.schema.json` (if exists)

Add the `fragment_text` field:

```json
{
  "$id": "https://cognitiveinsight.ai/schemas/text-forensic-fragment.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CIAF Text Forensic Fragment",
  "type": "object",
  "required": [
    "fragment_id",
    "fragment_text",
    "fragment_hash_before",
    "offset_start",
    "offset_end"
  ],
  "properties": {
    "fragment_id": {
      "type": "string",
      "description": "Unique fragment identifier"
    },
    "fragment_text": {
      "type": "string",
      "description": "The actual sampled text for sliding window matching"
    },
    "fragment_hash_before": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$",
      "description": "SHA-256 hash of fragment before watermark"
    },
    "fragment_hash_after": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$",
      "description": "SHA-256 hash of fragment after watermark"
    }
  }
}
```

---

## Fix 5: Add Tests

**File**: `tests/test_watermarks.py` (new test)

```python
def test_fragment_verification_fix():
    """Test that fragment verification works correctly with actual text."""
    
    # Create test text
    raw_text = "The quarterly risk summary indicates elevated model drift. This is important."
    watermarked_text = raw_text + "\n---\nAI Provenance Tag: wmk-test-123"
    
    # Select fragments (should now store fragment_text)
    fragments = select_text_forensic_fragments(
        raw_text=raw_text,
        watermarked_text=watermarked_text,
        num_fragments=2,
    )
    
    # Verify fragments have text stored
    for fragment in fragments:
        assert hasattr(fragment, 'fragment_text'), "Fragment must have fragment_text field"
        assert len(fragment.fragment_text) > 0, "Fragment text must not be empty"
        assert isinstance(fragment.fragment_text, str), "Fragment text must be string"
    
    # Test verification with exact match
    result = verify_text_fragments(raw_text, fragments)
    assert result.fragments_matched >= 1, "Should match at least one fragment"
    assert result.match_confidence > 0.5, "Should have decent confidence"
    
    # Test verification with slight modification
    modified_text = raw_text.replace("elevated", "reduced")
    result2 = verify_text_fragments(modified_text, fragments)
    # May or may not match depending on which fragment was selected
    
    # Test verification with watermark removed
    suspect_without_watermark = raw_text
    result3 = verify_text_fragments(suspect_without_watermark, fragments)
    assert result3.fragments_matched >= 1, "Should still match original content"
    
    print("✓ Fragment verification fix validated")
```

---

## Migration Strategy

### For Existing Data (If Fragments Already Stored)

If you have existing `TextForensicFragment` records without `fragment_text`:

```python
def migrate_existing_fragments(vault):
    """Migrate existing fragments to include text."""
    
    print("⚠️ WARNING: Existing fragments missing fragment_text")
    print("   They cannot be used for verification.")
    print("   Recommendation: Regenerate fragments with fixed code.")
    
    # Option 1: Mark as deprecated
    # Option 2: Try to reconstruct from original documents (if available)
    # Option 3: Delete and regenerate
```

### For New Deployments

Just apply the fixes above. No migration needed.

---

## Testing Checklist

Before marking as fixed:

- [ ] Updated `TextForensicFragment` model with `fragment_text` field
- [ ] Updated fragment selection to extract and store `fragment_text`
- [ ] Updated verification logic to use `fragment_text` instead of hash
- [ ] Added unit tests for fragment matching
- [ ] Tested with real suspect documents
- [ ] Tested with modified content
- [ ] Tested with watermark removed
- [ ] Tested with completely unrelated text
- [ ] Updated documentation
- [ ] Updated JSON schema (if exists)

---

## Verification

After fix, this should work:

```python
from ciaf.watermarks import (
    build_text_artifact_evidence,
    select_text_forensic_fragments,
    verify_text_fragments,
)

# Generate watermarked text
evidence, watermarked = build_text_artifact_evidence(
    raw_text="The quarterly risk summary indicates elevated model drift.",
    model_id="test-model",
    model_version="1.0",
    actor_id="test-user",
    prompt="test",
    verification_base_url="https://test.com"
)

# Select fragments (now includes fragment_text)
fragments = select_text_forensic_fragments(
    raw_text=evidence.output_hash_raw,  # Would need to store original text
    watermarked_text=watermarked,
    num_fragments=3
)

# Verify against suspect text
suspect = "The quarterly risk summary indicates elevated model drift."
result = verify_text_fragments(suspect, fragments)

# Should match!
assert result.fragments_matched >= 1
assert result.match_confidence > 0.7
print("✓ Fragment verification working correctly")
```

---

## Priority: HIGH

This is a **blocking bug** for forensic fragment verification feature.

**Estimated Fix Time**: 2-4 hours  
**Testing Time**: 2-3 hours  
**Total**: 1 day max

---

**Status**: Fix pending  
**Assignee**: Development team  
**Target**: Before v1.3.2 release
