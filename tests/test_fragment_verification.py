#!/usr/bin/env python3
"""
CIAF Watermarks - Fragment Verification Tests

Tests for forensic fragment verification system, specifically testing:
- Bug #161 fix: Fragment text vs hash matching
- Sliding window verification
- Hash integrity verification
- Attack scenario detection

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks.fragment_selection import select_text_forensic_fragments
from ciaf.watermarks.fragment_verification import (
    verify_text_fragments,
    verify_text_fragment_sliding_window,
)
from ciaf.watermarks.models import sha256_text


def test_fragment_text_field_population():
    """
    Test that TextForensicFragment.fragment_text is properly populated.

    This ensures the fix for bug #161 is working - fragments must store
    the actual text content, not just hashes.
    """
    print("\n[TEST] Fragment Text Field Population")
    print("=" * 60)

    # Use longer text (> 300 chars needed for fragment_length=200)
    original_text = """The quarterly risk assessment reveals significant model drift 
    in production segment B, affecting critical business operations across multiple 
    departments. We observed a 12% increase in false positives over the last 30 days, 
    which has elevated operational risk levels beyond acceptable thresholds. The 
    elevated risk requires immediate attention from the model governance team and 
    executive stakeholders. We strongly recommend retraining with recent data samples 
    and implementing additional validation checkpoints in the deployment pipeline."""

    print(f"  Text length: {len(original_text)} chars")

    # Compute hashes for the text (required by function signature but not used internally)
    hash_before = sha256_text(original_text)
    hash_after = sha256_text(
        original_text + " [WATERMARK]"
    )  # Simulated watermarked version

    # Select fragments - use low entropy threshold to ensure we get some
    fragments = select_text_forensic_fragments(
        raw_text=original_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,  # Accept any entropy to ensure fragments are selected
    )

    print(f"  Selected {len(fragments)} fragments")
    assert (
        len(fragments) > 0
    ), f"Should select at least one fragment (got {len(fragments)})"

    for frag in fragments:
        # ✅ Critical check: fragment_text must exist and be populated
        assert hasattr(frag, "fragment_text"), "Fragment missing fragment_text field!"
        assert frag.fragment_text is not None, "fragment_text is None!"
        assert len(frag.fragment_text) > 0, "fragment_text is empty!"
        assert (
            frag.fragment_text in original_text
        ), "fragment_text not found in original!"

        # Verify fragment text matches the hash
        computed_hash = sha256_text(frag.fragment_text)
        assert computed_hash == frag.fragment_hash_before, (
            f"Fragment text hash mismatch! Expected {frag.fragment_hash_before[:8]}..., "
            f"got {computed_hash[:8]}..."
        )

        print(f"  ✅ Fragment {frag.fragment_id}:")
        print(f"     Text: '{frag.fragment_text[:30]}...'")
        print(f"     Hash: {frag.fragment_hash_before[:16]}...")
        print(f"     Length: {len(frag.fragment_text)} chars")

    print("[OK] All fragments have valid fragment_text field\n")


def test_sliding_window_exact_match():
    """
    Test sliding window matching with exact fragment text.

    This tests the core fix for bug #161 - verify_text_fragment_sliding_window
    should receive actual text content, not SHA-256 hashes.
    """
    print("\n[TEST] Sliding Window Exact Match")
    print("=" * 60)

    suspect_text = """The quarterly risk assessment reveals significant model drift 
    in production segment B. We observed a 12% increase in false positives."""

    fragment_text = "quarterly risk assessment reveals significant"

    # ✅ Should find exact match
    result = verify_text_fragment_sliding_window(suspect_text, fragment_text)

    assert result is not None, "Should find exact match!"
    pos, confidence = result

    assert pos >= 0, "Position should be non-negative"
    assert confidence >= 0.95, f"Confidence too low for exact match: {confidence}"

    # Verify the match is correct
    extracted = suspect_text[pos : pos + len(fragment_text)]
    assert extracted == fragment_text, f"Extracted text doesn't match: '{extracted}'"

    print(f"  ✅ Found at position {pos} with confidence {confidence:.2f}")
    print(f"  ✅ Extracted: '{extracted[:30]}...'")
    print("[OK] Sliding window exact match test passed\n")


def test_sliding_window_case_variation():
    """
    Test sliding window with case variations.

    Note: The current implementation uses character-by-character matching,
    so case variations will reduce confidence. This test verifies that
    the system handles case variations reasonably (doesn't crash, provides
    meaningful confidence scores).
    """
    print("\n[TEST] Sliding Window Case Variation")
    print("=" * 60)

    suspect_text = "THE QUARTERLY RISK ASSESSMENT reveals significant model drift."
    fragment_text = "quarterly risk assessment reveals"

    # Try to find match despite case variation
    result = verify_text_fragment_sliding_window(suspect_text, fragment_text)

    if result is not None:
        pos, confidence = result
        print(f"  ✅ Found at position {pos} with confidence {confidence:.2f}")
        print("     (Case variation reduces confidence from exact match)")
    else:
        print("  ℹ️  No high-confidence match found (expected due to case sensitivity)")
        print("     Current implementation uses case-sensitive character matching")

    print("[OK] Case variation test completed\n")


def test_fragment_verification_workflow():
    """
    Test the complete fragment verification workflow.

    This is the integration test for bug #161 fix - ensuring the entire
    chain from fragment selection to verification works correctly.
    """
    print("\n[TEST] Complete Fragment Verification Workflow")
    print("=" * 60)

    # Original text with watermark removed (must be >300 chars for fragment_length=200)
    original_text = """The AI model governance framework requires continuous monitoring 
    of production deployments across all organizational units and business segments. 
    Our forensic provenance layer ensures full traceability of all AI-generated artifacts, 
    providing complete audit trails and regulatory compliance. The verification system uses 
    dual-state hashing to detect tampering and watermark removal, enabling defensible 
    attribution even when content has been modified or stripped of metadata."""

    # Compute hashes
    hash_before = sha256_text(original_text)
    hash_after = sha256_text(original_text + " [W]")

    # Select fragments from original (use min_entropy=0.0 to ensure selection)
    fragments = select_text_forensic_fragments(
        raw_text=original_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )

    assert len(fragments) > 0, "Should select fragments"
    print(f"  Selected {len(fragments)} fragments")

    # Scenario 1: Verify against exact original (should match all)
    print("\n  Scenario 1: Exact match")
    result = verify_text_fragments(original_text, fragments)

    assert result.fragments_matched == len(
        fragments
    ), f"Should match all fragments! Got {result.fragments_matched}/{len(fragments)}"
    assert (
        result.match_confidence >= 0.95
    ), f"Confidence too low: {result.match_confidence}"

    print(f"    ✅ All {result.fragments_matched} fragments matched")
    print(f"    ✅ Confidence: {result.match_confidence:.2f}")

    # Scenario 2: Verify against modified text (should still match some)
    print("\n  Scenario 2: Modified text")
    modified_text = original_text.replace("AI model", "Machine learning model")
    result2 = verify_text_fragments(modified_text, fragments)

    # Should still find most fragments
    assert result2.fragments_matched > 0, "Should find at least some fragments!"
    print(f"    ✅ Found {result2.fragments_matched}/{len(fragments)} fragments")
    print(f"    ✅ Confidence: {result2.match_confidence:.2f}")

    # Scenario 3: Verify against completely different text (should not match)
    print("\n  Scenario 3: Unrelated text")
    unrelated_text = """This is completely different content about weather patterns 
    and climate change. Nothing to do with AI governance or forensics."""
    result3 = verify_text_fragments(unrelated_text, fragments)

    assert (
        result3.fragments_matched == 0
    ), f"Should not match unrelated text! Found {result3.fragments_matched} matches"
    assert (
        result3.match_confidence < 0.5
    ), f"Confidence should be low for unrelated text: {result3.match_confidence}"

    print("    ✅ No matches in unrelated text (correct)")
    print(f"    ✅ Confidence: {result3.match_confidence:.2f}")

    print("[OK] Complete workflow test passed\n")


def test_hash_verification_after_match():
    """
    Test that hash verification occurs after finding a fragment match.

    This ensures the added security check is working - we verify that
    the matched fragment's hash matches the stored hash.
    """
    print("\n[TEST] Hash Verification After Match")
    print("=" * 60)

    original_text = """The forensic provenance system ensures artifact traceability and 
    complete audit trails across all AI-generated content in production environments. 
    This comprehensive framework provides defensible attribution and regulatory compliance 
    through advanced cryptographic hashing techniques and forensic fragment verification. 
    The system maintains chain-of-custody evidence for all generated artifacts."""

    hash_before = sha256_text(original_text)
    hash_after = sha256_text(original_text + "[W]")

    fragments = select_text_forensic_fragments(
        raw_text=original_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )

    assert len(fragments) >= 1, "Should select at least one fragment"
    fragment = fragments[0]

    # Verify against original - hash should match
    result = verify_text_fragments(original_text, fragments)

    assert result.fragments_matched >= 1, "Should find at least one fragment"
    match_result = result.forensic_matches[0]

    # Check match details for hash verification message
    assert (
        "hash verified" in match_result.match_details.lower()
    ), "Match details should mention hash verification"

    print(f"  ✅ Fragment matched at position {match_result.match_position}")
    print(f"  ✅ Match details: {match_result.match_details}")

    # Now modify the text slightly and verify
    modified_text = original_text.replace("artifact", "content")
    result2 = verify_text_fragments(modified_text, fragments)

    # Should still match (sliding window finds it) but confidence may be reduced
    if result2.fragments_matched > 0:
        match_result2 = result2.forensic_matches[0]
        print("\n  Modified text:")
        print(f"  ✅ Fragment matched at position {match_result2.match_position}")
        print(f"  ✅ Match details: {match_result2.match_details}")

        # If hash doesn't match, confidence should be reduced
        if "hash mismatch" in match_result2.match_details.lower():
            assert (
                match_result2.confidence < match_result.confidence
            ), "Confidence should be reduced for hash mismatch"
            print("  ✅ Confidence correctly reduced due to hash mismatch")

    print("[OK] Hash verification test passed\n")


def test_bug_161_regression():
    """
    Regression test for Bug #161.

    Before fix: fragment_verification.py:161 passed fragment.fragment_hash_before
                (a 64-char SHA-256 hex string) to the sliding window matcher,
                causing it to search for the hash string instead of the actual text.

    After fix:  Passes fragment.fragment_text (the actual sampled content).

    This test ensures the bug doesn't resurface.
    """
    print("\n[TEST] Bug #161 Regression Test")
    print("=" * 60)

    test_text = """The critical risk requires immediate governance oversight from executive 
    leadership and compliance teams. This escalated situation demands comprehensive review 
    of all operational controls and risk mitigation strategies. The governance framework must 
    ensure proper authorization workflows and audit trails are maintained throughout the 
    entire incident response process and subsequent remediation activities."""

    hash_before = sha256_text(test_text)
    hash_after = sha256_text(test_text + "[W]")

    fragments = select_text_forensic_fragments(
        raw_text=test_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )

    assert len(fragments) >= 1, "Should select at least one fragment"
    fragment = fragments[0]

    print(f"  Fragment text: '{fragment.fragment_text}'")
    print(f"  Fragment hash: {fragment.fragment_hash_before}")

    # Critical check: fragment_text should NOT be a 64-character hex string
    assert len(fragment.fragment_text) != 64 or not all(
        c in "0123456789abcdef" for c in fragment.fragment_text.lower()
    ), "❌ BUG #161 DETECTED! fragment_text appears to be a hash, not actual text!"

    # Critical check: fragment_hash_before SHOULD be a 64-character hex string
    assert (
        len(fragment.fragment_hash_before) == 64
    ), "fragment_hash_before should be 64 chars"
    assert all(
        c in "0123456789abcdef" for c in fragment.fragment_hash_before.lower()
    ), "fragment_hash_before should be a hex string"

    print("  ✅ fragment_text is actual text (not hash)")
    print("  ✅ fragment_hash_before is proper SHA-256 hex")

    # Verify the fragment is found
    result = verify_text_fragments(test_text, fragments)

    assert (
        result.fragments_matched >= 1
    ), "❌ BUG #161 REGRESSION! Fragment not found - likely passing hash instead of text to matcher"

    print("  ✅ Fragment successfully verified in original text")
    print(f"  ✅ Confidence: {result.match_confidence:.2f}")

    print("[OK] Bug #161 regression test passed - fix is working!\n")


def run_all_tests():
    """Run all fragment verification tests."""
    print("\n" + "=" * 60)
    print("CIAF WATERMARKS - FRAGMENT VERIFICATION TEST SUITE")
    print("Testing Bug #161 fix and fragment verification workflow")
    print("=" * 60)

    try:
        # Test 1: Fragment text field population
        fragments = test_fragment_text_field_population()

        # Test 2: Sliding window exact match
        test_sliding_window_exact_match()

        # Test 3: Sliding window case variation
        test_sliding_window_case_variation()

        # Test 4: Complete workflow
        test_fragment_verification_workflow()

        # Test 5: Hash verification
        test_hash_verification_after_match()

        # Test 6: Bug #161 regression
        test_bug_161_regression()

        # Summary
        print("\n" + "=" * 60)
        print("✅ ALL FRAGMENT VERIFICATION TESTS PASSED")
        print("=" * 60)
        print("\nBug #161 Fix Verified:")
        print("  ✅ fragment_text field properly populated")
        print("  ✅ Sliding window receives actual text (not hash)")
        print("  ✅ Hash verification works as security layer")
        print("  ✅ Complete workflow functions correctly")
        print("  ✅ No regression detected")
        print("\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
