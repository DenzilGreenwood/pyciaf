"""
Test to verify dual-state hashing works correctly.

This test validates that:
1. content_hash_before_watermark is computed on the ORIGINAL AI output
2. content_hash_after_watermark is computed on the WATERMARKED output
3. The two hashes are DIFFERENT
4. The watermark includes the AI response
"""

import sys

sys.path.insert(0, ".")

from ciaf.watermarks.text import build_text_artifact_evidence
from ciaf.watermarks.hashing import sha256_text


def test_dual_state_hashing():
    """Test that dual-state hashing produces different hashes."""

    print("=" * 70)
    print("DUAL-STATE HASHING VERIFICATION TEST")
    print("=" * 70)

    # Simulate AI-generated response
    original_ai_output = """The capital of France is Paris. Paris is known for its
art, culture, and the Eiffel Tower. It is one of the most visited cities
in the world."""

    prompt = "What is the capital of France?"

    print("\n1. ORIGINAL AI OUTPUT (before watermark):")
    print("-" * 70)
    print(original_ai_output)
    print("-" * 70)

    # Build artifact evidence (this applies watermark and computes hashes)
    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text=original_ai_output,
        model_id="gpt-4",
        model_version="1.0",
        actor_id="test_user",
        prompt=prompt,
        verification_base_url="https://verify.example.com",
        watermark_style="footer",
        include_simhash=True,
    )

    print("\n2. WATERMARKED OUTPUT (after watermark):")
    print("-" * 70)
    print(watermarked_text)
    print("-" * 70)

    # Get the hashes from the evidence
    hash_before = evidence.hashes.content_hash_before_watermark
    hash_after = evidence.hashes.content_hash_after_watermark

    print("\n3. HASH COMPARISON:")
    print("-" * 70)
    print(f"Hash BEFORE watermark: {hash_before}")
    print(f"Hash AFTER watermark:  {hash_after}")
    print(f"Hashes are different:  {hash_before != hash_after}")
    print("-" * 70)

    # Verify hash_before matches manual computation on original text
    manual_hash_before = sha256_text(original_ai_output)
    print("\n4. VERIFICATION - Hash Before Watermark:")
    print("-" * 70)
    print(f"Stored hash_before:      {hash_before}")
    print(f"Manual hash of original: {manual_hash_before}")
    print(f"Match:                   {hash_before == manual_hash_before}")
    print("-" * 70)

    # Verify hash_after matches manual computation on watermarked text
    manual_hash_after = sha256_text(watermarked_text)
    print("\n5. VERIFICATION - Hash After Watermark:")
    print("-" * 70)
    print(f"Stored hash_after:        {hash_after}")
    print(f"Manual hash of watermark: {manual_hash_after}")
    print(f"Match:                    {hash_after == manual_hash_after}")
    print("-" * 70)

    # Check that watermark contains AI response
    print("\n6. WATERMARK CONTENT CHECK:")
    print("-" * 70)
    original_content_in_watermark = original_ai_output in watermarked_text
    watermark_tag_present = "AI Provenance Tag:" in watermarked_text
    verification_url_present = "Verify:" in watermarked_text
    ciaf_attribution = "CIAF" in watermarked_text

    print(f"Original AI response in watermarked text: {original_content_in_watermark}")
    print(f"Watermark tag present:                    {watermark_tag_present}")
    print(f"Verification URL present:                 {verification_url_present}")
    print(f"CIAF attribution present:                 {ciaf_attribution}")
    print("-" * 70)

    # Additional evidence record checks
    print("\n7. EVIDENCE RECORD VALIDATION:")
    print("-" * 70)
    print(f"Artifact ID:           {evidence.artifact_id}")
    print(f"Watermark ID:          {evidence.watermark.watermark_id}")
    print(f"Model ID:              {evidence.model_id}")
    print(f"Prompt hash:           {evidence.prompt_hash}")
    print(f"Output hash raw:       {evidence.output_hash_raw}")
    print(f"Output hash distrib:   {evidence.output_hash_distributed}")
    print(f"Watermark type:        {evidence.watermark.watermark_type.value}")
    print(f"Removal resistance:    {evidence.watermark.removal_resistance}")
    print("-" * 70)

    # Normalized hashes
    print("\n8. NORMALIZED HASHES:")
    print("-" * 70)
    print(f"Normalized before: {evidence.hashes.normalized_hash_before}")
    print(f"Normalized after:  {evidence.hashes.normalized_hash_after}")
    print(
        f"Different:         {evidence.hashes.normalized_hash_before != evidence.hashes.normalized_hash_after}"
    )
    print("-" * 70)

    # SimHash fingerprints
    if evidence.hashes.simhash_before and evidence.hashes.simhash_after:
        print("\n9. SIMHASH FINGERPRINTS:")
        print("-" * 70)
        print(f"SimHash before: {evidence.hashes.simhash_before}")
        print(f"SimHash after:  {evidence.hashes.simhash_after}")
        print(
            f"Different:      {evidence.hashes.simhash_before != evidence.hashes.simhash_after}"
        )
        print("-" * 70)

    # FINAL ASSERTIONS
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    # Critical assertions
    assert hash_before != hash_after, "❌ FAIL: Hashes should be different!"
    assert (
        hash_before == manual_hash_before
    ), "❌ FAIL: hash_before doesn't match original!"
    assert (
        hash_after == manual_hash_after
    ), "❌ FAIL: hash_after doesn't match watermarked!"
    assert (
        original_content_in_watermark
    ), "❌ FAIL: Original AI response not in watermark!"
    assert watermark_tag_present, "❌ FAIL: Watermark tag missing!"
    assert verification_url_present, "❌ FAIL: Verification URL missing!"

    # Additional checks
    assert evidence.output_hash_raw == hash_before, "❌ FAIL: output_hash_raw mismatch!"
    assert (
        evidence.output_hash_distributed == hash_after
    ), "❌ FAIL: output_hash_distributed mismatch!"

    print("✅ ALL TESTS PASSED!")
    print()
    print("VERIFIED:")
    print("  ✓ Dual-state hashing produces different hashes")
    print("  ✓ hash_before computed on ORIGINAL AI output")
    print("  ✓ hash_after computed on WATERMARKED output")
    print("  ✓ Watermark contains original AI response")
    print("  ✓ Watermark includes provenance tags")
    print("  ✓ Evidence record properly structured")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_dual_state_hashing()
        print("\n✅ SUCCESS: Dual-state hashing is working correctly!\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
