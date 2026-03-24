#!/usr/bin/env python3
"""
CIAF Watermarks Module - Integration Tests

Tests for forensic provenance watermarking system.

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
from ciaf.watermarks import (
    # Models
    ArtifactType,
    WatermarkType,
    ArtifactEvidence,
    VerificationResult,

    # Text watermarking
    build_text_artifact_evidence,
    apply_text_watermark,
    quick_watermark_text,
    extract_watermark_id,
    has_watermark,
    remove_watermark,

    # Verification
    verify_text_artifact,
    quick_verify,
    analyze_suspect_text,
    format_verification_report,

    # Hashing
    simhash_text,
    simhash_distance,
    normalized_text_hash,

    # Vault
    create_watermark_vault,
)


def test_text_watermarking():
    """Test basic text watermarking."""
    print("\n[TEST] Text Watermarking")
    print("=" * 60)

    raw_text = "The quarterly risk summary indicates elevated model drift in segment B."
    print(f"Original text: {raw_text[:50]}...")

    # Apply watermark
    watermarked, artifact_id = quick_watermark_text(
        text=raw_text,
        model_id="gpt-governed-prod",
        verification_url="https://vault.example.com"
    )

    print(f"Artifact ID: {artifact_id}")
    print(f"Watermarked length: {len(watermarked)} (orig: {len(raw_text)})")

    # Check watermark detection
    assert has_watermark(watermarked), "Watermark not detected!"
    watermark_id = extract_watermark_id(watermarked)
    assert watermark_id is not None, "Watermark ID not extracted!"

    print(f"[OK] Watermark detected: {watermark_id[:20]}...")
    print("[OK] Text watermarking test passed\n")

    return raw_text, watermarked, artifact_id


def test_watermark_removal_detection():
    """Test detection of watermark removal."""
    print("\n[TEST] Watermark Removal Detection")
    print("=" * 60)

    raw_text = "This is AI-generated content with important information."

    # Create watermarked evidence
    evidence, watermarked = build_text_artifact_evidence(
        raw_text=raw_text,
        model_id="test-model",
        model_version="1.0",
        actor_id="test-user",
        prompt="Generate content",
        verification_base_url="https://vault.example.com"
    )

    print(f"Evidence created: {evidence.artifact_id}")

    # Test 1: Verify exact watermarked version
    result = verify_text_artifact(watermarked, evidence)
    assert result.exact_match_after_watermark, "Watermarked version should match!"
    assert result.is_authentic(), "Should be authentic!"
    print("[OK] Exact watermarked match verified")

    # Test 2: Remove watermark and verify
    stripped = remove_watermark(watermarked)
    result2 = verify_text_artifact(stripped, evidence)

    assert result2.exact_match_before_watermark, "Should match pre-watermark version!"
    assert result2.likely_tag_removed, "Should detect watermark removal!"
    assert result2.is_authentic(), "Should still be authentic!"
    print("[OK] Watermark removal detected correctly")

    # Test 3: Completely different text
    fake_text = "This is completely different content."
    result3 = verify_text_artifact(fake_text, evidence)

    assert not result3.exact_match_before_watermark, "Should not match!"
    assert not result3.exact_match_after_watermark, "Should not match!"
    assert not result3.is_authentic(), "Should not be authentic!"
    print("[OK] Unrelated text correctly rejected")

    print("[OK] Watermark removal detection test passed\n")


def test_similarity_matching():
    """Test SimHash similarity matching."""
    print("\n[TEST] Similarity Matching")
    print("=" * 60)

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The fast brown fox leaps over the sleepy dog."  # Similar but different
    text3 = "Completely unrelated text about rockets and space."  # Very different

    # Compute SimHashes
    hash1 = simhash_text(text1)
    hash2 = simhash_text(text2)
    hash3 = simhash_text(text3)

    # Compute distances
    dist_12 = simhash_distance(hash1, hash2)
    dist_13 = simhash_distance(hash1, hash3)

    print(f"Distance (similar): {dist_12}")
    print(f"Distance (different): {dist_13}")

    assert dist_12 < dist_13, "Similar texts should have lower distance!"
    assert dist_12 < 20, "Similar texts should be close!"
    print("[OK] SimHash similarity detection works")

    print("[OK] Similarity matching test passed\n")


def test_normalized_hashing():
    """Test normalized hashing (format-resilient)."""
    print("\n[TEST] Normalized Hashing")
    print("=" * 60)

    text1 = "This is some text."
    text2 = "  THIS   IS    SOME   TEXT.  "  # Different formatting
    text3 = "This is different text."

    hash1 = normalized_text_hash(text1)
    hash2 = normalized_text_hash(text2)
    hash3 = normalized_text_hash(text3)

    print(f"Hash 1: {hash1[:16]}...")
    print(f"Hash 2: {hash2[:16]}...")
    print(f"Hash 3: {hash3[:16]}...")

    assert hash1 == hash2, "Same content with different formatting should match!"
    assert hash1 != hash3, "Different content should not match!"
    print("[OK] Normalized hashing works correctly")

    print("[OK] Normalized hashing test passed\n")


def test_vault_storage():
    """Test watermark vault storage."""
    print("\n[TEST] Vault Storage")
    print("=" * 60)

    # Create temporary vault
    temp_dir = tempfile.mkdtemp(prefix="watermark_vault_test_")
    vault = create_watermark_vault(storage_path=temp_dir)

    try:
        # Create artifact evidence
        evidence, watermarked = build_text_artifact_evidence(
            raw_text="Test content for vault storage.",
            model_id="vault-test-model",
            model_version="1.0",
            actor_id="test-user",
            prompt="Test prompt",
            verification_base_url="https://vault.example.com"
        )

        print(f"Artifact ID: {evidence.artifact_id}")

        # Store in vault
        success = vault.store_evidence(evidence)
        assert success, "Storage should succeed!"
        print("[OK] Evidence stored in vault")

        # Retrieve from vault
        retrieved = vault.retrieve_evidence(evidence.artifact_id)
        assert retrieved is not None, "Should retrieve evidence!"
        assert retrieved.artifact_id == evidence.artifact_id, "IDs should match!"
        print("[OK] Evidence retrieved from vault")

        # Search by model
        results = vault.search_by_model("vault-test-model")
        assert len(results) > 0, "Should find artifacts!"
        print(f"[OK] Found {len(results)} artifact(s) for model")

        # Search by watermark
        found = vault.search_by_watermark(evidence.watermark.watermark_id)
        assert found is not None, "Should find by watermark ID!"
        print("[OK] Search by watermark ID works")

        print("[OK] Vault storage test passed\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_verification_report():
    """Test verification report formatting."""
    print("\n[TEST] Verification Report")
    print("=" * 60)

    # Create evidence and verify
    raw_text = "Sample AI-generated content for testing."
    evidence, watermarked = build_text_artifact_evidence(
        raw_text=raw_text,
        model_id="report-test",
        model_version="1.0",
        actor_id="tester",
        prompt="Generate sample",
        verification_base_url="https://vault.example.com"
    )

    result = verify_text_artifact(watermarked, evidence)

    # Generate report
    report = format_verification_report(result, detailed=True)

    print(report)
    print()

    assert "AUTHENTIC" in report, "Report should show authentic!"
    assert result.artifact_id in report, "Report should include artifact ID!"
    print("[OK] Verification report generated")

    print("[OK] Verification report test passed\n")


def test_suspect_artifact_analysis():
    """Test analysis of suspect artifacts."""
    print("\n[TEST] Suspect Artifact Analysis")
    print("=" * 60)

    # Watermarked text
    watermarked, artifact_id = quick_watermark_text(
        "Some AI content",
        "test-model"
    )

    # Analyze
    analysis = analyze_suspect_text(watermarked)

    print(f"Has watermark: {analysis['has_ciaf_watermark']}")
    print(f"Watermark ID: {analysis['watermark_id']}")
    print(f"Text length: {analysis['text_length']}")

    assert analysis['has_ciaf_watermark'], "Should detect watermark!"
    assert analysis['watermark_id'] is not None, "Should extract ID!"
    print("[OK] Suspect analysis works")

    print("[OK] Suspect artifact analysis test passed\n")


def run_all_tests():
    """Run all watermark tests."""
    print("=" * 60)
    print("CIAF Watermarks - Integration Tests")
    print("=" * 60)

    tests = [
        test_text_watermarking,
        test_watermark_removal_detection,
        test_similarity_matching,
        test_normalized_hashing,
        test_vault_storage,
        test_verification_report,
        test_suspect_artifact_analysis,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
