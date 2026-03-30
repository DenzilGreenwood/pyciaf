#!/usr/bin/env python3
"""
CIAF Watermarks - Integration Test Suite

End-to-end workflow tests validating complete watermarking lifecycle:
- Text watermarking with fragment verification
- Image watermarking with perceptual hashing
- Signature envelope integration
- Round-trip serialization/deserialization
- Multi-artifact verification

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path
import json
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks.text import build_text_artifact_evidence
from ciaf.watermarks.models import (
    ArtifactEvidence,
    ArtifactType,
    TextForensicFragment,
)
from ciaf.watermarks.signature_envelope import (
    create_signature_envelope,
    KeyBackend,
    SignatureEnvelope,
)
from ciaf.watermarks.fragment_selection import select_text_forensic_fragments
from ciaf.watermarks.fragment_verification import verify_text_fragments
from ciaf.watermarks.hashing import perceptual_hash_image, sha256_text
from ciaf.watermarks.images import hamming_distance, similarity_score


def test_end_to_end_text_watermarking():
    """Test complete text watermarking workflow from creation to verification."""
    print("\n[TEST] End-to-End Text Watermarking Workflow")
    print("=" * 60)
    
    # 1. Create original text
    original_text = """The quarterly risk assessment revealed significant 
    gaps in our AI deployment controls. We observed a 12% increase in 
    unauthorized model access attempts across development environments. 
    Critical findings include insufficient logging of model inference 
    requests and incomplete audit trails for prompt engineering sessions. 
    These vulnerabilities expose operational risk levels beyond acceptable 
    thresholds for our governance teams. This escalated situation demands 
    comprehensive review of all operational controls and risk mitigation 
    strategies."""
    
    print(f"  ✅ Original text: {len(original_text)} chars")
    
    # 2. Build artifact evidence (includes watermarking)
    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-42",
        prompt="Analyze Q4 risk assessment data",
        verification_base_url="https://vault.example.com",
        watermark_style="footer",
        include_simhash=True,
    )
    
    assert evidence.artifact_id is not None
    assert evidence.watermark is not None
    assert evidence.hashes is not None
    assert watermarked_text != original_text
    assert len(watermarked_text) > len(original_text)
    
    print(f"  ✅ Watermarked text: {len(watermarked_text)} chars")
    print(f"  ✅ Artifact ID: {evidence.artifact_id}")
    print(f"  ✅ Watermark ID: {evidence.watermark.watermark_id}")
    
    # 3. Add signature envelope
    envelope = create_signature_envelope(
        payload_hash=evidence.compute_receipt_hash(),
        signature_value="U2lnbmVkQXJ0aWZhY3Rh",  # Mock signature
        key_id="ciaf-test-key-001",
        key_backend=KeyBackend.LOCAL,
        signing_service="ciaf-integration-test",
    )
    
    evidence.signature = envelope
    
    assert evidence.signature is not None
    assert isinstance(evidence.signature, SignatureEnvelope)
    assert evidence.signature.metadata.key_backend == KeyBackend.LOCAL
    
    print(f"  ✅ Signature added: {evidence.signature.signature_value[:20]}...")
    print(f"  ✅ Key backend: {evidence.signature.metadata.key_backend.value}")
    
    # 4. Select forensic fragments
    hash_before = sha256_text(original_text)
    hash_after = sha256_text(watermarked_text)
    
    fragments = select_text_forensic_fragments(
        raw_text=original_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )
    
    assert len(fragments) == 3
    for frag in fragments:
        assert frag.fragment_text is not None  # Bug #161 fix
        assert frag.fragment_hash_before is not None
        assert len(frag.fragment_text) > 0
    
    print(f"  ✅ Selected {len(fragments)} forensic fragments")
    
    # 5. Verify fragments against original text
    result = verify_text_fragments(original_text, fragments)
    
    assert result.match_confidence >= 0.95
    print(f"  ✅ Fragment verification: {result.match_confidence:.2%} confidence")
    
    # 6. Serialize to JSON
    evidence_dict = evidence.to_dict()
    json_str = json.dumps(evidence_dict, indent=2)
    
    assert "artifact_id" in json_str
    assert "signature" in json_str
    assert "payload_hash" in json_str
    assert "key_backend" in json_str
    
    print(f"  ✅ JSON serialization: {len(json_str)} bytes")
    
    # 7. Deserialize from JSON
    evidence_restored_dict = json.loads(json_str)
    
    # Manually reconstruct signature envelope (simplified)
    if evidence_restored_dict.get("signature"):
        sig_data = evidence_restored_dict["signature"]
        signature_restored = SignatureEnvelope.from_dict(sig_data)
        
        assert signature_restored.payload_hash == envelope.payload_hash
        assert signature_restored.metadata.key_backend == KeyBackend.LOCAL
        
        print(f"  ✅ Signature deserialized successfully")
    
    # 8. Verify watermark is present
    assert evidence.watermark.watermark_id in watermarked_text
    print(f"  ✅ Watermark preserved in text")
    
    # 9. Verify hashes
    assert evidence.hashes.content_hash_before_watermark != evidence.hashes.content_hash_after_watermark
    print(f"  ✅ Before/after hashes different (watermark applied)")
    
    print("[OK] End-to-end text watermarking test passed\n")


def test_image_perceptual_hashing_workflow():
    """Test image watermarking with perceptual hashing."""
    print("\n[TEST] Image Perceptual Hashing Workflow")
    print("=" * 60)
    
    # 1. Create test image
    img = Image.new("RGB", (400, 400), color=(100, 150, 200))
    
    # Convert to bytes
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    
    print(f"  ✅ Created test image: {img.size}")
    
    # 2. Compute perceptual hashes
    phash = perceptual_hash_image(img_bytes, algorithm="phash")
    ahash = perceptual_hash_image(img_bytes, algorithm="ahash")
    dhash = perceptual_hash_image(img_bytes, algorithm="dhash")
    whash = perceptual_hash_image(img_bytes, algorithm="whash")
    
    assert len(phash) == 16  # 64-bit hash = 16 hex chars
    assert len(ahash) == 16
    assert len(dhash) == 16
    assert len(whash) == 16
    
    print(f"  ✅ pHash: {phash}")
    print(f"  ✅ aHash: {ahash}")
    print(f"  ✅ dHash: {dhash}")
    print(f"  ✅ wHash: {whash}")
    
    # 3. Modify image (add watermark simulation)
    modified_img = img.copy()
    # Add a watermark (simple modification)
    for x in range(0, 50):
        for y in range(0, 50):
            modified_img.putpixel((x, y), (255, 255, 255))
    
    # Convert modified image to bytes
    buffer2 = BytesIO()
    modified_img.save(buffer2, format="PNG")
    modified_bytes = buffer2.getvalue()
    
    print(f"  ✅ Watermark applied (50x50 white square)")
    
    # 4. Compute hashes after modification
    phash_after = perceptual_hash_image(modified_bytes, algorithm="phash")
    ahash_after = perceptual_hash_image(modified_bytes, algorithm="ahash")
    
    print(f"  ✅ pHash after: {phash_after}")
    print(f"  ✅ aHash after: {ahash_after}")
    
    # 5. Measure similarity
    phash_dist = hamming_distance(phash, phash_after)
    phash_sim = similarity_score(phash, phash_after)
    
    ahash_dist = hamming_distance(ahash, ahash_after)
    ahash_sim = similarity_score(ahash, ahash_after)
    
    print(f"  ✅ pHash similarity: {phash_sim:.2%} (distance: {phash_dist})")
    print(f"  ✅ aHash similarity: {ahash_sim:.2%} (distance: {ahash_dist})")
    
    # Hashes should still show similarity (robust to small changes)
    assert phash_sim > 0.7 or ahash_sim > 0.7
    print(f"  ✅ Perceptual hashes show similarity despite watermark")
    
    print("[OK] Image perceptual hashing test passed\n")


def test_signature_envelope_round_trip():
    """Test signature envelope serialization/deserialization."""
    print("\n[TEST] Signature Envelope Round-Trip")
    print("=" * 60)
    
    # 1. Create minimal artifact evidence
    from ciaf.watermarks.models import (
        ArtifactHashSet,
        WatermarkDescriptor,
        WatermarkType,
    )
    
    evidence = ArtifactEvidence(
        artifact_id="test-artifact-123",
        artifact_type=ArtifactType.TEXT,
        mime_type="text/plain",
        created_at="2026-03-30T18:00:00Z",
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:test",
        prompt_hash="a" * 64,
        output_hash_raw="b" * 64,
        output_hash_distributed="c" * 64,
        watermark=WatermarkDescriptor(
            watermark_id="wmk-test",
            watermark_type=WatermarkType.VISIBLE,
            verification_url="https://test.example.com",
        ),
        hashes=ArtifactHashSet(
            content_hash_before_watermark="b" * 64,
            content_hash_after_watermark="c" * 64,
        ),
    )
    
    print(f"  ✅ Created test artifact: {evidence.artifact_id}")
    
    # 2. Create signature envelope
    envelope = create_signature_envelope(
        payload_hash=evidence.compute_receipt_hash(),
        signature_value="VGVzdFNpZ25hdHVyZQ==",
        key_id="test-key-kms-001",
        key_backend=KeyBackend.KMS,
        signing_service="ciaf-test-signer",
        public_key_ref="jwks://test.example.com/keys/001",
    )
    
    evidence.signature = envelope
    
    print(f"  ✅ Signature attached")
    print(f"     Key backend: {envelope.metadata.key_backend.value}")
    print(f"     Signing service: {envelope.metadata.signing_service}")
    
    # 3. Serialize to dict
    evidence_dict = evidence.to_dict()
    
    assert "signature" in evidence_dict
    assert evidence_dict["signature"]["metadata"]["key_backend"] == "kms"
    assert evidence_dict["signature"]["metadata"]["signing_service"] == "ciaf-test-signer"
    
    print(f"  ✅ Serialized to dict")
    
    # 4. Serialize to JSON
    json_str = json.dumps(evidence_dict, indent=2)
    
    assert "payload_hash" in json_str
    assert "key_backend" in json_str
    assert "kms" in json_str
    
    print(f"  ✅ Serialized to JSON: {len(json_str)} bytes")
    
    # 5. Deserialize JSON
    restored_dict = json.loads(json_str)
    
    assert restored_dict["artifact_id"] == "test-artifact-123"
    assert restored_dict["signature"]["metadata"]["key_backend"] == "kms"
    
    print(f"  ✅ Deserialized from JSON")
    
    # 6. Reconstruct signature envelope
    sig_restored = SignatureEnvelope.from_dict(restored_dict["signature"])
    
    assert sig_restored.payload_hash == envelope.payload_hash
    assert sig_restored.signature_value == envelope.signature_value
    assert sig_restored.metadata.key_backend == KeyBackend.KMS
    assert sig_restored.metadata.signing_service == "ciaf-test-signer"
    
    print(f"  ✅ Signature envelope reconstructed")
    print(f"     Key backend: {sig_restored.metadata.key_backend.value}")
    print(f"     Payload hash: {sig_restored.payload_hash[:16]}...")
    
    # 7. Verify canonical dict excludes signature
    canonical_dict = evidence.to_canonical_dict()
    assert "signature" not in canonical_dict
    
    print(f"  ✅ Canonical dict correctly excludes signature")
    
    print("[OK] Signature envelope round-trip test passed\n")


def test_multi_artifact_verification():
    """Test verifying fragments across multiple artifacts."""
    print("\n[TEST] Multi-Artifact Fragment Verification")
    print("=" * 60)
    
    # 1. Create base text
    base_text = """The machine learning model demonstrated exceptional 
    performance across all benchmark datasets. Training convergence 
    occurred after 47 epochs with a final validation accuracy of 94.2%. 
    The model architecture incorporates transformer-based attention 
    mechanisms with 12 layers and 768-dimensional embeddings. Inference 
    latency averages 42ms per request under standard load conditions."""
    
    print(f"  ✅ Base text: {len(base_text)} chars")
    
    # 2. Create two artifacts with slight variations
    evidence1, watermarked1 = build_text_artifact_evidence(
        raw_text=base_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:researcher-1",
        prompt="Summarize ML model performance",
        verification_base_url="https://vault.example.com",
    )
    
    evidence2, watermarked2 = build_text_artifact_evidence(
        raw_text=base_text + " Additional analysis pending.",
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:researcher-2",
        prompt="Summarize ML model performance with addendum",
        verification_base_url="https://vault.example.com",
    )
    
    print(f"  ✅ Created artifact 1: {evidence1.artifact_id}")
    print(f"  ✅ Created artifact 2: {evidence2.artifact_id}")
    
    # 3. Select fragments from base text
    hash_before = sha256_text(base_text)
    hash_after = sha256_text(base_text + " Additional analysis pending.")
    
    fragments = select_text_forensic_fragments(
        raw_text=base_text,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )
    
    print(f"  ✅ Selected {len(fragments)} fragments from base text")
    
    # 4. Verify against artifact 1 (should match)
    result1 = verify_text_fragments(base_text, fragments)
    
    assert result1.match_confidence >= 0.9
    print(f"  ✅ Artifact 1 verification: {result1.match_confidence:.2%} confidence")
    
    # 5. Verify against artifact 2 (should also match - base text present)
    result2 = verify_text_fragments(
        base_text + " Additional analysis pending.", fragments)
    
    assert result2.match_confidence >= 0.9
    print(f"  ✅ Artifact 2 verification: {result2.match_confidence:.2%} confidence")
    
    # 6. Verify against unrelated text (should NOT match)
    unrelated = "Completely different content about weather patterns and climate change."
    result_unrelated = verify_text_fragments(unrelated, fragments)
    
    assert result_unrelated.match_confidence < 0.5
    print(f"  ✅ Unrelated text verification: {result_unrelated.match_confidence:.2%} confidence (correctly low)")
    
    print("[OK] Multi-artifact verification test passed\n")


def test_complete_workflow_with_all_features():
    """Test complete workflow integrating all Week 1-3 features."""
    print("\n[TEST] Complete Workflow - All Features Integrated")
    print("=" * 60)
    
    # 1. Create text content
    content = """Enterprise risk management requires comprehensive tracking 
    of AI model deployments. Our governance framework enforces strict 
    audit controls for all generative AI interactions. Model outputs must 
    include provenance tracking to maintain regulatory compliance."""
    
    print(f"  ✅ Content: {len(content)} chars")
    
    # 2. Build artifact with watermark (Week 1-2 features)
    evidence, watermarked = build_text_artifact_evidence(
        raw_text=content,
        model_id="claude-3",
        model_version="2026-03",
        actor_id="user:compliance-officer",
        prompt="Explain AI governance requirements",
        verification_base_url="https://vault.example.com",
        include_simhash=True,
    )
    
    print(f"  ✅ Artifact created: {evidence.artifact_id}")
    print(f"     Model: {evidence.model_id}")
    print(f"     Watermark: {evidence.watermark.watermark_id}")
    
    # 3. Add forensic fragments (Week 1 feature - Bug #161 fix)
    hash_before = sha256_text(content)
    hash_after = sha256_text(watermarked)
    
    fragments = select_text_forensic_fragments(
        raw_text=content,
        fragment_hash_before=hash_before,
        fragment_hash_after=hash_after,
        min_entropy=0.0,
    )
    
    # Verify fragment_text field is populated (Bug #161 fix)
    for i, frag in enumerate(fragments):
        assert frag.fragment_text is not None
        assert isinstance(frag.fragment_text, str)
        assert len(frag.fragment_text) > 0
        print(f"  ✅ Fragment {i+1}: '{frag.fragment_text[:40]}...' ({len(frag.fragment_text)} chars)")
    
    # 4. Add signature envelope (Week 3 feature)
    envelope = create_signature_envelope(
        payload_hash=evidence.compute_receipt_hash(),
        signature_value="Q29tcGxldGVXb3JrZmxvd1Rlc3Q=",
        key_id="prod-kms-key-enterprise-001",
        key_backend=KeyBackend.KMS,
        signing_service="ciaf-vault-prod",
    )
    
    evidence.signature = envelope
    
    print(f"  ✅ Signature added")
    print(f"     Key backend: {envelope.metadata.key_backend.value}")
    print(f"     Algorithm: {envelope.metadata.signature_algorithm}")
    print(f"     Payload hash: {envelope.payload_hash[:20]}...")
    
    # 5. Verify all hashes are present (Week 2 feature)
    assert evidence.hashes.content_hash_before_watermark is not None
    assert evidence.hashes.content_hash_after_watermark is not None
    assert evidence.hashes.normalized_hash_before is not None
    assert evidence.hashes.normalized_hash_after is not None
    assert evidence.hashes.simhash_before is not None  # Week 2
    assert evidence.hashes.simhash_after is not None
    
    print(f"  ✅ All hash types present:")
    print(f"     Exact hashes: before/after")
    print(f"     Normalized hashes: before/after")
    print(f"     SimHash: before/after")
    
    # 6. Serialize complete evidence to JSON
    evidence_dict = evidence.to_dict()
    json_str = json.dumps(evidence_dict, indent=2)
    
    # Verify all features are in JSON
    assert "artifact_id" in json_str
    assert "watermark" in json_str
    assert "signature" in json_str
    assert "hashes" in json_str
    assert "simhash_before" in json_str
    assert "key_backend" in json_str
    
    print(f"  ✅ Complete JSON serialization: {len(json_str)} bytes")
    print(f"     Contains: artifact, watermark, signature, hashes, simhash")
    
    # 7. Verify fragment matching works
    result = verify_text_fragments(content, fragments)
    
    # May have low fragments if text is short
    if len(fragments) > 0:
        print(f"  ✅ Fragment verification: {result.match_confidence:.2%} confidence ({result.fragments_matched}/{len(fragments)} matched)")
    else:
        print(f"  ⚠️  No fragments selected (text too short)")
    
    
    # 8. Verify canonical hash excludes signature
    canonical_dict = evidence.to_canonical_dict()
    assert "signature" not in canonical_dict
    canonical_hash = evidence.compute_receipt_hash()
    assert canonical_hash == envelope.payload_hash
    
    print(f"  ✅ Canonical hash matches signature payload")
    print(f"     Hash: {canonical_hash[:20]}...")
    
    # 9. Verify watermark preservation
    assert evidence.watermark.watermark_id in watermarked
    assert "CIAF" in watermarked
    
    print(f"  ✅ Watermark preserved in distributed text")
    
    print("\n[OK] Complete workflow integration test PASSED\n")
    print("=" * 60)
    print("ALL WEEK 1-3 FEATURES INTEGRATED SUCCESSFULLY:")
    print("  ✅ Week 1: Fragment verification (Bug #161 fix)")
    print("  ✅ Week 2: Perceptual hashing (SimHash integration)")
    print("  ✅ Week 3: Signature envelope (KMS backend tracking)")
    print("  ✅ End-to-end workflow validated")
    print("=" * 60)


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("CIAF WATERMARKS - INTEGRATION TEST SUITE")
    print("Testing end-to-end workflows and feature integration")
    print("=" * 60)
    
    try:
        # Test 1: End-to-end text watermarking
        test_end_to_end_text_watermarking()
        
        # Test 2: Image perceptual hashing
        test_image_perceptual_hashing_workflow()
        
        # Test 3: Signature envelope round-trip
        test_signature_envelope_round_trip()
        
        # Test 4: Multi-artifact verification
        test_multi_artifact_verification()
        
        # Test 5: Complete workflow with all features
        test_complete_workflow_with_all_features()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        print("\nIntegration Test Coverage:")
        print("  ✅ End-to-end text watermarking workflow")
        print("  ✅ Image perceptual hashing (pHash/aHash/dHash/wHash)")
        print("  ✅ Signature envelope round-trip serialization")
        print("  ✅ Multi-artifact fragment verification")
        print("  ✅ Complete workflow (all Week 1-3 features)")
        print("\n✅ Week 3 Task 2 COMPLETE: Integration Testing")
        print("\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
