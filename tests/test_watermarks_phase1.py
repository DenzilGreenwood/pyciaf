"""
CIAF Watermarks - Phase 1 Integration Tests

Tests for image and PDF watermarking features.

Test Coverage:
- Image visual watermarking
- Image perceptual hashing
- QR code generation
- PDF metadata watermarking
- Dual-state hashing verification
- Watermark removal detection

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False


def create_test_image(width=400, height=300, color=(100, 150, 200)):
    """Create a simple test image."""
    if not PIL_AVAILABLE:
        return None
    img = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_test_pdf():
    """Create a simple test PDF."""
    if not PYPDF_AVAILABLE:
        return None

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "This is AI-generated content for testing.")
    c.drawString(100, 730, "Created by CIAF Framework watermarking tests.")
    c.showPage()
    c.save()
    return buffer.getvalue()


def test_image_visual_watermark():
    """Test 1: Image visual watermarking with dual-state hashing."""
    print("\n[TEST 1] Image Visual Watermarking... ", end="", flush=True)

    if not PIL_AVAILABLE:
        print("[SKIP] PIL not available")
        return False

    try:
        from ciaf.watermarks import (
            build_image_artifact_evidence,
            ImageWatermarkSpec,
            sha256_bytes,
        )

        # Create test image
        test_image = create_test_image()

        # Build watermarked artifact
        spec = ImageWatermarkSpec(
            mode="visual",
            text="AI Generated Test",
            opacity=0.5,
            position="bottom_right",
            include_qr=False,
        )

        evidence, watermarked = build_image_artifact_evidence(
            image_bytes=test_image,
            model_id="test-model-v1",
            model_version="1.0",
            actor_id="test-user",
            prompt="Create test image",
            verification_base_url="https://test.example.com",
            watermark_spec=spec,
            include_perceptual_hashes=False,
        )

        # Verify dual-state hashing
        hash_before = sha256_bytes(test_image)
        hash_after = sha256_bytes(watermarked)

        assert evidence.hashes.content_hash_before_watermark == hash_before
        assert evidence.hashes.content_hash_after_watermark == hash_after
        assert hash_before != hash_after  # Watermark changed the content
        assert evidence.watermark.watermark_type.value == "visible"
        assert evidence.artifact_type.value == "image"
        assert len(watermarked) > len(test_image) * 0.5  # Reasonable size

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_image_perceptual_hashing():
    """Test 2: Image perceptual hashing."""
    print("[TEST 2] Image Perceptual Hashing... ", end="", flush=True)

    if not PIL_AVAILABLE or not IMAGEHASH_AVAILABLE:
        print("[SKIP] PIL or imagehash not available")
        return False

    try:
        from ciaf.watermarks import (
            compute_all_hashes,
            hamming_distance,
            similarity_score,
        )

        # Create two similar images
        img1 = create_test_image(color=(100, 150, 200))
        img2 = create_test_image(color=(105, 155, 205))  # Slightly different

        # Compute hashes
        phash1, ahash1, dhash1, whash1 = compute_all_hashes(img1)
        phash2, ahash2, dhash2, whash2 = compute_all_hashes(img2)

        # Check that hashes are computed
        assert len(phash1) == 16  # 8x8 = 64 bits = 16 hex chars
        assert len(ahash1) == 16

        # Calculate similarity
        dist = hamming_distance(phash1, phash2)
        score = similarity_score(phash1, phash2)

        # Similar images should have low distance and high similarity
        assert dist < 10  # Low hamming distance
        assert score > 0.85  # High similarity score

        print(f"[OK] PASSED (distance={dist}, similarity={score:.2f})")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_qr_code_generation():
    """Test 3: QR code generation."""
    print("[TEST 3] QR Code Generation... ", end="", flush=True)

    if not QRCODE_AVAILABLE:
        print("[SKIP] qrcode not available")
        return False

    try:
        from ciaf.watermarks import (
            make_verification_url_qr,
            make_compact_token_qr,
        )

        # Generate verification URL QR
        qr_bytes = make_verification_url_qr(
            artifact_id="test-artifact-123",
            base_url="https://vault.example.com"
        )

        # Check it's valid PNG
        assert qr_bytes.startswith(b'\x89PNG')
        assert len(qr_bytes) > 100  # Reasonable size

        # Generate compact token QR
        compact_qr = make_compact_token_qr(
            artifact_id="test-artifact-123",
            watermark_id="wmk-abc123def456",
            receipt_hash_prefix="a1b2c3d4"
        )
        assert compact_qr.startswith(b'\x89PNG')

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_image_with_qr_watermark():
    """Test 4: Combined image watermark (text + QR)."""
    print("[TEST 4] Combined Image Watermark (Text + QR)... ", end="", flush=True)

    if not PIL_AVAILABLE or not QRCODE_AVAILABLE:
        print("[SKIP] PIL or qrcode not available")
        return False

    try:
        from ciaf.watermarks import (
            build_image_artifact_evidence,
            ImageWatermarkSpec,
        )

        # Create test image
        test_image = create_test_image()

        # Build watermarked artifact with QR
        spec = ImageWatermarkSpec(
            mode="visual",
            text="AI Generated",
            opacity=0.4,
            position="bottom_right",
            include_qr=True,  # Include QR code
            qr_position="top_right",
            qr_size=80,
        )

        evidence, watermarked = build_image_artifact_evidence(
            image_bytes=test_image,
            model_id="test-model",
            model_version="1.0",
            actor_id="user",
            prompt="Test",
            verification_base_url="https://vault.example.com",
            watermark_spec=spec,
            include_perceptual_hashes=False,
        )

        # QR payload should be set
        assert evidence.watermark.qr_payload is not None
        assert "verify" in evidence.watermark.qr_payload
        assert evidence.metadata.get("has_qr_code") == True

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_image_with_perceptual_hashing():
    """Test 5: Image watermarking with perceptual hashes."""
    print("[TEST 5] Image with Perceptual Hashing... ", end="", flush=True)

    if not PIL_AVAILABLE or not IMAGEHASH_AVAILABLE:
        print("[SKIP] PIL or imagehash not available")
        return False

    try:
        from ciaf.watermarks import build_image_artifact_evidence

        test_image = create_test_image()

        evidence, watermarked = build_image_artifact_evidence(
            image_bytes=test_image,
            model_id="test-model",
            model_version="1.0",
            actor_id="user",
            prompt="Test",
            verification_base_url="https://vault.example.com",
            include_perceptual_hashes=True,  # Enable perceptual hashing
        )

        # Check perceptual hashes were computed
        assert evidence.hashes.perceptual_hash_before is not None
        assert evidence.hashes.perceptual_hash_after is not None
        assert len(evidence.fingerprints) >= 4  # phash, ahash, dhash, whash

        # Find pHash entries
        phash_before = [f for f in evidence.fingerprints if f.algorithm == "phash" and "before" in f.role]
        phash_after = [f for f in evidence.fingerprints if f.algorithm == "phash" and "after" in f.role]

        assert len(phash_before) == 1
        assert len(phash_after) == 1

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_metadata_watermark():
    """Test 6: PDF metadata watermarking."""
    print("[TEST 6] PDF Metadata Watermarking... ", end="", flush=True)

    if not PYPDF_AVAILABLE:
        print("[SKIP] pypdf/PyPDF2 not available")
        return False

    try:
        # Try to import reportlab for PDF creation
        import reportlab
    except ImportError:
        print("[SKIP] reportlab not available")
        return False

    try:
        from ciaf.watermarks import (
            build_pdf_artifact_evidence,
            extract_pdf_metadata_watermark,
            has_pdf_watermark,
            sha256_bytes,
        )

        # Create test PDF
        test_pdf = create_test_pdf()

        # Build watermarked PDF
        evidence, watermarked_pdf = build_pdf_artifact_evidence(
            pdf_bytes=test_pdf,
            model_id="pdf-gen-model",
            model_version="2.0",
            actor_id="test-system",
            prompt="Generate test PDF",
            verification_base_url="https://vault.example.com",
        )

        # Verify dual-state hashing
        hash_before = sha256_bytes(test_pdf)
        hash_after = sha256_bytes(watermarked_pdf)

        assert evidence.hashes.content_hash_before_watermark == hash_before
        assert evidence.hashes.content_hash_after_watermark == hash_after
        assert hash_before != hash_after  # Metadata changed

        # Check watermark presence
        assert has_pdf_watermark(watermarked_pdf)

        # Extract watermark
        extracted = extract_pdf_metadata_watermark(watermarked_pdf)
        assert extracted is not None
        assert extracted["watermark_id"] == evidence.watermark.watermark_id
        assert extracted["artifact_id"] == evidence.artifact_id

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_watermark_removal_detection():
    """Test 7: PDF watermark removal detection."""
    print("[TEST 7] PDF Watermark Removal Detection... ", end="", flush=True)

    if not PYPDF_AVAILABLE:
        print("[SKIP] pypdf/PyPDF2 not available")
        return False

    try:
        import reportlab
    except ImportError:
        print("[SKIP] reportlab not available")
        return False

    try:
        from ciaf.watermarks import (
            build_pdf_artifact_evidence,
            verify_pdf_artifact,
        )

        # Create and watermark PDF
        test_pdf = create_test_pdf()

        evidence, watermarked_pdf = build_pdf_artifact_evidence(
            pdf_bytes=test_pdf,
            model_id="test-model",
            model_version="1.0",
            actor_id="user",
            prompt="Test",
            verification_base_url="https://vault.example.com",
        )

        # Test 1: Verify watermarked version
        result1 = verify_pdf_artifact(watermarked_pdf, evidence)
        assert result1.exact_match_after_watermark
        assert result1.confidence == 1.0
        assert result1.watermark_present

        # Test 2: Verify original (simulates watermark removal)
        result2 = verify_pdf_artifact(test_pdf, evidence)
        assert result2.exact_match_before_watermark
        assert result2.likely_tag_removed  # Detected removal!
        assert not result2.watermark_present

        print("[OK] PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("CIAF Watermarks - Phase 1 Integration Tests")
    print("=" * 60)

    # Check dependencies
    print("\nDependency Check:")
    print(f"  PIL (Pillow): {'[OK]' if PIL_AVAILABLE else '[MISSING]'}")
    print(f"  imagehash:    {'[OK]' if IMAGEHASH_AVAILABLE else '[MISSING]'}")
    print(f"  qrcode:       {'[OK]' if QRCODE_AVAILABLE else '[MISSING]'}")
    print(f"  pypdf:        {'[OK]' if PYPDF_AVAILABLE else '[MISSING]'}")

    try:
        import reportlab
        print(f"  reportlab:    [OK]")
    except ImportError:
        print(f"  reportlab:    [MISSING]")

    # Run tests
    results = []

    results.append(test_image_visual_watermark())
    results.append(test_image_perceptual_hashing())
    results.append(test_qr_code_generation())
    results.append(test_image_with_qr_watermark())
    results.append(test_image_with_perceptual_hashing())
    results.append(test_pdf_metadata_watermark())
    results.append(test_pdf_watermark_removal_detection())

    # Summary
    passed = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r is False and r is not None)
    failed = sum(1 for r in results if r is None)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"Phase 1 Test Results: {passed} passed, {failed} failed, {skipped} skipped (of {total})")
    print("=" * 60)

    if failed == 0 and passed > 0:
        print("\n[OK] All Phase 1 tests passed!")
        return 0
    elif skipped == total:
        print("\n[WARN] All tests skipped - install dependencies")
        return 1
    else:
        print(f"\n[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
