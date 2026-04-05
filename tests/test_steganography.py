"""
Tests for LSB Steganography

Tests the invisible watermarking via LSB embedding/extraction.

Created: 2026-04-04
Author: Denzil James Greenwood
"""

import sys
from pathlib import Path
from io import BytesIO
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed. Install with: pip install Pillow")

import pytest

from ciaf.watermarks.images.steganography import (
    embed_watermark_lsb,
    extract_watermark_lsb,
    verify_lsb_watermark,
    has_lsb_watermark,
    SteganographyError,
    PIL_AVAILABLE,
)


def create_test_image(width=100, height=100, color=(128, 128, 128)):
    """Create a simple test image."""
    if not PIL_AVAILABLE:
        pytest.skip("Pillow not installed")
    
    img = Image.new('RGB', (width, height), color)
    output = BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestLSBSteganography:
    """Test LSB steganography embedding and extraction."""
    
    def test_embed_and_extract_basic(self):
        """Test basic embedding and extraction."""
        # Create test image
        image_bytes = create_test_image(200, 200)
        
        # Embed watermark
        watermark_id = f"wmk-{uuid.uuid4()}"
        verification_url = "https://vault.example.com/verify/test"
        created_at = "2026-04-04T10:00:00Z"
        
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id=watermark_id,
            verification_url=verification_url,
            created_at=created_at,
        )
        
        # Should return bytes
        assert isinstance(watermarked, bytes)
        assert len(watermarked) > 0
        
        # Extract watermark
        extracted = extract_watermark_lsb(watermarked)
        
        assert extracted is not None
        assert extracted['watermark_id'] == watermark_id
        assert extracted['verification_url'] == verification_url
        assert extracted['created_at'] == created_at
    
    def test_embed_with_artifact_id(self):
        """Test embedding with optional artifact ID."""
        image_bytes = create_test_image(200, 200)
        
        watermark_id = "wmk-test-123"
        artifact_id = "artifact-456"
        
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id=watermark_id,
            verification_url="https://example.com/verify",
            created_at="2026-04-04T10:00:00Z",
            artifact_id=artifact_id,
        )
        
        extracted = extract_watermark_lsb(watermarked)
        
        assert extracted['artifact_id'] == artifact_id
    
    def test_extract_from_non_watermarked_image(self):
        """Test extraction from image without watermark."""
        image_bytes = create_test_image(200, 200)
        
        # Should return None (no watermark)
        extracted = extract_watermark_lsb(image_bytes)
        assert extracted is None
    
    def test_has_lsb_watermark(self):
        """Test watermark detection function."""
        # Non-watermarked image
        image_bytes = create_test_image(200, 200)
        assert has_lsb_watermark(image_bytes) is False
        
        # Watermarked image
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id="test-id",
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        assert has_lsb_watermark(watermarked) is True
    
    def test_verify_lsb_watermark(self):
        """Test watermark verification."""
        image_bytes = create_test_image(200, 200)
        watermark_id = "wmk-verify-test"
        
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id=watermark_id,
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        
        # Verify with correct ID
        is_valid, data = verify_lsb_watermark(watermarked, watermark_id)
        assert is_valid is True
        assert data['watermark_id'] == watermark_id
        
        # Verify with wrong ID
        is_valid, data = verify_lsb_watermark(watermarked, "wrong-id")
        assert is_valid is False
        assert data['watermark_id'] == watermark_id  # Still returns data
    
    def test_image_too_small_for_message(self):
        """Test error when image too small for watermark."""
        # Very small image
        tiny_image = create_test_image(2, 2)  # Only 12 pixels
        
        with pytest.raises(SteganographyError, match="Image too small"):
            embed_watermark_lsb(
                image_bytes=tiny_image,
                watermark_id="test" * 100,  # Long ID to force error
                verification_url="https://example.com/" + "x" * 500,  # Long URL
                created_at="2026-04-04T10:00:00Z",
            )
    
    def test_watermark_survives_lossless_save(self):
        """Test that watermark survives PNG save/load."""
        image_bytes = create_test_image(200, 200)
        watermark_id = "wmk-lossless-test"
        
        # Embed watermark
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id=watermark_id,
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        
        # Load and re-save as PNG (lossless)
        img = Image.open(BytesIO(watermarked))
        output = BytesIO()
        img.save(output, format='PNG')
        resaved_bytes = output.getvalue()
        
        # Watermark should still be extractable
        extracted = extract_watermark_lsb(resaved_bytes)
        assert extracted is not None
        assert extracted['watermark_id'] == watermark_id
    
    def test_different_image_modes(self):
        """Test embedding on RGB and RGBA images."""
        # RGB image
        rgb_img = Image.new('RGB', (100, 100), (128, 128, 128))
        rgb_bytes = BytesIO()
        rgb_img.save(rgb_bytes, format='PNG')
        
        watermarked_rgb = embed_watermark_lsb(
            image_bytes=rgb_bytes.getvalue(),
            watermark_id="rgb-test",
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        
        extracted = extract_watermark_lsb(watermarked_rgb)
        assert extracted['watermark_id'] == "rgb-test"
        
        # RGBA image (with alpha channel)
        rgba_img = Image.new('RGBA', (100, 100), (128, 128, 128, 255))
        rgba_bytes = BytesIO()
        rgba_img.save(rgba_bytes, format='PNG')
        
        watermarked_rgba = embed_watermark_lsb(
            image_bytes=rgba_bytes.getvalue(),
            watermark_id="rgba-test",
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        
        extracted = extract_watermark_lsb(watermarked_rgba)
        assert extracted['watermark_id'] == "rgba-test"
    
    def test_checksum_validation(self):
        """Test that checksum detects corruption."""
        image_bytes = create_test_image(200, 200)
        
        watermarked = embed_watermark_lsb(
            image_bytes=image_bytes,
            watermark_id="checksum-test",
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )
        
        # Normal extraction should work
        extracted = extract_watermark_lsb(watermarked)
        assert extracted is not None
        
        # Corrupt some pixels (this might break the watermark)
        img = Image.open(BytesIO(watermarked))
        pixels = list(img.getdata())
        # Flip many bits to corrupt the checksum
        corrupted_pixels = [(p[0] ^ 0xFF, p[1] ^ 0xFF, p[2] ^ 0xFF) if i < 100 else p 
                           for i, p in enumerate(pixels)]
        img.putdata(corrupted_pixels)
        
        corrupted = BytesIO()
        img.save(corrupted, format='PNG')
        
        # Extraction should detect corruption or return None
        try:
            result = extract_watermark_lsb(corrupted.getvalue())
            # If it extracts anything, checksum should fail
            if result is not None:
                pytest.fail("Should have detected corruption")
        except SteganographyError:
            pass  # Expected - checksum mismatch


def test_steganography_unavailable_without_pillow():
    """Test graceful failure when Pillow not installed."""
    if PIL_AVAILABLE:
        pytest.skip("Pillow is installed")
    
    # Should raise error
    with pytest.raises(SteganographyError, match="Pillow not installed"):
        embed_watermark_lsb(
            image_bytes=b"fake",
            watermark_id="test",
            verification_url="https://example.com",
            created_at="2026-04-04T10:00:00Z",
        )


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("CIAF WATERMARKS - LSB STEGANOGRAPHY TEST SUITE")
    print("=" * 70)
    
    if not PIL_AVAILABLE:
        print("\n⚠️  Pillow not installed - tests will be skipped")
        print("Install with: pip install Pillow\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
