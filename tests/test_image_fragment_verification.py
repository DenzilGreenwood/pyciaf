"""
Tests for Image Fragment Verification

Tests the spatial patch matching for forensic image verification.

Created: 2026-04-04
Author: Denzil James Greenwood
"""

import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow/numpy not installed. Install with: pip install Pillow numpy")

import pytest

from ciaf.watermarks.fragment_selection import (
    select_image_forensic_patches,
    compute_image_patch_entropy,
)
from ciaf.watermarks.fragment_verification import (
    verify_image_fragments,
    verify_image_fragment_spatial_search,
)
from ciaf.watermarks.models import ImageForensicFragment


def create_test_image_with_patterns(width=400, height=400):
    """Create a test image with distinct visual patterns for patch selection."""
    if not PIL_AVAILABLE:
        pytest.skip("Pillow not installed")
    
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Add distinct patterns in different regions
    # Top-left: Red rectangle
    draw.rectangle([20, 20, 120, 120], fill=(255, 50, 50), outline=(0, 0, 0), width=2)
    
    # Top-right: Blue circle
    draw.ellipse([280, 20, 380, 120], fill=(50, 50, 255), outline=(0, 0, 0), width=2)
    
    # Bottom-left: Green triangle
    draw.polygon([(20, 380), (120, 380), (70, 280)], fill=(50, 255, 50), outline=(0, 0, 0))
    
    # Bottom-right: Yellow pattern
    for i in range(0, 100, 10):
        draw.line([(280 + i, 280), (280 + i, 380)], fill=(255, 255, 0), width=2)
        draw.line([(280, 280 + i), (380, 280 + i)], fill=(255, 255, 0), width=2)
    
    # Center: Complex pattern (high entropy)
    for i in range(10):
        x = 150 + (i * 10)
        y = 150 + (i * 10)
        draw.rectangle([x, y, x + 8, y + 8], fill=(i * 25, 100 + i * 15, 200 - i * 10))
    
    output = BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestImageFragmentSelection:
    """Test image forensic fragment selection."""
    
    def test_select_image_patches_basic(self):
        """Test basic patch selection from image."""
        image_bytes = create_test_image_with_patterns(400, 400)
        
        # Select 4 patches
        patches = select_image_forensic_patches(
            image_bytes=image_bytes,
            num_patches=4,
            patch_size=64,
            min_entropy=0.3
        )
        
        assert len(patches) > 0
        assert len(patches) <= 4
        
        # Check that patches have required fields
        for patch in patches:
            assert patch.fragment_id.startswith("img_patch_")
            assert patch.fragment_type == "image_patch"
            assert patch.entropy_score >= 0.3
            assert len(patch.patch_hash_before) == 64  # SHA-256 hash
            assert patch.region_coordinates[2] == 64  # Width
            assert patch.region_coordinates[3] == 64  # Height
    
    def test_patch_entropy_high_complexity_region(self):
        """Test entropy computation for complex vs simple regions."""
        image_bytes = create_test_image_with_patterns(400, 400)
        
        # High entropy region (center with complex pattern)
        entropy_high = compute_image_patch_entropy(image_bytes, 150, 150, 64, 64)
        
        # Low entropy region (blank gray background)
        entropy_low = compute_image_patch_entropy(image_bytes, 0, 200, 64, 64)
        
        # Complex region should have higher entropy
        assert entropy_high > entropy_low
        print(f"High entropy: {entropy_high:.3f}, Low entropy: {entropy_low:.3f}")
    
    def test_patches_avoid_blank_regions(self):
        """Test that patch selection avoids low-entropy regions."""
        # Create image with mostly blank space
        img = Image.new('RGB', (400, 400), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Only one high-entropy region (top-left corner)
        draw.rectangle([20, 20, 100, 100], fill=(255, 0, 0))
        for i in range(10):
            draw.line([(20 + i*8, 20), (20 + i*8, 100)], fill=(0, 255, 0), width=1)
        
        output = BytesIO()
        img.save(output, format='PNG')
        image_bytes = output.getvalue()
        
        patches = select_image_forensic_patches(
            image_bytes=image_bytes,
            num_patches=3,
            patch_size=64,
            min_entropy=0.4
        )
        
        # Should find at least 1 patch in the complex region
        assert len(patches) >= 1
        
        # All selected patches should have decent entropy
        for patch in patches:
            assert patch.entropy_score >= 0.4


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestImageFragmentVerification:
    """Test image forensic fragment verification."""
    
    def test_verify_exact_match(self):
        """Test verification of exact image match."""
        # Create original image
        original_bytes = create_test_image_with_patterns(400, 400)
        
        # Select fragments with lower entropy threshold
        fragments = select_image_forensic_patches(
            image_bytes=original_bytes,
            num_patches=4,
            patch_size=64,
            min_entropy=0.3,  # Lower threshold for test images
        )
        
        assert len(fragments) > 0, "Should find at least 1 fragment"
        
        # Verify against same image
        result = verify_image_fragments(original_bytes, fragments)
        
        assert result.total_fragments_checked == len(fragments)
        assert result.fragments_matched >= 2  # Should match most/all patches
        assert result.match_confidence >= 0.65
        print(f"Matched: {result.fragments_matched}/{result.total_fragments_checked}")
        print(f"Confidence: {result.match_confidence:.2f}")
    
    def test_verify_single_fragment_spatial_search(self):
        """Test spatial search for a single fragment."""
        # Create image and select one patch
        original_bytes = create_test_image_with_patterns(400, 400)
        fragments = select_image_forensic_patches(original_bytes, num_patches=1, min_entropy=0.3)
        
        assert len(fragments) == 1, "Should find exactly 1 fragment"
        fragment = fragments[0]
        
        # Search for this fragment in the same image
        match = verify_image_fragment_spatial_search(original_bytes, fragment)
        
        assert match is not None
        (x, y), confidence = match
        
        # Should find exact match
        assert confidence == 1.0
        
        # Position should be close to original (within patch stride)
        orig_x, orig_y, _, _ = fragment.region_coordinates
        assert abs(x - orig_x) < 20  # Allow some tolerance for stride
        assert abs(y - orig_y) < 20
    
    def test_verify_modified_image_no_match(self):
        """Test that modified image doesn't match."""
        # Create original
        original_bytes = create_test_image_with_patterns(400, 400)
        fragments = select_image_forensic_patches(original_bytes, num_patches=3, min_entropy=0.3)
        
        # Create completely different image
        different_img = Image.new('RGB', (400, 400), color=(50, 50, 50))
        draw = ImageDraw.Draw(different_img)
        draw.ellipse([100, 100, 300, 300], fill=(100, 100, 100))
        
        different_bytes = BytesIO()
        different_img.save(different_bytes, format='PNG')
        
        # Verify against different image
        result = verify_image_fragments(different_bytes.getvalue(), fragments)
        
        # Should find no matches
        assert result.fragments_matched == 0
        assert result.match_confidence < 0.5
    
    def test_verify_cropped_image_partial_match(self):
        """Test verification of cropped image (partial matches)."""
        # Create original
        original_bytes = create_test_image_with_patterns(400, 400)
        fragments = select_image_forensic_patches(original_bytes, num_patches=4, min_entropy=0.3)
        
        # Crop image (keep top-left quadrant)
        original_img = Image.open(BytesIO(original_bytes))
        cropped_img = original_img.crop((0, 0, 250, 250))
        
        cropped_bytes = BytesIO()
        cropped_img.save(cropped_bytes, format='PNG')
        
        # Verify cropped image
        result = verify_image_fragments(cropped_bytes.getvalue(), fragments)
        
        # Should match some but not all patches
        # (only patches from top-left quadrant will match)
        print(f"Cropped matches: {result.fragments_matched}/{result.total_fragments_checked}")
        
        # At least some patches should be gone
        assert result.fragments_matched < result.total_fragments_checked
    
    def test_verification_with_watermark_added(self):
        """Test that fragments can match even with watermark added."""
        # Create original
        original_bytes = create_test_image_with_patterns(400, 400)
        
        # Select fragments from original
        fragments = select_image_forensic_patches(original_bytes, num_patches=3, min_entropy=0.3)
        
        # Add watermark (text overlay)
        img = Image.open(BytesIO(original_bytes))
        draw = ImageDraw.Draw(img)
        draw.rectangle([150, 370, 390, 395], fill=(255, 255, 255, 128))
        draw.text((160, 375), "AI Generated Watermark", fill=(0, 0, 0))
        
        watermarked_bytes = BytesIO()
        img.save(watermarked_bytes, format='PNG')
        
        # Verify watermarked image
        result = verify_image_fragments(watermarked_bytes.getvalue(), fragments)
        
        # Most patches should still match (watermark only affects small region)
        print(f"With watermark: {result.fragments_matched}/{result.total_fragments_checked}")
        
        # Should have at least some matches from unaffected regions
        assert result.fragments_matched >= 1
    
    def test_verification_confidence_levels(self):
        """Test confidence scoring based on number of matches."""
        original_bytes = create_test_image_with_patterns(400, 400)
        fragments = select_image_forensic_patches(original_bytes, num_patches=4, min_entropy=0.3)
        
        # Full match → high confidence
        result_full = verify_image_fragments(original_bytes, fragments)
        assert result_full.match_confidence >= 0.9
        assert result_full.legal_defensibility == "high"
        
        # Partial match → medium confidence
        # (manually create scenario with 1 match)
        if len(fragments) > 0:
            # Verify with just 1 fragment (should give medium confidence if matched)
            result_partial = verify_image_fragments(original_bytes, fragments[:1])
            
            if result_partial.fragments_matched == 1:
                assert result_partial.match_confidence >= 0.6
                assert result_partial.legal_defensibility == "medium"
    
    def test_empty_fragment_list(self):
        """Test verification with no fragments."""
        original_bytes = create_test_image_with_patterns(400, 400)
        
        result = verify_image_fragments(original_bytes, [])
        
        assert result.total_fragments_checked == 0
        assert result.fragments_matched == 0
        assert result.match_confidence == 0.0


def test_image_fragment_unavailable_without_pillow():
    """Test graceful degradation when Pillow not available."""
    if PIL_AVAILABLE:
        pytest.skip("Pillow is installed")
    
    # Should return empty list when Pillow not available
    result = select_image_forensic_patches(b"fake", num_patches=4)
    assert result == []


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("CIAF WATERMARKS - IMAGE FRAGMENT VERIFICATION TEST SUITE")
    print("=" * 70)
    
    if not PIL_AVAILABLE:
        print("\n⚠️  Pillow/numpy not installed - tests will be skipped")
        print("Install with: pip install Pillow numpy\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
