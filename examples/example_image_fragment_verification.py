"""
Image Fragment Verification Examples
=====================================

Demonstrates forensic image verification using high-entropy fragments to detect:
- Image splicing
- Cropping
- Content alteration
- Watermark addition/removal

This module shows how to:
1. Select distinctive fragments from an original image
2. Verify fragments in suspect images (exact match)
3. Detect partial matches (cropped images)
4. Handle modified images (watermarked but same content)
"""

import os
import sys
from io import BytesIO
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

from ciaf.watermarks.fragment_selection import select_image_forensic_patches
from ciaf.watermarks.fragment_verification import verify_image_fragments


def create_test_image(width=400, height=300, complexity="medium"):
    """
    Create a test image with varying complexity regions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        complexity: 'low', 'medium', or 'high'
    
    Returns:
        PIL Image object
    """
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    
    if complexity == "low":
        # Simple gradient
        for y in range(height):
            color = int(255 * y / height)
            draw.rectangle([(0, y), (width, y + 1)], fill=(color, color, color))
    
    elif complexity == "medium":
        # Geometric patterns
        draw.rectangle([(50, 50), (150, 100)], fill="red", outline="black")
        draw.ellipse([(200, 50), (350, 150)], fill="blue", outline="yellow")
        draw.polygon([(100, 200), (200, 250), (50, 250)], fill="green", outline="black")
        
        # Add text for high entropy
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((20, 20), "FORENSIC TEST IMAGE", fill="black", font=font)
    
    else:  # high
        # Random noise pattern (high entropy)
        import numpy as np
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "HIGH ENTROPY", fill="white")
    
    return img


def example_1_exact_match_verification():
    """
    Example 1: Verify exact match
    
    Demonstrates that fragments from original image are found in identical copy.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Exact Match Verification")
    print("=" * 70)
    
    # Create original image
    original_img = create_test_image(400, 300, "medium")
    print("✓ Created original test image (400x300)")
    
    # Save to BytesIO
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    # Select distinctive fragments
    fragments = select_image_forensic_patches(
        image_bytes=img_bytes.getvalue(),
        num_patches=5,
        patch_size=64,
        min_entropy=0.3
    )
    print(f"✓ Selected {len(fragments)} distinctive fragments")
    for frag in fragments:
        print(f"  - Fragment at {frag.region_coordinates[:2]}, "
              f"entropy={frag.entropy_score:.4f}")
    
    # Verify against identical image
    verification = verify_image_fragments(img_bytes.getvalue(), fragments)
    
    print(f"\n📊 Verification Results:")
    print(f"  Total fragments: {verification.total_fragments_checked}")
    print(f"  Matched: {verification.fragments_matched}")
    print(f"  Confidence: {verification.match_confidence:.2%}")
    print(f"  Legal defensibility: {verification.legal_defensibility}")
    
    assert verification.fragments_matched == len(fragments), "All fragments should match!"
    print("\n✅ SUCCESS: All fragments matched as expected")


def example_2_cropped_image_detection():
    """
    Example 2: Detect cropped image
    
    Shows partial fragment matching when image is cropped.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Cropped Image Detection")
    print("=" * 70)
    
    # Create original
    original_img = create_test_image(400, 300, "medium")
    print("✓ Created original image (400x300)")
    
    # Select fragments from original
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    fragments = select_image_forensic_patches(
        image_bytes=img_bytes.getvalue(),
        num_patches=6,
        patch_size=64,
        min_entropy=0.3
    )
    print(f"✓ Selected {len(fragments)} fragments from original")
    
    # Create cropped version (remove right half)
    cropped_img = original_img.crop((0, 0, 200, 300))
    print("✓ Created cropped version (200x300, left half only)")
    
    cropped_bytes = BytesIO()
    cropped_img.save(cropped_bytes, format="PNG")
    cropped_bytes.seek(0)
    
    # Verify against cropped image
    verification = verify_image_fragments(cropped_bytes.getvalue(), fragments)
    
    print(f"\n📊 Verification Results:")
    print(f"  Total fragments: {verification.total_fragments_checked}")
    print(f"  Matched: {verification.fragments_matched}")
    print(f"  Match rate: {verification.fragments_matched / verification.total_fragments_checked:.1%}")
    print(f"  Confidence: {verification.match_confidence:.2%}")
    print(f"  Legal defensibility: {verification.legal_defensibility}")
    
    # Expect partial match
    assert 0 < verification.fragments_matched < len(fragments), \
        "Should have partial match for cropped image"
    print(f"\n✅ SUCCESS: Partial match detected "
          f"({verification.fragments_matched}/{len(fragments)} fragments)")


def example_3_modified_image_detection():
    """
    Example 3: Detect modified image
    
    Shows how adding watermark affects fragment matching.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Modified Image Detection")
    print("=" * 70)
    
    # Create original
    original_img = create_test_image(400, 300, "medium")
    print("✓ Created original image")
    
    # Select fragments
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    fragments = select_image_forensic_patches(
        image_bytes=img_bytes.getvalue(),
        num_patches=5,
        patch_size=64,
        min_entropy=0.3
    )
    print(f"✓ Selected {len(fragments)} fragments")
    
    # Create modified version (add visible overlay)
    modified_img = original_img.copy()
    draw = ImageDraw.Draw(modified_img, "RGBA")
    draw.rectangle(
        [(50, 50), (350, 100)],
        fill=(255, 0, 0, 100),  # Semi-transparent red
        outline="red"
    )
    draw.text((60, 70), "MODIFIED IMAGE", fill="white")
    print("✓ Created modified version (added overlay)")
    
    modified_bytes = BytesIO()
    modified_img.save(modified_bytes, format="PNG")
    modified_bytes.seek(0)
    
    # Verify
    verification = verify_image_fragments(modified_bytes.getvalue(), fragments)
    
    print(f"\n📊 Verification Results:")
    print(f"  Total fragments: {verification.total_fragments_checked}")
    print(f"  Matched: {verification.fragments_matched}")
    print(f"  Match rate: {verification.fragments_matched / verification.total_fragments_checked:.1%}")
    print(f"  Confidence: {verification.match_confidence:.2%}")
    
    # Fragment details
    print("\n📝 Fragment Match Details:")
    for result in verification.forensic_matches:
        status = "✓ MATCHED" if result.matched else "✗ NO MATCH"
        print(f"  {status}: Fragment {result.fragment_id}")
        if result.matched:
            print(f"    Match position: {result.match_position}, "
                  f"Confidence: {result.confidence:.2%}")
            if result.match_details:
                print(f"    Details: {result.match_details}")
    
    if verification.fragments_matched < len(fragments):
        print(f"\n⚠️  TAMPERING DETECTED: Only {verification.fragments_matched}/{len(fragments)} "
              f"fragments matched - image may be modified")
    else:
        print("\n✅ All fragments matched despite visual changes (overlay didn't affect distinctive regions)")


def example_4_spatial_search_tolerance():
    """
    Example 4: Spatial search with shifted image
    
    Demonstrates that spatial search can find fragments even if 
    image is slightly shifted or re-encoded.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Spatial Search Tolerance")
    print("=" * 70)
    
    # Create original with complex pattern
    original_img = create_test_image(400, 300, "medium")
    print("✓ Created original image")
    
    # Select fragments
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    fragments = select_image_forensic_patches(
        image_bytes=img_bytes.getvalue(),
        num_patches=4,
        patch_size=64,
        min_entropy=0.3
    )
    print(f"✓ Selected {len(fragments)} fragments")
    
    # Create padded version (add 10px border)
    padded_img = Image.new("RGB", (420, 320), color="white")
    padded_img.paste(original_img, (10, 10))
    print("✓ Created padded version (10px border added)")
    
    padded_bytes = BytesIO()
    padded_img.save(padded_bytes, format="PNG")
    padded_bytes.seek(0)
    
    # Verify with spatial search
    verification = verify_image_fragments(padded_bytes.getvalue(), fragments)
    
    print(f"\n📊 Verification Results:")
    print(f"  Matched: {verification.fragments_matched}/{verification.total_fragments_checked}")
    print(f"  Confidence: {verification.match_confidence:.2%}")
    
    print("\n📍 Fragment Positions:")
    for result in verification.forensic_matches:
        if result.matched:
            print(f"  Fragment {result.fragment_id}: "
                  f"Found at position {result.match_position}")
    
    if verification.fragments_matched > 0:
        print(f"\n✅ Spatial search found fragments despite 10px shift")
    else:
        print(f"\n❌ No matches - padding may have disrupted fragments")


def example_5_save_and_export():
    """
    Example 5: Save fragments and verification report
    
    Shows how to save fragments and export verification results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Save Fragments and Export Report")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "fragment_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Create test image
    original_img = create_test_image(400, 300, "high")
    original_path = output_dir / "original.png"
    original_img.save(original_path)
    print(f"✓ Saved original: {original_path.name}")
    
    # Select fragments
    # Select fragments
    with open(original_path, "rb") as f:
        fragments = select_image_forensic_patches(
            image_bytes=f.read(),
            num_patches=8,
            patch_size=64,
            min_entropy=0.4
        )
    print(f"✓ Selected {len(fragments)} high-entropy fragments")
    
    # Visualize fragments on image
    vis_img = original_img.copy()
    draw = ImageDraw.Draw(vis_img)
    
    for frag in fragments:
        x, y, w, h = frag.region_coordinates
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        draw.text((x + 2, y + 2), frag.fragment_id.split("_")[-1], fill="yellow")
    
    vis_path = output_dir / "fragments_visualized.png"
    vis_img.save(vis_path)
    print(f"✓ Saved visualization: {vis_path.name}")
    
    # Export fragment data
    fragment_report = output_dir / "fragments_report.txt"
    with open(fragment_report, "w") as f:
        f.write("IMAGE FRAGMENT VERIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Fragments: {len(fragments)}\n\n")
        
        for frag in fragments:
            f.write(f"Fragment: {frag.fragment_id}\n")
            f.write(f"  Position: {frag.region_coordinates[:2]}\n")
            f.write(f"  Size: {frag.region_coordinates[2:]} pixels\n")
            f.write(f"  Entropy: {frag.entropy_score:.6f}\n")
            f.write(f"  Hash (before): {frag.patch_hash_before[:16]}...\n")
            f.write(f"  Hash (after): {frag.patch_hash_after[:16]}...\n\n")
    
    print(f"✓ Saved report: {fragment_report.name}")
    
    # Verify against original
    with open(original_path, "rb") as f:
        verification = verify_image_fragments(f.read(), fragments)
    
    # Export verification results
    verify_report = output_dir / "verification_results.txt"
    with open(verify_report, "w") as f:
        f.write("FRAGMENT VERIFICATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Fragments: {verification.total_fragments_checked}\n")
        f.write(f"Matched: {verification.fragments_matched}\n")
        f.write(f"Confidence: {verification.match_confidence:.2%}\n")
        f.write(f"Legal Defensibility: {verification.legal_defensibility}\n\n")
        
        f.write("Fragment Details:\n")
        for result in verification.forensic_matches:
            f.write(f"\n  {result.fragment_id}:\n")
            f.write(f"    Matched: {result.matched}\n")
            if result.matched:
                f.write(f"    Position: {result.match_position}\n")
                f.write(f"    Confidence: {result.confidence:.4f}\n")
                if result.match_details:
                    f.write(f"    Details: {result.match_details}\n")
    
    print(f"✓ Saved verification: {verify_report.name}")
    print(f"\n📁 All files saved to: {output_dir}")
    print("\n✅ Export complete")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print(" IMAGE FRAGMENT VERIFICATION EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating forensic image verification using high-entropy fragments")
    
    try:
        example_1_exact_match_verification()
        example_2_cropped_image_detection()
        example_3_modified_image_detection()
        example_4_spatial_search_tolerance()
        example_5_save_and_export()
        
        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  1. Fragment verification enables forensic image authentication")
        print("  2. Spatial search finds fragments even in modified images")
        print("  3. Partial matches indicate cropping or content removal")
        print("  4. High-entropy regions provide reliable forensic anchors")
        print("  5. Confidence scores quantify verification certainty")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
