#!/usr/bin/env python3
"""
CIAF Watermarks - LSB Steganography Example

Demonstrates invisible watermarking using LSB (Least Significant Bit) embedding.

This example shows:
1. Embedding invisible watermarks in images
2. Extracting watermarks from watermarked images
3. Comparing visual vs steganographic vs hybrid modes
4. Verification workflows

LSB Steganography:
- Invisible to human eye
- No visible artifacts
- Survives viewing and lossless operations
- Does NOT survive lossy compression (JPEG)
- Best for: Legal evidence, forensic trails, archival

Usage:
    python examples/example_steganography.py

Output:
    Creates watermarked images in ./output/ directory

Created: 2026-04-04
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)

from ciaf.watermarks.images import (
    embed_watermark_lsb,
    extract_watermark_lsb,
    verify_lsb_watermark,
    has_lsb_watermark,
    build_image_artifact_evidence,
    ImageWatermarkSpec,
)


def create_sample_image(width=800, height=600, text="AI Generated Image"):
    """Create a sample image for demonstration."""
    img = Image.new('RGB', (width, height), color=(70, 130, 180))  # Steel blue
    
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([100, 100, 300, 300], fill=(255, 200, 100), outline=(0, 0, 0))
    draw.ellipse([400, 150, 650, 400], fill=(100, 255, 150), outline=(0, 0, 0))
    draw.polygon([(200, 450), (350, 500), (250, 550)], fill=(255, 100, 150))
    
    # Add text
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 50), text, fill=(255, 255, 255))
    
    output = BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()


def example_1_basic_steganography():
    """Example 1: Basic LSB watermark embedding and extraction."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic LSB Steganography")
    print("=" * 70)
    
    # Create sample image
    print("\n1. Creating sample image...")
    image_bytes = create_sample_image(text="Confidential AI Report")
    print(f"   Image size: {len(image_bytes):,} bytes")
    
    # Embed watermark
    print("\n2. Embedding invisible LSB watermark...")
    watermark_id = "wmk-confidential-2026"
    verification_url = "https://vault.cognitiveinsight.ai/verify/conf-001"
    
    watermarked = embed_watermark_lsb(
        image_bytes=image_bytes,
        watermark_id=watermark_id,
        verification_url=verification_url,
        created_at="2026-04-04T14:30:00Z",
        artifact_id="artifact-conf-001",
    )
    
    print(f"   Watermarked size: {len(watermarked):,} bytes")
    print(f"   Size change: {len(watermarked) - len(image_bytes):,} bytes")
    print("   ✓ Watermark embedded (invisible to human eye)")
    
    # Extract watermark
    print("\n3. Extracting watermark...")
    extracted = extract_watermark_lsb(watermarked)
    
    if extracted:
        print("   ✓ Watermark successfully extracted!")
        print(f"   Watermark ID: {extracted['watermark_id']}")
        print(f"   Verification URL: {extracted['verification_url']}")
        print(f"   Created At: {extracted['created_at']}")
        print(f"   Artifact ID: {extracted.get('artifact_id', 'N/A')}")
    else:
        print("   ✗ No watermark found")
    
    # Verification
    print("\n4. Verifying watermark...")
    is_valid, data = verify_lsb_watermark(watermarked, watermark_id)
    
    if is_valid:
        print("   ✓ Watermark verified - matches expected ID")
    else:
        print("   ✗ Watermark verification failed")
    
    # Save watermarked image
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "steganography_basic.png"
    with open(output_path, "wb") as f:
        f.write(watermarked)
    
    print(f"\n5. Saved watermarked image: {output_path}")
    print("   (Visually identical to original - watermark is invisible)")


def example_2_comparison_modes():
    """Example 2: Compare visual vs steganographic vs hybrid watermarking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparing Watermark Modes")
    print("=" * 70)
    
    # Create sample image
    original_bytes = create_sample_image(text="AI Generated Landscape")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Mode 1: Visual watermark (visible text overlay)
    print("\n1. Visual watermark (visible overlay)...")
    visual_spec = ImageWatermarkSpec(
        mode="visual",
        text="AI Generated",
        opacity=0.4,
        position="bottom_right",
    )
    
    evidence_visual, watermarked_visual = build_image_artifact_evidence(
        image_bytes=original_bytes,
        model_id="stable-diffusion-v3",
        model_version="2026.03",
        actor_id="user:demo",
        prompt="Generate landscape image",
        verification_base_url="https://vault.example.com",
        watermark_spec=visual_spec,
    )
    
    visual_path = output_dir / "comparison_visual.png"
    with open(visual_path, "wb") as f:
        f.write(watermarked_visual)
    
    print(f"   ✓ Saved: {visual_path}")
    print(f"   Watermark ID: {evidence_visual.watermark.watermark_id}")
    print(f"   Type: {evidence_visual.watermark.watermark_type.value}")
    print(f"   Removal Resistance: {evidence_visual.watermark.removal_resistance}")
    
    # Mode 2: Steganographic watermark (invisible)
    print("\n2. Steganographic watermark (invisible LSB)...")
    steg_spec = ImageWatermarkSpec(mode="steganographic")
    
    evidence_steg, watermarked_steg = build_image_artifact_evidence(
        image_bytes=original_bytes,
        model_id="stable-diffusion-v3",
        model_version="2026.03",
        actor_id="user:demo",
        prompt="Generate landscape image",
        verification_base_url="https://vault.example.com",
        watermark_spec=steg_spec,
    )
    
    steg_path = output_dir / "comparison_steganographic.png"
    with open(steg_path, "wb") as f:
        f.write(watermarked_steg)
    
    print(f"   ✓ Saved: {steg_path}")
    print(f"   Watermark ID: {evidence_steg.watermark.watermark_id}")
    print(f"   Type: {evidence_steg.watermark.watermark_type.value}")
    print(f"   Removal Resistance: {evidence_steg.watermark.removal_resistance}")
    
    # Verify LSB watermark
    is_present = has_lsb_watermark(watermarked_steg)
    print(f"   LSB watermark detected: {is_present}")
    
    # Mode 3: Hybrid watermark (both visible + LSB)
    print("\n3. Hybrid watermark (visible + LSB)...")
    hybrid_spec = ImageWatermarkSpec(
        mode="hybrid",
        text="AI Generated",
        opacity=0.3,
        position="bottom_right",
    )
    
    evidence_hybrid, watermarked_hybrid = build_image_artifact_evidence(
        image_bytes=original_bytes,
        model_id="stable-diffusion-v3",
        model_version="2026.03",
        actor_id="user:demo",
        prompt="Generate landscape image",
        verification_base_url="https://vault.example.com",
        watermark_spec=hybrid_spec,
    )
    
    hybrid_path = output_dir / "comparison_hybrid.png"
    with open(hybrid_path, "wb") as f:
        f.write(watermarked_hybrid)
    
    print(f"   ✓ Saved: {hybrid_path}")
    print(f"   Watermark ID: {evidence_hybrid.watermark.watermark_id}")
    print(f"   Type: {evidence_hybrid.watermark.watermark_type.value}")
    print(f"   Removal Resistance: {evidence_hybrid.watermark.removal_resistance}")
    
    # Verify both layers
    is_present = has_lsb_watermark(watermarked_hybrid)
    print(f"   LSB watermark detected: {is_present}")
    
    print("\n4. Summary:")
    print("   • Visual: Easy to see, easy to remove (crop)")
    print("   • Steganographic: Invisible, survives viewing (not compression)")
    print("   • Hybrid: Best of both - visible deterrent + invisible proof")


def example_3_forensic_verification():
    """Example 3: Forensic verification workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Forensic Verification Workflow")
    print("=" * 70)
    
    # Create and watermark image
    print("\n1. Creating watermarked evidence...")
    original_bytes = create_sample_image(text="Legal Evidence Document")
    
    spec = ImageWatermarkSpec(mode="steganographic")
    evidence, watermarked = build_image_artifact_evidence(
        image_bytes=original_bytes,
        model_id="legal-doc-gen-v2",
        model_version="2026.04",
        actor_id="system:legal-bot",
        prompt="Generate evidence document",
        verification_base_url="https://vault.legal.example.com",
        watermark_spec=spec,
    )
    
    watermark_id = evidence.watermark.watermark_id
    print(f"   Watermark ID: {watermark_id}")
    print(f"   Artifact ID: {evidence.artifact_id}")
    print(f"   Hash (before): {evidence.hashes.content_hash_before_watermark[:16]}...")
    print(f"   Hash (after): {evidence.hashes.content_hash_after_watermark[:16]}...")
    
    # Simulate receiving suspect image
    print("\n2. Receiving suspect image for verification...")
    suspect_image = watermarked  # In real scenario, this comes from external source
    
    # Extract watermark
    print("\n3. Extracting forensic watermark...")
    extracted = extract_watermark_lsb(suspect_image)
    
    if extracted:
        print("   ✓ Watermark found in suspect image")
        print(f"   Extracted ID: {extracted['watermark_id']}")
        
        # Verify against known watermark
        print("\n4. Verifying against evidence database...")
        if extracted['watermark_id'] == watermark_id:
            print("   ✓ MATCH: Image is authentic")
            print(f"   Generated by: {evidence.model_id}")
            print(f"   Created: {evidence.created_at}")
            print(f"   Actor: {evidence.actor_id}")
        else:
            print("   ✗ MISMATCH: Watermark ID does not match")
    else:
        print("   ✗ No watermark found - image may not be authentic")
    
    # Test with non-watermarked image
    print("\n5. Testing with non-watermarked image...")
    fake_image = create_sample_image(text="Fake Document")
    extracted_fake = extract_watermark_lsb(fake_image)
    
    if extracted_fake:
        print("   Watermark found (unexpected!)")
    else:
        print("   ✓ No watermark - image is not authentic")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CIAF WATERMARKS - LSB STEGANOGRAPHY EXAMPLES")
    print("=" * 70)
    print("\nInvisible watermarking for forensic provenance tracking")
    print("Demonstrates LSB (Least Significant Bit) embedding")
    
    if not PIL_AVAILABLE:
        print("\n⚠️  ERROR: Pillow not installed")
        print("Install with: pip install Pillow")
        return
    
    # Run examples
    example_1_basic_steganography()
    example_2_comparison_modes()
    example_3_forensic_verification()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nCheck ./output/ directory for generated images")
    print("\nNOTE: LSB watermarks are invisible but do NOT survive:")
    print("  • JPEG compression (lossy)")
    print("  • Image resizing")
    print("  • Color adjustments")
    print("  • Filters or effects")
    print("\nBest for: Lossless archival, legal evidence, pristine copies")


if __name__ == "__main__":
    main()
