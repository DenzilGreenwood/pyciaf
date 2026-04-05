"""
Example: Unified Watermarking Interface

Demonstrates the unified watermark_ai_output() function that can watermark
ANY artifact type (text, images, PDF) with a single API call.

This example shows:
1. Automatic type detection
2. Watermarking different artifact types with one function
3. Configuration system for defaults
4. Integration at model inference points

Created: 2026-04-04
"""

from io import BytesIO
from PIL import Image
import os

# Import unified interface
from ciaf.watermarks import (
    watermark_ai_output,
    quick_watermark,
    set_default_watermark_config,
    detect_artifact_type,
    verify_text_artifact,
)


def example_1_automatic_type_detection():
    """Example 1: Automatic type detection."""
    print("\n" + "=" * 70)
    print("Example 1: Automatic Type Detection")
    print("=" * 70)

    # Text artifact
    text = "This is AI-generated text content that needs watermarking."
    print(f"\n1. Text artifact: '{text[:50]}...'")

    evidence, watermarked = watermark_ai_output(
        artifact=text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-17",
        prompt="Generate summary",
        verification_base_url="https://vault.example.com"
    )

    print(f"   ✓ Detected type: {evidence.artifact_type}")
    print(f"   ✓ Artifact ID: {evidence.artifact_id}")
    print(f"   ✓ Watermarked: {watermarked[:80]}...")

    # Image artifact
    img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    print(f"\n2. Image artifact: PNG image ({len(image_bytes)} bytes)")

    evidence, watermarked = watermark_ai_output(
        artifact=image_bytes,
        model_id="stable-diffusion-v3",
        model_version="2026-03",
        actor_id="user:artist-42",
        prompt="Generate landscape",
        verification_base_url="https://vault.example.com"
    )

    print(f"   ✓ Detected type: {evidence.artifact_type}")
    print(f"   ✓ Artifact ID: {evidence.artifact_id}")
    print(f"   ✓ Watermarked: {len(watermarked)} bytes")

    # Save watermarked image
    os.makedirs("output", exist_ok=True)
    with open("output/unified_watermarked_image.png", "wb") as f:
        f.write(watermarked)
    print(f"   ✓ Saved to: output/unified_watermarked_image.png")


def example_2_unified_api_one_function():
    """Example 2: One function for all types."""
    print("\n" + "=" * 70)
    print("Example 2: One Function for All Types")
    print("=" * 70)

    def watermark_any_ai_output(ai_output, model_info):
        """
        Watermark any AI output regardless of type.

        This is how you'd integrate at the inference point.
        """
        evidence, watermarked = watermark_ai_output(
            artifact=ai_output,
            model_id=model_info["model_id"],
            model_version=model_info["model_version"],
            actor_id=model_info["user_id"],
            prompt=model_info["prompt"],
            verification_base_url="https://vault.example.com"
        )
        return watermarked, evidence

    # Test with different types
    model_info = {
        "model_id": "multi-modal-model",
        "model_version": "2026-03",
        "user_id": "user:test-123",
        "prompt": "Generate content"
    }

    # Text
    text_output = "AI generated report content..."
    watermarked_text, text_evidence = watermark_any_ai_output(text_output, model_info)
    print(f"\n1. Text output watermarked:")
    print(f"   Type: {text_evidence.artifact_type}")
    print(f"   ID: {text_evidence.artifact_id}")

    # Image
    img = Image.new("RGB", (200, 200), color="green")
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_output = buf.getvalue()

    watermarked_image, image_evidence = watermark_any_ai_output(image_output, model_info)
    print(f"\n2. Image output watermarked:")
    print(f"   Type: {image_evidence.artifact_type}")
    print(f"   ID: {image_evidence.artifact_id}")

    print("\n✓ Same function handled both types!")


def example_3_configuration_system():
    """Example 3: Configuration system."""
    print("\n" + "=" * 70)
    print("Example 3: Configuration System")
    print("=" * 70)

    # Set global defaults
    print("\n1. Setting global defaults...")
    set_default_watermark_config({
        "verification_base_url": "https://vault.mycompany.com",
        "store_in_vault": False,
        "text": {
            "style": "header",  # Watermark at top
        },
        "image": {
            "opacity": 0.5,
            "include_qr": True,
        }
    })

    # Use defaults
    text = "Content using default configuration"
    evidence, watermarked = watermark_ai_output(
        artifact=text,
        model_id="test-model",
        model_version="1.0",
        actor_id="user:test",
        prompt="Test"
        # verification_base_url not needed - uses default
    )

    print(f"   ✓ Used default URL: {evidence.watermark.verification_url[:50]}...")
    print(f"   ✓ Watermark at top: {watermarked[:50]}...")

    # Override defaults
    print("\n2. Overriding defaults per-call...")
    evidence2, watermarked2 = watermark_ai_output(
        artifact=text,
        model_id="test-model",
        model_version="1.0",
        actor_id="user:test",
        prompt="Test",
        verification_base_url="https://custom.vault.com",  # Override
        watermark_config={"style": "footer"}  # Override
    )

    print(f"   ✓ Used custom URL: {evidence2.watermark.verification_url[:50]}...")
    print(f"   ✓ Watermark at bottom: ...{watermarked2[-80:]}")


def example_4_quick_watermark():
    """Example 4: Quick watermark convenience function."""
    print("\n" + "=" * 70)
    print("Example 4: Quick Watermark")
    print("=" * 70)

    # Simplest possible usage
    text = "Quick watermarking example"

    watermarked, artifact_id = quick_watermark(
        artifact=text,
        model_id="quick-model"
    )

    print(f"\n✓ Watermarked: {watermarked[:100]}...")
    print(f"✓ Artifact ID: {artifact_id}")

    # Works for images too
    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    watermarked_img, img_id = quick_watermark(
        artifact=image_bytes,
        model_id="quick-model"
    )

    print(f"\n✓ Image watermarked: {len(watermarked_img)} bytes")
    print(f"✓ Image ID: {img_id}")


def example_5_type_detection_details():
    """Example 5: Type detection details."""
    print("\n" + "=" * 70)
    print("Example 5: Type Detection Details")
    print("=" * 70)

    # Different artifact types
    test_artifacts = [
        ("Text string", "Hello world"),
        ("PNG image", b'\x89PNG\r\n\x1a\n' + b'\x00' * 100),
        ("JPEG image", b'\xff\xd8\xff' + b'\x00' * 100),
        ("PDF", b'%PDF-1.4\n%Test content'),
        ("JSON", b'{"key": "value"}'),
        ("Binary", b'\x00\x01\x02\x03\x04'),
    ]

    print("\nDetecting types:")
    for name, artifact in test_artifacts:
        detected_type = detect_artifact_type(artifact)
        print(f"   {name:15} → {detected_type}")


def example_6_inference_point_integration():
    """Example 6: Integration at model inference point."""
    print("\n" + "=" * 70)
    print("Example 6: Inference Point Integration")
    print("=" * 70)

    # Simulate a model wrapper
    class AIModelWrapper:
        """Example model wrapper with automatic watermarking."""

        def __init__(self, model_id, auto_watermark=True):
            self.model_id = model_id
            self.auto_watermark = auto_watermark

        def generate(self, prompt, user_id):
            """Generate AI output and automatically watermark."""
            # Step 1: Generate (simulated)
            raw_output = f"AI generated response to: {prompt}"

            # Step 2: Automatically watermark if enabled
            if self.auto_watermark:
                evidence, watermarked = watermark_ai_output(
                    artifact=raw_output,
                    model_id=self.model_id,
                    model_version="2026-03",
                    actor_id=f"user:{user_id}",
                    prompt=prompt,
                    verification_base_url="https://vault.example.com"
                )

                return {
                    "content": watermarked,
                    "evidence": evidence,
                    "watermarked": True
                }
            else:
                return {
                    "content": raw_output,
                    "evidence": None,
                    "watermarked": False
                }

    # Use the wrapped model
    model = AIModelWrapper(model_id="wrapped-gpt-4", auto_watermark=True)

    response = model.generate(
        prompt="Summarize the quarterly report",
        user_id="analyst-17"
    )

    print(f"\n✓ Generated and watermarked automatically!")
    print(f"   Content: {response['content'][:80]}...")
    print(f"   Artifact ID: {response['evidence'].artifact_id}")
    print(f"   Watermarked: {response['watermarked']}")


def example_7_complete_workflow():
    """Example 7: Complete workflow with verification."""
    print("\n" + "=" * 70)
    print("Example 7: Complete Workflow (Watermark + Verify)")
    print("=" * 70)

    # Step 1: Generate and watermark
    text = "This is AI-generated content for the complete workflow example."

    print("\n1. Watermarking...")
    evidence, watermarked = watermark_ai_output(
        artifact=text,
        model_id="workflow-model",
        model_version="1.0",
        actor_id="user:workflow-test",
        prompt="Complete workflow test",
        verification_base_url="https://vault.example.com"
    )

    print(f"   ✓ Artifact ID: {evidence.artifact_id}")
    print(f"   ✓ Watermarked: {len(watermarked)} chars")

    # Step 2: Verify
    print("\n2. Verifying...")
    result = verify_text_artifact(watermarked, evidence)

    print(f"   ✓ Authentic: {result.is_authentic()}")
    print(f"   ✓ Confidence: {result.confidence:.1%}")
    print(f"   ✓ Exact match: {result.exact_match_after_watermark}")

    # Step 3: Test watermark removal detection
    print("\n3. Testing removal detection...")
    stripped = text  # Original without watermark

    result2 = verify_text_artifact(stripped, evidence)
    print(f"   ✓ Watermark removed: {result2.likely_tag_removed}")
    print(f"   ✓ Content authentic: {result2.exact_match_before_watermark}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CIAF Unified Watermarking Interface Examples")
    print("=" * 70)

    example_1_automatic_type_detection()
    example_2_unified_api_one_function()
    example_3_configuration_system()
    example_4_quick_watermark()
    example_5_type_detection_details()
    example_6_inference_point_integration()
    example_7_complete_workflow()

    print("\n" + "=" * 70)
    print("All Examples Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. ✓ One function (watermark_ai_output) handles ALL types")
    print("2. ✓ Automatic type detection - no need to specify")
    print("3. ✓ Configuration system for defaults")
    print("4. ✓ Easy integration at model inference points")
    print("5. ✓ Backward compatible with type-specific functions")
    print("\nSee ciaf/watermarks/unified_interface.py for implementation details.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
