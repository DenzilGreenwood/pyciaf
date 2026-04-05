"""
Tests for CIAF Unified Watermarking Interface

Tests the unified watermark_ai_output() function and automatic type detection.

Run with:
    python -m pytest tests/test_unified_watermarking.py -v
"""

import pytest
from io import BytesIO
from PIL import Image

from ciaf.watermarks.unified_interface import (
    detect_artifact_type,
    watermark_ai_output,
    quick_watermark,
    set_default_watermark_config,
    get_default_watermark_config,
    WatermarkDispatcher,
)
from ciaf.watermarks.models import ArtifactType


# ============================================================================
# TYPE DETECTION TESTS
# ============================================================================


class TestTypeDetection:
    """Test automatic artifact type detection."""

    def test_detect_text_string(self):
        """Test detection of text (string)."""
        text = "This is a text artifact"
        detected_type = detect_artifact_type(text)
        assert detected_type == ArtifactType.TEXT

    def test_detect_png_image(self):
        """Test detection of PNG image."""
        # Create PNG bytes
        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        detected_type = detect_artifact_type(png_bytes)
        assert detected_type == ArtifactType.IMAGE

    def test_detect_jpeg_image(self):
        """Test detection of JPEG image."""
        # Create JPEG bytes
        img = Image.new("RGB", (100, 100), color="blue")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        detected_type = detect_artifact_type(jpeg_bytes)
        assert detected_type == ArtifactType.IMAGE

    def test_detect_pdf(self):
        """Test detection of PDF."""
        # PDF magic bytes
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"

        detected_type = detect_artifact_type(pdf_bytes)
        assert detected_type == ArtifactType.PDF

    def test_detect_json(self):
        """Test detection of JSON."""
        json_bytes = b'{"key": "value", "number": 123}'

        detected_type = detect_artifact_type(json_bytes)
        assert detected_type == ArtifactType.JSON

    def test_detect_json_array(self):
        """Test detection of JSON array."""
        json_bytes = b'[1, 2, 3, "test"]'

        detected_type = detect_artifact_type(json_bytes)
        assert detected_type == ArtifactType.JSON

    def test_detect_invalid_json_as_binary(self):
        """Test that invalid JSON is detected as binary."""
        invalid_json = b'{"key": invalid}'

        detected_type = detect_artifact_type(invalid_json)
        assert detected_type == ArtifactType.BINARY

    def test_detect_binary(self):
        """Test detection of generic binary."""
        binary_data = b"\x00\x01\x02\x03\x04\x05"

        detected_type = detect_artifact_type(binary_data)
        assert detected_type == ArtifactType.BINARY

    def test_detect_empty_bytes(self):
        """Test detection of empty bytes."""
        empty = b""

        detected_type = detect_artifact_type(empty)
        assert detected_type == ArtifactType.BINARY

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Artifact must be str or bytes"):
            detect_artifact_type(123)  # Neither str nor bytes

    def test_detect_webp_image(self):
        """Test detection of WebP image."""
        # WebP magic bytes
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBPVP8 "

        detected_type = detect_artifact_type(webp_bytes)
        assert detected_type == ArtifactType.IMAGE

    def test_detect_gif_image(self):
        """Test detection of GIF image."""
        # GIF89a magic bytes
        gif_bytes = b"GIF89a\x00\x00"

        detected_type = detect_artifact_type(gif_bytes)
        assert detected_type == ArtifactType.IMAGE


# ============================================================================
# UNIFIED WATERMARKING TESTS
# ============================================================================


class TestUnifiedWatermarking:
    """Test unified watermark_ai_output() function."""

    def test_watermark_text_artifact(self):
        """Test watermarking text with auto-detection."""
        text = "This is AI-generated content that needs watermarking."

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test prompt",
            verification_base_url="https://vault.test.com",
        )

        # Verify evidence
        assert evidence.artifact_id is not None
        assert evidence.model_id == "test-model"
        assert evidence.artifact_type == ArtifactType.TEXT

        # Verify watermarked text
        assert isinstance(watermarked, str)
        assert "AI Provenance Tag:" in watermarked
        assert text in watermarked  # Original text should be present

    def test_watermark_image_artifact(self):
        """Test watermarking image with auto-detection."""
        # Create test image
        img = Image.new("RGB", (200, 200), color="green")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        evidence, watermarked = watermark_ai_output(
            artifact=image_bytes,
            model_id="image-gen-model",
            model_version="2.0",
            actor_id="user:artist",
            prompt="Generate green image",
            verification_base_url="https://vault.test.com",
        )

        # Verify evidence
        assert evidence.artifact_id is not None
        assert evidence.model_id == "image-gen-model"
        assert evidence.artifact_type == ArtifactType.IMAGE

        # Verify watermarked image
        assert isinstance(watermarked, bytes)
        assert len(watermarked) > 0

        # Verify it's a valid image
        watermarked_img = Image.open(BytesIO(watermarked))
        assert watermarked_img.size == (200, 200)

    def test_watermark_with_explicit_type(self):
        """Test watermarking with explicit artifact type."""
        text = "Test content"

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url="https://vault.test.com",
            artifact_type=ArtifactType.TEXT,  # Explicit type
        )

        assert evidence.artifact_type == ArtifactType.TEXT
        assert isinstance(watermarked, str)

    def test_watermark_with_custom_config(self):
        """Test watermarking with custom configuration."""
        text = "Test content"

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url="https://vault.test.com",
            watermark_config={"style": "header"},  # Custom config
        )

        # Watermark should be at the top (header style)
        assert watermarked.startswith("AI Provenance Tag:")

    def test_watermark_unsupported_type_raises_error(self):
        """Test that unsupported types raise appropriate errors."""
        json_data = b'{"key": "value"}'

        with pytest.raises(NotImplementedError, match="JSON watermarking not yet"):
            watermark_ai_output(
                artifact=json_data,
                model_id="test-model",
                model_version="1.0",
                actor_id="user:test",
                prompt="Test",
                verification_base_url="https://vault.test.com",
            )

    def test_watermark_with_forensic_fragments(self):
        """Test watermarking with forensic fragments enabled."""
        text = "This is a longer text that should have forensic fragments. " * 10

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url="https://vault.test.com",
            enable_forensic_fragments=True,
        )

        # Note: build_text_artifact_evidence doesn't automatically create
        # forensic fragments - that requires separate fragment selection.
        # This test verifies that the parameter is accepted without error.
        assert evidence.artifact_id is not None
        assert evidence.artifact_type == ArtifactType.TEXT

    def test_watermark_uses_default_url_when_none(self):
        """Test that default URL is used when verification_base_url is None."""
        text = "Test content"

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url=None,  # Should use default
        )

        # Default URL should be present
        assert evidence.watermark.verification_url is not None
        assert "vault.example.com" in evidence.watermark.verification_url


# ============================================================================
# CONFIGURATION SYSTEM TESTS
# ============================================================================


class TestConfigurationSystem:
    """Test configuration defaults and overrides."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_watermark_config()

        assert "verification_base_url" in config
        assert "text" in config
        assert "image" in config

    def test_set_default_config(self):
        """Test setting default configuration."""
        # Set custom default
        set_default_watermark_config({
            "verification_base_url": "https://custom.vault.com",
            "store_in_vault": True,
        })

        config = get_default_watermark_config()
        assert config["verification_base_url"] == "https://custom.vault.com"
        assert config["store_in_vault"] is True

        # Reset to original
        set_default_watermark_config({
            "verification_base_url": "https://vault.example.com",
            "store_in_vault": False,
        })

    def test_config_override_per_call(self):
        """Test that per-call config overrides defaults."""
        # Set default
        set_default_watermark_config({
            "text": {"style": "header"}
        })

        text = "Test content"

        # Override with footer
        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url="https://vault.test.com",
            watermark_config={"style": "footer"},  # Override
        )

        # Should use footer (override)
        assert not watermarked.startswith("AI Provenance Tag:")
        # Footer style adds watermark at the end (with newline)
        assert "AI Provenance Tag:" in watermarked
        assert watermarked.index("AI Provenance Tag:") > watermarked.index("Test content")

        # Reset
        set_default_watermark_config({"text": {"style": "footer"}})


# ============================================================================
# QUICK WATERMARK TESTS
# ============================================================================


class TestQuickWatermark:
    """Test quick_watermark() convenience function."""

    def test_quick_watermark_text(self):
        """Test quick watermarking of text."""
        text = "Quick watermark test"

        watermarked, artifact_id = quick_watermark(
            artifact=text,
            model_id="test-model"
        )

        assert isinstance(watermarked, str)
        assert isinstance(artifact_id, str)
        assert "AI Provenance Tag:" in watermarked

    def test_quick_watermark_image(self):
        """Test quick watermarking of image."""
        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        watermarked, artifact_id = quick_watermark(
            artifact=image_bytes,
            model_id="test-model"
        )

        assert isinstance(watermarked, bytes)
        assert isinstance(artifact_id, str)


# ============================================================================
# DISPATCHER TESTS
# ============================================================================


class TestWatermarkDispatcher:
    """Test WatermarkDispatcher class."""

    def test_dispatcher_text(self):
        """Test dispatcher with text artifact."""
        dispatcher = WatermarkDispatcher()

        evidence, watermarked = dispatcher.dispatch(
            artifact="Test text",
            artifact_type=ArtifactType.TEXT,
            model_id="test-model",
            model_version="1.0",
            actor_id="user:test",
            prompt="Test",
            verification_base_url="https://vault.test.com",
            watermark_config=None,
            enable_forensic_fragments=True,
        )

        assert evidence.artifact_type == ArtifactType.TEXT
        assert isinstance(watermarked, str)

    def test_dispatcher_unsupported_type(self):
        """Test dispatcher with unsupported type."""
        dispatcher = WatermarkDispatcher()

        with pytest.raises(NotImplementedError):
            dispatcher.dispatch(
                artifact=b"audio data",
                artifact_type=ArtifactType.AUDIO,
                model_id="test-model",
                model_version="1.0",
                actor_id="user:test",
                prompt="Test",
                verification_base_url="https://vault.test.com",
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_text_workflow(self):
        """Test complete text watermarking and verification workflow."""
        from ciaf.watermarks import verify_text_artifact

        # Step 1: Watermark
        text = "This is AI-generated content for integration testing."

        evidence, watermarked = watermark_ai_output(
            artifact=text,
            model_id="integration-test-model",
            model_version="1.0",
            actor_id="user:integration-test",
            prompt="Integration test prompt",
            verification_base_url="https://vault.test.com",
        )

        # Step 2: Verify exact match
        result = verify_text_artifact(watermarked, evidence)

        assert result.exact_match_after_watermark is True
        assert result.is_authentic() is True
        assert result.confidence >= 0.99

    def test_multiple_artifact_types_in_sequence(self):
        """Test watermarking multiple artifact types in sequence."""
        # Text
        text_evidence, text_watermarked = watermark_ai_output(
            artifact="Text artifact",
            model_id="multi-model",
            model_version="1.0",
            actor_id="user:multi",
            prompt="Text",
            verification_base_url="https://vault.test.com",
        )

        # Image
        img = Image.new("RGB", (50, 50), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        image_evidence, image_watermarked = watermark_ai_output(
            artifact=image_bytes,
            model_id="multi-model",
            model_version="1.0",
            actor_id="user:multi",
            prompt="Image",
            verification_base_url="https://vault.test.com",
        )

        # Verify different artifact IDs
        assert text_evidence.artifact_id != image_evidence.artifact_id

        # Verify correct types
        assert text_evidence.artifact_type == ArtifactType.TEXT
        assert image_evidence.artifact_type == ArtifactType.IMAGE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
