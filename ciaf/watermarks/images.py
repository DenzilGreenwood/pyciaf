"""
CIAF Watermarking - Image Support (Placeholder)

Image watermarking and verification.

TODO: Implement full image watermarking with:
- Visible watermarks (logo/text overlay)
- Invisible watermarks (steganography)
- Perceptual hashing (pHash, dHash, aHash, wHash)
- QR code embedding
- Exif metadata watermarking

For production, integrate:
- Pillow (PIL) for image manipulation
- imagehash for perceptual hashing
- opencv-python for advanced watermarking
- qrcode for QR code generation

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0 (PLACEHOLDER)
"""

from typing import Tuple, Optional
from .models import ArtifactEvidence


def apply_image_watermark(
    image_bytes: bytes,
    watermark_id: str,
    verification_url: str,
    style: str = "corner",
) -> bytes:
    """
    Apply watermark to image (placeholder).

    TODO: Implement with Pillow for:
    - Corner logo watermark
    - Full image overlay
    - QR code embedding
    - Steganographic embedding

    Args:
        image_bytes: Original image data
        watermark_id: Watermark identifier
        verification_url: Verification URL
        style: Watermark style

    Returns:
        Watermarked image bytes
    """
    raise NotImplementedError("Image watermarking not yet implemented. Install Pillow and implement.")


def build_image_artifact_evidence(
    image_bytes: bytes,
    model_id: str,
    model_version: str,
    actor_id: str,
    prompt: str,
    verification_base_url: str,
) -> Tuple[ArtifactEvidence, bytes]:
    """
    Build artifact evidence for image (placeholder).

    TODO: Implement with perceptual hashing.

    Returns:
        Tuple of (ArtifactEvidence, watermarked_image_bytes)
    """
    raise NotImplementedError("Image artifact evidence not yet implemented.")


def extract_image_watermark(image_bytes: bytes) -> Optional[str]:
    """
    Extract watermark ID from image (placeholder).

    TODO: Implement watermark extraction.

    Returns:
        Watermark ID if found
    """
    raise NotImplementedError("Image watermark extraction not yet implemented.")


__all__ = [
    "apply_image_watermark",
    "build_image_artifact_evidence",
    "extract_image_watermark",
]
