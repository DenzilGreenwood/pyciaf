"""
CIAF Watermarking - Image Support (DEPRECATED - PLACEHOLDER ONLY)

⚠️ DEPRECATION NOTICE ⚠️

This file is a non-functional placeholder and should NOT be used.

The actual working image watermarking implementation is in:
    ciaf/watermarks/images/

Use the images/ package for:
- apply_visible_watermark() - Logo/text overlay
- apply_invisible_watermark() - Steganographic embedding
- build_image_artifact_evidence() - Full forensic evidence generation
- extract_image_watermark() - Watermark extraction
- compute_perceptual_hash() - True pHash/dHash/aHash/wHash

This placeholder file exists for backwards compatibility only.
It will be removed in a future version.

To migrate:
    # OLD (don't use):
    from ciaf.watermarks.images import apply_image_watermark

    # NEW (use this):
    from ciaf.watermarks.images import apply_visible_watermark

See: ciaf/watermarks/images/__init__.py

Created: 2026-03-24
Deprecated: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0 (DEPRECATED PLACEHOLDER)
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
    Apply watermark to image (DEPRECATED - PLACEHOLDER ONLY).

    ⚠️ This function is a non-functional placeholder.

    Use instead:
        from ciaf.watermarks.images import apply_visible_watermark

    Args:
        image_bytes: Original image data
        watermark_id: Watermark identifier
        verification_url: Verification URL
        style: Watermark style

    Returns:
        Watermarked image bytes

    Raises:
        NotImplementedError: Always - this is a placeholder
    """
    raise NotImplementedError(
        "❌ This function is a deprecated placeholder. "
        "Use: from ciaf.watermarks.images import apply_visible_watermark"
    )


def build_image_artifact_evidence(
    image_bytes: bytes,
    model_id: str,
    model_version: str,
    actor_id: str,
    prompt: str,
    verification_base_url: str,
) -> Tuple[ArtifactEvidence, bytes]:
    """
    Build artifact evidence for image (DEPRECATED - PLACEHOLDER ONLY).

    ⚠️ This function is a non-functional placeholder.

    Use instead:
        from ciaf.watermarks.images import build_image_artifact_evidence

    Returns:
        Tuple of (ArtifactEvidence, watermarked_image_bytes)

    Raises:
        NotImplementedError: Always - this is a placeholder
    """
    raise NotImplementedError(
        "❌ This function is a deprecated placeholder. "
        "Use: from ciaf.watermarks.images import build_image_artifact_evidence"
    )


def extract_image_watermark(image_bytes: bytes) -> Optional[str]:
    """
    Extract watermark ID from image (DEPRECATED - PLACEHOLDER ONLY).

    ⚠️ This function is a non-functional placeholder.

    Use instead:
        from ciaf.watermarks.images import extract_image_watermark

    Returns:
        Watermark ID if found

    Raises:
        NotImplementedError: Always - this is a placeholder
    """
    raise NotImplementedError(
        "❌ This function is a deprecated placeholder. "
        "Use: from ciaf.watermarks.images import extract_image_watermark"
    )


__all__ = [
    "apply_image_watermark",
    "build_image_artifact_evidence",
    "extract_image_watermark",
]
