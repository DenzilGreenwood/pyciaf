"""
CIAF Watermarking - Text Implementation

Text watermarking, extraction, and verification.

Implements:
- Footer watermarks (visible)
- Header watermarks (visible)
- Metadata watermarks (invisible)
- QR code generation for verification

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import re
from typing import Optional, Dict, Any, Tuple

from .models import (
    ArtifactEvidence,
    ArtifactType,
    ArtifactHashSet,
    WatermarkDescriptor,
    WatermarkType,
    ArtifactFingerprint,
    utc_now_iso,
)
from .hashing import (
    sha256_text,
    normalized_text_hash,
    simhash_text,
    sha256_bytes,
)


def apply_text_watermark(
    raw_text: str,
    watermark_id: str,
    verification_url: str,
    style: str = "footer",
) -> str:
    """
    Apply visible text watermark to AI-generated content.

    Args:
        raw_text: Original AI output (before watermark)
        watermark_id: Unique watermark identifier
        verification_url: URL for verification
        style: Watermark style ("footer", "header", "inline")

    Returns:
        Watermarked text
    """
    if style == "footer":
        return _apply_footer_watermark(raw_text, watermark_id, verification_url)
    elif style == "header":
        return _apply_header_watermark(raw_text, watermark_id, verification_url)
    elif style == "inline":
        return _apply_inline_watermark(raw_text, watermark_id, verification_url)
    else:
        raise ValueError(f"Unknown watermark style: {style}")


def _apply_footer_watermark(
    text: str,
    watermark_id: str,
    verification_url: str,
) -> str:
    """Apply footer-style watermark."""
    footer = (
        "\n\n"
        "---\n"
        f"AI Provenance Tag: {watermark_id}\n"
        f"Verify: {verification_url}\n"
        "Generated with CIAF (Cognitive Insight Audit Framework)\n"
    )
    return text + footer


def _apply_header_watermark(
    text: str,
    watermark_id: str,
    verification_url: str,
) -> str:
    """Apply header-style watermark."""
    header = (
        f"AI Provenance Tag: {watermark_id}\n"
        f"Verify: {verification_url}\n"
        "---\n\n"
    )
    return header + text


def _apply_inline_watermark(
    text: str,
    watermark_id: str,
    verification_url: str,
) -> str:
    """Apply inline watermark (end of first paragraph)."""
    # Find first paragraph break
    match = re.search(r'\n\n', text)

    if match:
        pos = match.end()
        tag = f" [AI Generated: {watermark_id[:8]}...]"
        return text[:pos] + tag + text[pos:]
    else:
        # No paragraph break, use footer
        return _apply_footer_watermark(text, watermark_id, verification_url)


def extract_watermark_id(text: str) -> Optional[str]:
    """
    Extract watermark ID from text if present.

    Args:
        text: Text to check

    Returns:
        Watermark ID if found, None otherwise
    """
    # Try footer pattern
    match = re.search(r'AI Provenance Tag:\s*([a-zA-Z0-9_-]+)', text)
    if match:
        return match.group(1)

    # Try inline pattern
    match = re.search(r'\[AI Generated:\s*([a-zA-Z0-9_-]+)', text)
    if match:
        return match.group(1)

    return None


def extract_verification_url(text: str) -> Optional[str]:
    """
    Extract verification URL from text if present.

    Args:
        text: Text to check

    Returns:
        Verification URL if found, None otherwise
    """
    match = re.search(r'Verify:\s*(https?://[^\s\n]+)', text)
    if match:
        return match.group(1)
    return None


def has_watermark(text: str) -> bool:
    """
    Check if text contains a CIAF watermark.

    Args:
        text: Text to check

    Returns:
        True if watermark detected
    """
    return extract_watermark_id(text) is not None


def build_text_artifact_evidence(
    raw_text: str,
    model_id: str,
    model_version: str,
    actor_id: str,
    prompt: str,
    verification_base_url: str,
    watermark_style: str = "footer",
    include_simhash: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[ArtifactEvidence, str]:
    """
    Build complete artifact evidence for text with watermarking.

    This is the main function for creating watermarked text artifacts.

    Args:
        raw_text: Original AI-generated text (before watermark)
        model_id: Model identifier
        model_version: Model version
        actor_id: User/system identifier
        prompt: Input prompt
        verification_base_url: Base URL for verification
        watermark_style: Watermark style ("footer", "header", "inline")
        include_simhash: Whether to compute SimHash fingerprints
        additional_metadata: Extra metadata to store

    Returns:
        Tuple of (ArtifactEvidence, watermarked_text)
    """
    # Generate unique IDs
    artifact_id = str(uuid.uuid4())
    watermark_id = f"wmk-{uuid.uuid4()}"
    verification_url = f"{verification_base_url.rstrip('/')}/verify/{artifact_id}"

    # Apply watermark
    watermarked_text = apply_text_watermark(
        raw_text=raw_text,
        watermark_id=watermark_id,
        verification_url=verification_url,
        style=watermark_style,
    )

    # Compute hashes
    prompt_hash = sha256_text(prompt)
    hash_before = sha256_text(raw_text)
    hash_after = sha256_text(watermarked_text)

    # Compute normalized hashes
    norm_before = normalized_text_hash(raw_text)
    norm_after = normalized_text_hash(watermarked_text)

    # Build hash set
    hash_set = ArtifactHashSet(
        content_hash_before_watermark=hash_before,
        content_hash_after_watermark=hash_after,
        normalized_hash_before=norm_before,
        normalized_hash_after=norm_after,
    )

    # Optional: SimHash fingerprints
    fingerprints = []
    if include_simhash:
        simhash_before = simhash_text(raw_text)
        simhash_after = simhash_text(watermarked_text)

        hash_set.simhash_before = simhash_before
        hash_set.simhash_after = simhash_after

        fingerprints.append(ArtifactFingerprint(
            algorithm="simhash",
            value=simhash_before,
            role="exact_content_before_watermark",
        ))
        fingerprints.append(ArtifactFingerprint(
            algorithm="simhash",
            value=simhash_after,
            role="exact_content_after_watermark",
        ))

    # Build watermark descriptor
    watermark = WatermarkDescriptor(
        watermark_id=watermark_id,
        watermark_type=WatermarkType.VISIBLE,
        tag_text=f"AI Provenance Tag: {watermark_id}",
        verification_url=verification_url,
        embed_method=f"{watermark_style}_append_v1",
        removal_resistance="low",  # Text watermarks are easy to remove
        location=watermark_style,
    )

    # Build metadata
    metadata = {
        "distribution_state": "watermarked",
        "artifact_format_version": "1.0",
        "watermark_style": watermark_style,
        "text_length_before": len(raw_text),
        "text_length_after": len(watermarked_text),
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    # Create evidence record
    evidence = ArtifactEvidence(
        artifact_id=artifact_id,
        artifact_type=ArtifactType.TEXT,
        mime_type="text/plain",
        created_at=utc_now_iso(),
        model_id=model_id,
        model_version=model_version,
        actor_id=actor_id,
        prompt_hash=prompt_hash,
        output_hash_raw=hash_before,
        output_hash_distributed=hash_after,
        watermark=watermark,
        hashes=hash_set,
        fingerprints=fingerprints,
        metadata=metadata,
    )

    #Compute receipt hash
    receipt_hash = sha256_bytes(evidence.to_canonical_bytes())
    evidence.hashes.canonical_receipt_hash = receipt_hash

    return evidence, watermarked_text


def quick_watermark_text(
    text: str,
    model_id: str,
    verification_url: str = "https://vault.cognitiveinsight.ai",
) -> Tuple[str, str]:
    """
    Quick watermarking for simple use cases.

    Args:
        text: Text to watermark
        model_id: Model identifier
        verification_url: Base verification URL

    Returns:
        Tuple of (watermarked_text, artifact_id)
    """
    artifact_id = str(uuid.uuid4())
    watermark_id = f"wmk-{uuid.uuid4()}"
    verify_url = f"{verification_url.rstrip('/')}/verify/{artifact_id}"

    watermarked = apply_text_watermark(
        raw_text=text,
        watermark_id=watermark_id,
        verification_url=verify_url,
    )

    return watermarked, artifact_id


def remove_watermark(text: str) -> str:
    """
    Remove CIAF watermark from text.

    Use case: Internal testing, or extracting clean content.

    Args:
        text: Watermarked text

    Returns:
        Text with watermark removed
    """
    # Remove footer watermarks
    text = re.sub(r'\n+---+\n+AI Provenance Tag:.*$', '', text, flags=re.DOTALL | re.MULTILINE)

    # Remove header watermarks
    text = re.sub(r'^AI Provenance Tag:.*?\n+---+\n+', '', text, flags=re.MULTILINE)

    # Remove inline watermarks
    text = re.sub(r'\s*\[AI Generated:.*?\]', '', text)

    return text.strip()


__all__ = [
    # Apply watermarks
    "apply_text_watermark",
    "build_text_artifact_evidence",
    "quick_watermark_text",

    # Extract/detect watermarks
    "extract_watermark_id",
    "extract_verification_url",
    "has_watermark",
    "remove_watermark",
]
