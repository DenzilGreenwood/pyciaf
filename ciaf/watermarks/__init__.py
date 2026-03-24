"""
CIAF Watermarking Module

Forensic provenance system for AI-generated artifacts with dual-state integrity model.

This module implements watermarking and verification for AI outputs,
enabling detection of watermark removal and content tampering.

Key Features:
- Dual-state hashing (before/after watermark)
- Multiple verification strategies
- Format-resilient matching
- Content similarity detection
- Vault integration for persistent storage

Quick Start:
    from ciaf.watermarks import build_text_artifact_evidence, verify_text_artifact

    # Create watermarked artifact
    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text="AI generated content...",
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-17",
        prompt="Generate a summary...",
        verification_base_url="https://vault.example.com"
    )

    # Later: verify suspect text
    result = verify_text_artifact(suspect_text, evidence)
    print(f"Authentic: {result.is_authentic()}")
    print(f"Confidence: {result.confidence:.1%}")

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

# Core models
from .models import (
    ArtifactType,
    WatermarkType,
    ArtifactEvidence,
    ArtifactHashSet,
    ArtifactFingerprint,
    WatermarkDescriptor,
    VerificationResult,
    ForensicArtifactProfile,
    utc_now_iso,
    sha256_bytes,
    sha256_text,
    canonical_json,
)

# Hashing utilities
from .hashing import (
    normalize_text_for_forensics,
    normalized_text_hash,
    strip_common_watermarks,
    text_with_watermark_stripped_hash,
    SimHash,
    simhash_text,
    simhash_distance,
    MinHash,
    minhash_text,
    minhash_similarity,
)

# Text watermarking
from .text import (
    apply_text_watermark,
    build_text_artifact_evidence,
    quick_watermark_text,
    extract_watermark_id,
    extract_verification_url,
    has_watermark,
    remove_watermark,
)

# Verification
from .verify import (
    verify_text_artifact,
    verify_against_multiple_evidence,
    quick_verify,
    analyze_suspect_text,
    format_verification_report,
)

# Vault integration
from .vault_adapter import (
    WatermarkVaultAdapter,
    create_watermark_vault,
)

# Image watermarking (Phase 1)
from .images import (
    ImageWatermarkSpec,
    Position,
    apply_visual_watermark,
    apply_qr_watermark,
    apply_combined_watermark,
    build_image_artifact_evidence,
    PIL_AVAILABLE,
    compute_all_hashes,
    compute_perceptual_hash,
    compute_average_hash,
    compute_difference_hash,
    compute_wavelet_hash,
    hamming_distance,
    similarity_score,
    is_similar_image,
    ImageFingerprintSet,
    IMAGEHASH_AVAILABLE,
    make_qr_code_bytes,
    make_verification_url_qr,
    make_compact_token_qr,
    QRCODE_AVAILABLE,
)

# PDF watermarking (Phase 1)
from .pdf import (
    apply_pdf_metadata_watermark,
    build_pdf_artifact_evidence,
    extract_pdf_metadata_watermark,
    has_pdf_watermark,
    verify_pdf_artifact,
    PYPDF_AVAILABLE,
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "ArtifactType",
    "WatermarkType",

    # Models
    "ArtifactEvidence",
    "ArtifactHashSet",
    "ArtifactFingerprint",
    "WatermarkDescriptor",
    "VerificationResult",
    "ForensicArtifactProfile",

    # Utility functions
    "utc_now_iso",
    "sha256_bytes",
    "sha256_text",
    "canonical_json",

    # Hashing
    "normalize_text_for_forensics",
    "normalized_text_hash",
    "strip_common_watermarks",
    "text_with_watermark_stripped_hash",
    "SimHash",
    "simhash_text",
    "simhash_distance",
    "MinHash",
    "minhash_text",
    "minhash_similarity",

    # Text watermarking
    "apply_text_watermark",
    "build_text_artifact_evidence",
    "quick_watermark_text",
    "extract_watermark_id",
    "extract_verification_url",
    "has_watermark",
    "remove_watermark",

    # Verification
    "verify_text_artifact",
    "verify_against_multiple_evidence",
    "quick_verify",
    "analyze_suspect_text",
    "format_verification_report",

    # Vault integration
    "WatermarkVaultAdapter",
    "create_watermark_vault",

    # Image watermarking (Phase 1)
    "ImageWatermarkSpec",
    "Position",
    "apply_visual_watermark",
    "apply_qr_watermark",
    "apply_combined_watermark",
    "build_image_artifact_evidence",
    "PIL_AVAILABLE",
    "compute_all_hashes",
    "compute_perceptual_hash",
    "compute_average_hash",
    "compute_difference_hash",
    "compute_wavelet_hash",
    "hamming_distance",
    "similarity_score",
    "is_similar_image",
    "ImageFingerprintSet",
    "IMAGEHASH_AVAILABLE",
    "make_qr_code_bytes",
    "make_verification_url_qr",
    "make_compact_token_qr",
    "QRCODE_AVAILABLE",

    # PDF watermarking (Phase 1)
    "apply_pdf_metadata_watermark",
    "build_pdf_artifact_evidence",
    "extract_pdf_metadata_watermark",
    "has_pdf_watermark",
    "verify_pdf_artifact",
    "PYPDF_AVAILABLE",
]
