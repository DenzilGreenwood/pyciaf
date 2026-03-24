"""
CIAF Watermarking - PDF Package

PDF watermarking with metadata embedding.

Components:
- metadata.py: Metadata watermarking (Phase 1)

Main Functions:
- build_pdf_artifact_evidence(): Complete watermarked PDF with evidence
- apply_pdf_metadata_watermark(): Embed watermark in PDF metadata
- extract_pdf_metadata_watermark(): Extract watermark from metadata
- verify_pdf_artifact(): Verify PDF against evidence

Future modules:
- visible.py: Visible PDF stamps (Phase 2)
- stamps.py: Header/footer watermarks
- qr.py: QR code embedding in PDFs

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .metadata import (
    apply_pdf_metadata_watermark,
    extract_pdf_metadata_watermark,
    has_pdf_watermark,
    build_pdf_artifact_evidence,
    verify_pdf_artifact,
    PYPDF_AVAILABLE,
)


__all__ = [
    # Watermarking
    "apply_pdf_metadata_watermark",
    "build_pdf_artifact_evidence",

    # Extraction
    "extract_pdf_metadata_watermark",
    "has_pdf_watermark",

    # Verification
    "verify_pdf_artifact",

    # Constants
    "PYPDF_AVAILABLE",
]
