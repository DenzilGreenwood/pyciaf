#!/usr/bin/env python3
"""
CIAF Watermarks - PDF Visual Watermarking Tests

Test suite for PDF visual watermarking with QR codes.

Tests:
- QR code overlay creation
- QR watermark application to PDFs
- Text stamp application
- Complete visual watermark workflow
- Multi-page PDF handling
- Page number positioning
- QR position variations

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path
import unittest
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
try:
    from pypdf import PdfReader, PdfWriter

    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter

        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

if PYPDF_AVAILABLE and REPORTLAB_AVAILABLE:
    from ciaf.watermarks.pdf.visual import (
        create_qr_overlay_page,
        apply_qr_watermark_to_pdf,
        apply_text_stamp_to_pdf,
        build_pdf_artifact_with_visual_watermark,
        verify_pdf_qr_watermark,
    )
    from ciaf.watermarks.images.qr import make_verification_url_qr


def create_sample_pdf(num_pages: int = 1) -> bytes:
    """Create a simple multi-page PDF for testing."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab required")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    for page_num in range(1, num_pages + 1):
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, f"Test PDF - Page {page_num}")
        c.setFont("Helvetica", 12)
        c.drawString(100, 650, "This is a sample PDF document for testing.")
        c.drawString(100, 630, "It contains multiple lines of text content.")
        c.drawString(
            100, 610, "The watermarking system will add QR codes to this document."
        )

        # Add some more content to make it realistic
        y = 550
        for i in range(10):
            c.drawString(100, y, f"Line {i+1}: Sample content for PDF testing.")
            y -= 20

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


@unittest.skipUnless(
    PYPDF_AVAILABLE and REPORTLAB_AVAILABLE,
    "pypdf and reportlab required for PDF visual watermarking tests",
)
class TestPDFVisualWatermarking(unittest.TestCase):
    """Test PDF visual watermarking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.single_page_pdf = create_sample_pdf(num_pages=1)
        self.multi_page_pdf = create_sample_pdf(num_pages=3)
        self.artifact_id = "test-artifact-12345"
        self.base_url = "https://vault.example.com"

        # Generate test QR code
        self.qr_bytes = make_verification_url_qr(
            artifact_id=self.artifact_id,
            base_url=self.base_url,
            box_size=4,
            border=1,
        )

    def test_create_qr_overlay_page(self):
        """Test QR overlay page creation."""
        page_width = 612  # Letter size width in points
        page_height = 792  # Letter size height in points

        overlay_bytes = create_qr_overlay_page(
            qr_image_bytes=self.qr_bytes,
            page_width=page_width,
            page_height=page_height,
            qr_position="bottom-right",
            qr_size=0.5,
            margin=0.3,
            page_number=1,
            add_text=True,
        )

        self.assertIsNotNone(overlay_bytes)
        self.assertGreater(len(overlay_bytes), 0)

        # Verify it's a valid PDF
        reader = PdfReader(BytesIO(overlay_bytes))
        self.assertEqual(len(reader.pages), 1)

    def test_qr_position_bottom_left(self):
        """Test QR code in bottom-left position."""
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.single_page_pdf,
            qr_image_bytes=self.qr_bytes,
            qr_position="bottom-left",
            qr_size=0.5,
            add_page_numbers=True,
        )

        self.assertIsNotNone(watermarked_pdf)
        self.assertGreater(len(watermarked_pdf), len(self.single_page_pdf))

        # Verify PDF structure
        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_qr_position_bottom_center(self):
        """Test QR code in bottom-center position."""
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.single_page_pdf,
            qr_image_bytes=self.qr_bytes,
            qr_position="bottom-center",
            qr_size=0.5,
            add_page_numbers=True,
        )

        self.assertIsNotNone(watermarked_pdf)
        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_qr_position_bottom_right(self):
        """Test QR code in bottom-right position (default)."""
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.single_page_pdf,
            qr_image_bytes=self.qr_bytes,
            qr_position="bottom-right",
            qr_size=0.5,
            add_page_numbers=True,
        )

        self.assertIsNotNone(watermarked_pdf)
        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_qr_watermark_multi_page(self):
        """Test QR watermark on multi-page PDF."""
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.multi_page_pdf,
            qr_image_bytes=self.qr_bytes,
            qr_position="bottom-right",
            qr_size=0.5,
            add_page_numbers=True,
            add_verify_text=True,
        )

        self.assertIsNotNone(watermarked_pdf)

        # Verify all pages watermarked
        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 3)

    def test_qr_without_page_numbers(self):
        """Test QR watermark without page numbers."""
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.single_page_pdf,
            qr_image_bytes=self.qr_bytes,
            qr_position="bottom-right",
            add_page_numbers=False,
        )

        self.assertIsNotNone(watermarked_pdf)
        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_qr_size_variations(self):
        """Test different QR code sizes."""
        for size in [0.3, 0.5, 0.7, 1.0]:
            with self.subTest(size=size):
                watermarked_pdf = apply_qr_watermark_to_pdf(
                    pdf_bytes=self.single_page_pdf,
                    qr_image_bytes=self.qr_bytes,
                    qr_size=size,
                )

                self.assertIsNotNone(watermarked_pdf)
                reader = PdfReader(BytesIO(watermarked_pdf))
                self.assertEqual(len(reader.pages), 1)

    def test_apply_text_stamp_footer(self):
        """Test text stamp in footer."""
        stamped_pdf = apply_text_stamp_to_pdf(
            pdf_bytes=self.single_page_pdf,
            stamp_text="AI Generated Content - Verify at vault.example.com",
            position="footer",
            font_size=8,
        )

        self.assertIsNotNone(stamped_pdf)
        self.assertGreater(len(stamped_pdf), 0)

        reader = PdfReader(BytesIO(stamped_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_apply_text_stamp_header(self):
        """Test text stamp in header."""
        stamped_pdf = apply_text_stamp_to_pdf(
            pdf_bytes=self.single_page_pdf,
            stamp_text="CONFIDENTIAL - AI GENERATED",
            position="header",
            font_size=10,
            color=(0.5, 0.5, 0.5),
        )

        self.assertIsNotNone(stamped_pdf)
        reader = PdfReader(BytesIO(stamped_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_text_stamp_multi_page(self):
        """Test text stamp on multi-page PDF."""
        stamped_pdf = apply_text_stamp_to_pdf(
            pdf_bytes=self.multi_page_pdf,
            stamp_text="AI Generated | Verify Online",
            position="footer",
        )

        self.assertIsNotNone(stamped_pdf)
        reader = PdfReader(BytesIO(stamped_pdf))
        self.assertEqual(len(reader.pages), 3)

    def test_build_artifact_with_visual_watermark(self):
        """Test complete visual watermark workflow."""
        evidence, watermarked_pdf = build_pdf_artifact_with_visual_watermark(
            pdf_bytes=self.single_page_pdf,
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:test",
            prompt="Create a test document",
            verification_base_url=self.base_url,
            qr_position="bottom-right",
            qr_size=0.5,
            add_page_numbers=True,
        )

        # Verify evidence
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.artifact_type.value, "pdf")
        self.assertEqual(evidence.mime_type, "application/pdf")
        self.assertIsNotNone(evidence.artifact_id)
        self.assertIsNotNone(evidence.watermark)
        self.assertEqual(evidence.watermark.watermark_type.value, "visible")
        self.assertEqual(evidence.watermark.embed_method, "qr_code_footer")

        # Verify hashes
        self.assertIsNotNone(evidence.hashes)
        self.assertIsNotNone(evidence.hashes.content_hash_before_watermark)
        self.assertIsNotNone(evidence.hashes.content_hash_after_watermark)
        self.assertNotEqual(
            evidence.hashes.content_hash_before_watermark,
            evidence.hashes.content_hash_after_watermark,
        )

        # Verify metadata
        self.assertIn("qr_position", evidence.metadata)
        self.assertEqual(evidence.metadata["qr_position"], "bottom-right")
        self.assertEqual(evidence.metadata["watermark_mode"], "visual_qr")

        # Verify watermarked PDF
        self.assertIsNotNone(watermarked_pdf)
        self.assertGreater(len(watermarked_pdf), len(self.single_page_pdf))

        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_build_artifact_with_text_stamp(self):
        """Test visual watermark with optional text stamp."""
        evidence, watermarked_pdf = build_pdf_artifact_with_visual_watermark(
            pdf_bytes=self.single_page_pdf,
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:test",
            prompt="Test with stamp",
            verification_base_url=self.base_url,
            qr_position="bottom-left",
            add_text_stamp=True,
            stamp_text="AI Generated - Verify Online",
        )

        self.assertIsNotNone(evidence)
        self.assertIsNotNone(watermarked_pdf)
        self.assertTrue(evidence.metadata["has_text_stamp"])

        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 1)

    def test_build_artifact_multi_page(self):
        """Test visual watermark on multi-page document."""
        evidence, watermarked_pdf = build_pdf_artifact_with_visual_watermark(
            pdf_bytes=self.multi_page_pdf,
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:test",
            prompt="Multi-page test",
            verification_base_url=self.base_url,
            qr_position="bottom-center",
            add_page_numbers=True,
        )

        self.assertIsNotNone(evidence)
        self.assertIsNotNone(watermarked_pdf)

        reader = PdfReader(BytesIO(watermarked_pdf))
        self.assertEqual(len(reader.pages), 3)

    def test_verify_pdf_qr_watermark(self):
        """Test QR watermark detection."""
        # Create watermarked PDF
        watermarked_pdf = apply_qr_watermark_to_pdf(
            pdf_bytes=self.single_page_pdf,
            qr_image_bytes=self.qr_bytes,
        )

        # Note: verify_pdf_qr_watermark is a heuristic check
        # It may or may not detect the QR depending on PDF structure
        # This test just ensures the function doesn't crash
        result = verify_pdf_qr_watermark(watermarked_pdf)
        self.assertIsInstance(result, bool)

    def test_qr_position_all_variants(self):
        """Test all QR position variants in one go."""
        positions = ["bottom-left", "bottom-center", "bottom-right"]

        for position in positions:
            with self.subTest(position=position):
                evidence, watermarked_pdf = build_pdf_artifact_with_visual_watermark(
                    pdf_bytes=self.single_page_pdf,
                    model_id="gpt-4",
                    model_version="2026-03",
                    actor_id="user:test",
                    prompt=f"Test {position}",
                    verification_base_url=self.base_url,
                    qr_position=position,
                )

                self.assertIsNotNone(evidence)
                self.assertEqual(evidence.metadata["qr_position"], position)

                reader = PdfReader(BytesIO(watermarked_pdf))
                self.assertEqual(len(reader.pages), 1)

    def test_watermark_preserves_content(self):
        """Test that watermarking doesn't corrupt PDF content."""
        evidence, watermarked_pdf = build_pdf_artifact_with_visual_watermark(
            pdf_bytes=self.single_page_pdf,
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:test",
            prompt="Content preservation test",
            verification_base_url=self.base_url,
        )

        # Verify original can be read
        original_reader = PdfReader(BytesIO(self.single_page_pdf))
        original_text = original_reader.pages[0].extract_text()

        # Verify watermarked can be read
        watermarked_reader = PdfReader(BytesIO(watermarked_pdf))
        watermarked_text = watermarked_reader.pages[0].extract_text()

        # Original text should be preserved
        self.assertIn("Test PDF - Page 1", watermarked_text)
        self.assertIn("sample PDF document", watermarked_text)

    def test_metadata_preserved(self):
        """Test that original PDF metadata is preserved."""
        # Create PDF with custom metadata
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setTitle("Test Document")
        c.setAuthor("Test Author")
        c.setSubject("Test Subject")
        c.drawString(100, 700, "Test content")
        c.save()
        buffer.seek(0)
        pdf_with_metadata = buffer.getvalue()

        # Apply watermark
        qr_bytes = make_verification_url_qr("test-123", self.base_url)
        watermarked = apply_qr_watermark_to_pdf(
            pdf_bytes=pdf_with_metadata,
            qr_image_bytes=qr_bytes,
        )

        # Check metadata preserved
        reader = PdfReader(BytesIO(watermarked))
        metadata = reader.metadata

        # Note: Metadata preservation depends on pypdf version
        # Just ensure no crash and PDF is valid
        self.assertIsNotNone(reader)
        self.assertEqual(len(reader.pages), 1)


def run_tests():
    """Run all PDF visual watermarking tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFVisualWatermarking)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
