"""
MinHash Similarity Tests
========================

Tests for MinHash-based document similarity estimation used in watermarking.

MinHash enables fast similarity detection for large documents by computing
Jaccard similarity estimates using hash signatures.

Created: 2026-04-04
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ciaf.watermarks import MinHash, minhash_text, minhash_similarity


class TestMinHashClass:
    """Test MinHash class implementation."""

    def test_compute_basic(self):
        """Test basic MinHash signature computation."""
        text = "The quick brown fox jumps over the lazy dog"
        
        signature = MinHash.compute(text, num_perm=128)
        
        assert len(signature) == 128, "Should have 128 hash values"
        assert all(isinstance(h, int) for h in signature), "All values should be integers"
        assert all(h >= 0 for h in signature), "All hash values should be non-negative"

    def test_compute_identical_text(self):
        """Test that identical text produces identical signatures."""
        text = "Artificial intelligence governance framework"
        
        sig1 = MinHash.compute(text, num_perm=64)
        sig2 = MinHash.compute(text, num_perm=64)
        
        assert sig1 == sig2, "Identical text should produce identical signatures"

    def test_compute_similar_text(self):
        """Test that similar text produces similar signatures."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox leaps over the lazy dog"  # One word different
        
        sig1 = MinHash.compute(text1, num_perm=128)
        sig2 = MinHash.compute(text2, num_perm=128)
        
        # Signatures should be different but similar
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        assert 0.5 < similarity < 1.0, f"Similar text should have high similarity (got {similarity})"

    def test_compute_different_text(self):
        """Test that completely different text produces different signatures."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "Cryptocurrency blockchain decentralized finance DeFi protocols"
        
        sig1 = MinHash.compute(text1, num_perm=128)
        sig2 = MinHash.compute(text2, num_perm=128)
        
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        assert similarity < 0.3, f"Different text should have low similarity (got {similarity})"

    def test_compute_empty_text(self):
        """Test MinHash with empty text."""
        text = ""
        
        signature = MinHash.compute(text, num_perm=64)
        
        assert len(signature) == 64, "Should return signature with requested length"
        assert all(h == 0 for h in signature), "Empty text should produce zero signature"

    def test_compute_case_insensitive(self):
        """Test that MinHash is case-insensitive."""
        text1 = "The Quick Brown Fox"
        text2 = "the quick brown fox"
        
        sig1 = MinHash.compute(text1, num_perm=64)
        sig2 = MinHash.compute(text2, num_perm=64)
        
        assert sig1 == sig2, "MinHash should be case-insensitive"

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical signatures."""
        sig = [100, 200, 300, 400, 500]
        
        similarity = MinHash.jaccard_similarity(sig, sig)
        
        assert similarity == 1.0, "Identical signatures should have similarity 1.0"

    def test_jaccard_similarity_no_match(self):
        """Test Jaccard similarity for completely different signatures."""
        sig1 = [100, 200, 300, 400, 500]
        sig2 = [101, 201, 301, 401, 501]
        
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        
        assert similarity == 0.0, "Completely different signatures should have similarity 0.0"

    def test_jaccard_similarity_partial_match(self):
        """Test Jaccard similarity for partial match."""
        sig1 = [100, 200, 300, 400, 500]
        sig2 = [100, 200, 301, 401, 501]  # First 2 match
        
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        
        assert similarity == 0.4, f"2/5 match should give 0.4 similarity (got {similarity})"

    def test_jaccard_similarity_different_lengths(self):
        """Test that different length signatures raise error."""
        sig1 = [100, 200, 300]
        sig2 = [100, 200]
        
        with pytest.raises(ValueError, match="same length"):
            MinHash.jaccard_similarity(sig1, sig2)

    def test_different_num_perm(self):
        """Test with different number of permutations."""
        text = "Machine learning model training validation testing dataset"
        
        sig32 = MinHash.compute(text, num_perm=32)
        sig64 = MinHash.compute(text, num_perm=64)
        sig128 = MinHash.compute(text, num_perm=128)
        
        assert len(sig32) == 32
        assert len(sig64) == 64
        assert len(sig128) == 128


class TestMinHashFunctions:
    """Test public API functions for MinHash."""

    def test_minhash_text_basic(self):
        """Test basic MinHash text hashing."""
        text = "The quick brown fox jumps over the lazy dog"
        
        hash_value = minhash_text(text, num_perm=128)
        
        assert isinstance(hash_value, str), "Should return string"
        assert len(hash_value) > 0, "Hash should not be empty"
        # Base64 encoding of 128 * 4-byte integers = 512 bytes → ~683 chars
        assert len(hash_value) > 500, f"Hash should be long (got {len(hash_value)})"

    def test_minhash_text_deterministic(self):
        """Test that MinHash is deterministic."""
        text = "Forensic provenance watermarking system"
        
        hash1 = minhash_text(text)
        hash2 = minhash_text(text)
        
        assert hash1 == hash2, "Same text should produce same hash"

    def test_minhash_similarity_identical(self):
        """Test similarity for identical text."""
        text = "Artificial intelligence model governance framework"
        
        hash1 = minhash_text(text)
        hash2 = minhash_text(text)
        
        similarity = minhash_similarity(hash1, hash2)
        
        assert similarity == 1.0, "Identical text should have similarity 1.0"

    def test_minhash_similarity_high(self):
        """Test similarity for highly similar text."""
        text1 = "The quick brown fox jumps over the lazy dog and runs away"
        text2 = "The quick brown fox leaps over the lazy dog and runs away"
        
        hash1 = minhash_text(text1, num_perm=128)
        hash2 = minhash_text(text2, num_perm=128)
        
        similarity = minhash_similarity(hash1, hash2)
        
        print(f"\nSimilarity (1 word different): {similarity:.3f}")
        assert 0.7 < similarity < 1.0, f"Similar text should have high similarity (got {similarity})"

    def test_minhash_similarity_medium(self):
        """Test similarity for moderately similar text."""
        text1 = "Machine learning models require extensive training data and validation datasets"
        text2 = "Deep learning algorithms need large training datasets and proper validation methods"
        
        hash1 = minhash_text(text1, num_perm=128)
        hash2 = minhash_text(text2, num_perm=128)
        
        similarity = minhash_similarity(hash1, hash2)
        
        print(f"\nSimilarity (moderate overlap): {similarity:.3f}")
        assert 0.2 < similarity < 0.7, f"Moderate overlap should give medium similarity (got {similarity})"

    def test_minhash_similarity_low(self):
        """Test similarity for completely different text."""
        text1 = "Artificial intelligence machine learning deep neural networks"
        text2 = "Cryptocurrency blockchain decentralized finance smart contracts"
        
        hash1 = minhash_text(text1, num_perm=128)
        hash2 = minhash_text(text2, num_perm=128)
        
        similarity = minhash_similarity(hash1, hash2)
        
        print(f"\nSimilarity (completely different): {similarity:.3f}")
        assert similarity < 0.3, f"Different text should have low similarity (got {similarity})"


class TestMinHashUseCases:
    """Test real-world use cases for MinHash."""

    def test_detect_plagiarism(self):
        """Test detecting plagiarized content with minor modifications."""
        original = """
        The forensic provenance watermarking system enables detection of AI-generated
        content by embedding cryptographic signatures into artifacts. This ensures
        full traceability and auditability of machine learning model outputs across
        organizational boundaries and regulatory frameworks.
        """
        
        plagiarized = """
        The forensic provenance watermarking framework allows identification of AI-generated
        content by including cryptographic signatures within artifacts. This guarantees
        complete traceability and auditability of machine learning system outputs across
        organizational boundaries and compliance frameworks.
        """
        
        hash_orig = minhash_text(original, num_perm=128)
        hash_plag = minhash_text(plagiarized, num_perm=128)
        
        similarity = minhash_similarity(hash_orig, hash_plag)
        
        print(f"\nPlagiarism similarity: {similarity:.1%}")
        assert similarity > 0.5, "Plagiarized content should have high similarity"

    def test_detect_document_version(self):
        """Test detecting different versions of same document."""
        v1 = """
        Product Specification Document v1.0
        - Feature A: Implementation complete
        - Feature B: In progress
        - Feature C: Not started
        """
        
        v2 = """
        Product Specification Document v2.0
        - Feature A: Implementation complete
        - Feature B: Implementation complete
        - Feature C: In progress
        - Feature D: Not started
        """
        
        hash_v1 = minhash_text(v1, num_perm=128)
        hash_v2 = minhash_text(v2, num_perm=128)
        
        similarity = minhash_similarity(hash_v1, hash_v2)
        
        print(f"\nDocument version similarity: {similarity:.1%}")
        assert 0.4 < similarity <= 1.0, f"Different versions should have moderate to high similarity (got {similarity})"

    def test_large_document_comparison(self):
        """Test MinHash efficiency with large documents."""
        # Simulate large document with repeated paragraphs
        paragraph = "The quick brown fox jumps over the lazy dog. " * 20
        
        doc1 = paragraph * 50  # ~20KB document
        doc2 = paragraph * 50  # Identical
        doc3 = paragraph.replace("quick", "fast") * 50  # Slightly modified
        
        hash1 = minhash_text(doc1, num_perm=128)
        hash2 = minhash_text(doc2, num_perm=128)
        hash3 = minhash_text(doc3, num_perm=128)
        
        sim_identical = minhash_similarity(hash1, hash2)
        sim_modified = minhash_similarity(hash1, hash3)
        
        print(f"\nLarge document - Identical: {sim_identical:.1%}")
        print(f"Large document - Modified: {sim_modified:.1%}")
        
        assert sim_identical == 1.0, "Identical large docs should match"
        assert sim_modified > 0.7, f"Slightly modified large docs should be similar (got {sim_modified})"

    def test_watermark_removal_detection(self):
        """Test using MinHash to detect content after watermark removal."""
        original_text = """
        Financial Analysis Report Q4 2025
        
        Revenue projections show strong growth trajectory across all business units.
        Market sentiment remains positive despite macroeconomic headwinds.
        """
        
        # Add watermark
        watermarked = original_text + "\n\n---\nAI Provenance Tag: wmk-abc123\n"
        
        # Simulate watermark removal (back to original)
        suspected = original_text
        
        # MinHash should detect they're the same content
        hash_watermarked = minhash_text(watermarked, num_perm=128)
        hash_suspected = minhash_text(suspected, num_perm=128)
        
        similarity = minhash_similarity(hash_watermarked, hash_suspected)
        
        print(f"\nWatermark removal similarity: {similarity:.1%}")
        assert similarity > 0.7, f"Content should match after watermark removal (got {similarity})"


class TestMinHashEdgeCases:
    """Test edge cases and error handling."""

    def test_single_word(self):
        """Test MinHash with single word."""
        text = "hello"
        
        hash_value = minhash_text(text, num_perm=64)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_repeated_words(self):
        """Test MinHash with repeated words."""
        text1 = "hello hello hello"
        text2 = "hello"
        
        hash1 = minhash_text(text1, num_perm=64)
        hash2 = minhash_text(text2, num_perm=64)
        
        # Should be identical (MinHash uses set of tokens)
        similarity = minhash_similarity(hash1, hash2)
        assert similarity == 1.0, "Repeated words should not affect MinHash"

    def test_special_characters(self):
        """Test MinHash with special characters."""
        text = "Hello!!! World??? Test... #hashtag @mention"
        
        hash_value = minhash_text(text, num_perm=64)
        
        assert isinstance(hash_value, str)
        # Special chars are ignored, only words extracted
        assert len(hash_value) > 0

    def test_numbers_and_alphanumeric(self):
        """Test MinHash with numbers and alphanumeric strings."""
        text1 = "Model version 1.2.3 released in 2025"
        text2 = "Model version 1.2.3 released in 2026"
        
        hash1 = minhash_text(text1, num_perm=128)
        hash2 = minhash_text(text2, num_perm=128)
        
        similarity = minhash_similarity(hash1, hash2)
        
        # Should be very similar (only year differs)
        assert similarity > 0.7, f"Similar alphanumeric text should match (got {similarity})"

    def test_unicode_text(self):
        """Test MinHash with Unicode characters."""
        text = "Café résumé naïve Zürich 北京 東京"
        
        hash_value = minhash_text(text, num_perm=64)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" MINHASH SIMILARITY TESTS")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
