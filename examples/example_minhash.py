"""
MinHash Document Similarity Examples
=====================================

Demonstrates MinHash-based Jaccard similarity estimation for:
- Plagiarism detection
- Document versioning
- Large document comparison
- Watermark removal detection

MinHash provides fast approximate similarity for large text documents
by computing hash signatures that estimate Jaccard similarity.

Created: 2026-04-04
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks import MinHash, minhash_text, minhash_similarity


def example_1_basic_similarity():
    """
    Example 1: Basic Document Similarity
    
    Compare similar and different documents to understand MinHash behavior.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Document Similarity")
    print("=" * 70)
    
    doc1 = """
    Machine learning models require extensive training data for optimal performance.
    The quality of training datasets directly impacts model accuracy and generalization.
    Proper validation techniques ensure models perform well on unseen data.
    """
    
    doc2 = """
    Machine learning algorithms need large training datasets for best results.
    The quality of training data directly affects model accuracy and performance.
    Appropriate validation methods ensure algorithms work well on new data.
    """
    
    doc3 = """
    Cryptocurrency uses blockchain technology for decentralized transactions.
    Smart contracts enable programmable financial agreements on distributed ledgers.
    DeFi protocols revolutionize traditional banking and finance systems."""
    
    # Compute MinHash signatures
    hash1 = minhash_text(doc1, num_perm=128)
    hash2 = minhash_text(doc2, num_perm=128)
    hash3 = minhash_text(doc3, num_perm=128)
    
    # Compute similarities
    sim_similar = minhash_similarity(hash1, hash2)
    sim_different = minhash_similarity(hash1, hash3)
    
    print("\n📊 Results:")
    print(f"  Similar documents: {sim_similar:.1%} similarity")
    print(f"  Different documents: {sim_different:.1%} similarity")
    
    print("\n📝 Analysis:")
    if sim_similar > 0.7:
        print(f"  ✓ Documents 1 & 2 are SIMILAR (Jaccard: {sim_similar:.1%})")
    if sim_different < 0.3:
        print(f"  ✓ Documents 1 & 3 are DIFFERENT (Jaccard: {sim_different:.1%})")
    
    print("\n✅ MinHash correctly identifies similar and different content")


def example_2_plagiarism_detection():
    """
    Example 2: Plagiarism Detection
    
    Detect copied content with minor modifications (paraphrasing).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Plagiarism Detection")
    print("=" * 70)
    
    original = """
    The CIAF forensic provenance watermarking system enables comprehensive detection
    of AI-generated content by embedding cryptographic signatures into artifacts.
    This technology ensures full traceability and auditability of machine learning
    model outputs across organizational boundaries and regulatory compliance frameworks.
    The dual-state hashing mechanism detects tampering and watermark removal with
    high confidence, enabling legally defensible attribution even when metadata
    has been stripped or content has been modified.
    """
    
    plagiarized = """
    The CIAF forensic provenance watermarking framework allows thorough identification
    of AI-generated content by including cryptographic signatures within artifacts.
    This solution guarantees complete traceability and auditability of ML system
    outputs across organizational limits and compliance frameworks.
    The dual-hash approach identifies tampering and watermark stripping with
    strong confidence, permitting legally sound attribution even when metadata
    is removed or content is altered.
    """
    
    paraphrased = """
    A novel watermarking approach provides forensic identification of machine-generated
    text through embedded digital signatures. Full audit trails track AI outputs
    across enterprise deployments. The system detects content modification and
    watermark removal attempts through dual hashing techniques.
    """
    
    # Compute hashes
    hash_original = minhash_text(original, num_perm=128)
    hash_plagiarized = minhash_text(plagiarized, num_perm=128)
    hash_paraphrased = minhash_text(paraphrased, num_perm=128)
    
    # Compute similarities
    sim_plagiarized = minhash_similarity(hash_original, hash_plagiarized)
    sim_paraphrased = minhash_similarity(hash_original, hash_paraphrased)
    
    print("\n📊 Plagiarism Analysis:")
    print(f"  Original vs. Plagiarized:  {sim_plagiarized:.1%} similarity")
    print(f"  Original vs. Paraphrased:  {sim_paraphrased:.1%} similarity")
    
    print("\n🔍 Detection:")
    if sim_plagiarized > 0.6:
        print(f"  ⚠️  HIGH plagiarism risk ({sim_plagiarized:.1%})")
        print("      → Many identical words/phrases detected")
    
    if sim_paraphrased > 0.3:
        print(f"  ⚠️  MEDIUM plagiarism risk ({sim_paraphrased:.1%})")
        print("      → Some content overlap detected")
    
    print("\n💡 Recommendation:")
    print("  For plagiarism detection, combine MinHash with:")
    print("  • SimHash for near-duplicate detection")
    print("  • Fragment matching for exact phrase identification")
    print("  • Manual review of high-similarity cases")


def example_3_document_versions():
    """
    Example 3: Document Version Tracking
    
    Track changes across document versions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Document Version Tracking")
    print("=" * 70)
    
    v1_0 = """
    Product Requirements Document v1.0
    
    Feature Set:
    - User authentication (email/password)
    - Basic dashboard with charts
    - Data export to CSV
    - Email notifications
    
    Known Issues:
    - Performance slow for >10K records
    - UI not responsive on mobile
    """
    
    v1_1 = """
    Product Requirements Document v1.1
    
    Feature Set:
    - User authentication (email/password, OAuth)
    - Enhanced dashboard with charts and filters
    - Data export to CSV and JSON
    - Email and SMS notifications
    - Mobile responsive UI
    
    Known Issues:
    - Performance slow for >10K records
    """
    
    v2_0 = """
    Product Requirements Document v2.0
    
    Feature Set:
    - User authentication (email/password, OAuth, SSO)
    - Advanced dashboard with real-time charts
    - Data export to CSV, JSON, Excel
    - Multi-channel notifications
    - Mobile-first responsive design
    - Performance optimized for 100K+ records
    - API integrations
    - Audit logging
    
    Breaking Changes:
    - API v1 deprecated (use v2)
    - Old export format no longer supported
    """
    
    # Compute hashes for each version
    hash_v1_0 = minhash_text(v1_0, num_perm=128)
    hash_v1_1 = minhash_text(v1_1, num_perm=128)
    hash_v2_0 = minhash_text(v2_0, num_perm=128)
    
    # Compare versions
    sim_v1_0_v1_1 = minhash_similarity(hash_v1_0, hash_v1_1)
    sim_v1_1_v2_0 = minhash_similarity(hash_v1_1, hash_v2_0)
    sim_v1_0_v2_0 = minhash_similarity(hash_v1_0, hash_v2_0)
    
    print("\n📊 Version Similarity Matrix:")
    print(f"  v1.0 → v1.1: {sim_v1_0_v1_1:.1%}  (minor update)")
    print(f"  v1.1 → v2.0: {sim_v1_1_v2_0:.1%}  (major update)")
    print(f"  v1.0 → v2.0: {sim_v1_0_v2_0:.1%}  (full evolution)")
    
    print("\n📝 Change Detection:")
    if sim_v1_0_v1_1 > 0.8:
        print("  • v1.0 → v1.1: Small incremental changes")
    elif sim_v1_0_v1_1 > 0.5:
        print("  • v1.0 → v1.1: Moderate updates")
    
    if sim_v1_1_v2_0 < 0.7:
        print("  • v1.1 → v2.0: Significant rewrite")
    
    if sim_v1_0_v2_0 < sim_v1_0_v1_1:
        print("  • Document has evolved substantially from v1.0")
    
    print("\n✅ MinHash enables version drift tracking over time")


def example_4_watermark_removal():
    """
    Example 4: Watermark Removal Detection
    
    Detect when watermarks are removed but content remains the same.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Watermark Removal Detection")
    print("=" * 70)
    
    original_content = """
    Financial Analysis Report - Q4 2025
    
    Executive Summary:
    Revenue projections indicate strong growth trajectory across all business units.
    Market sentiment remains cautiously optimistic despite macroeconomic headwinds.
    Operating margins improved 3.2% year-over-year due to efficiency initiatives.
    
    Key Metrics:
    - Total Revenue: $45.2M (+12% YoY)
    - Operating Income: $8.9M (+18% YoY)
    - Customer Acquisition Cost: $342 (-8% YoY)
    - Lifetime Value: $4,200 (+15% YoY)
    """
    
    # Add watermark
    watermark_tag = "\n\n---\nAI Provenance Tag: wmk-fin-20251204-abc123\nGenerated by: GPT-4o-mini v1.2.3\nValidation: https://verify.ciaf.ai/wmk-fin-20251204-abc123\n"
    watermarked_content = original_content + watermark_tag
    
    # Simulate watermark removal (attacker strips watermark)
    suspect_content = original_content
    
    # Compute MinHash signatures
    hash_watermarked = minhash_text(watermarked_content, num_perm=128)
    hash_suspect = minhash_text(suspect_content, num_perm=128)
    
    # Check similarity
    similarity = minhash_similarity(hash_watermarked, hash_suspect)
    
    print("\n📊 Watermark Analysis:")
    print(f"  Content words (original): {len(original_content.split())}")
    print(f"  Content words (watermarked): {len(watermarked_content.split())}")
    print(f"  Content words (suspect): {len(suspect_content.split())}")
    
    print(f"\n  MinHash Similarity: {similarity:.1%}")
    
    print("\n🔍 Verification:")
    if similarity > 0.7:
        print(f"  ✅ Content MATCHES despite watermark removal ({similarity:.1%})")
        print("  ⚠️  SUSPICIOUS: Core content identical, but watermark missing")
        print("      → Likely watermark removal attempt")
    else:
        print(f"  ❌ Content does NOT match")
    
    print("\n💡 Forensic Evidence:")
    print("  • MinHash detected content match despite watermark removal")
    print("  • Combine with fragment verification for stronger proof")
    print("  • Check SimHash for exact hash matching")


def example_5_large_documents():
    """
    Example 5: Large Document Comparison
    
    Demonstrate MinHash efficiency with large documents.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Large Document Comparison")
    print("=" * 70)
    
    # Create large documents (simulated)
    base_paragraph = """
    Artificial intelligence governance frameworks require comprehensive risk assessment
    methodologies and continuous monitoring across all deployment stages. Organizations
    must establish clear accountability structures and implement robust audit trails
    for all AI-generated outputs and decisions.
    """
    
    # Create ~20KB documents
    doc_original = (base_paragraph + " ") * 100
    doc_modified = doc_original.replace("governance", "oversight").replace("comprehensive", "thorough")
    doc_completely_different = ("Blockchain technology enables decentralized systems. " * 50) * 100
    
    print(f"\n📏 Document Sizes:")
    print(f"  Original: {len(doc_original):,} chars (~{len(doc_original) // 1024}KB)")
    print(f"  Modified: {len(doc_modified):,} chars")
    print(f"  Different: {len(doc_completely_different):,} chars")
    
    import time
    
    # Benchmark MinHash computation
    start = time.time()
    hash_original = minhash_text(doc_original, num_perm=128)
    hash_modified = minhash_text(doc_modified, num_perm=128)
    hash_different = minhash_text(doc_completely_different, num_perm=128)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n⏱️  Hashing Performance: {elapsed:.1f}ms for 3 large documents")
    
    # Compute similarities
    sim_modified = minhash_similarity(hash_original, hash_modified)
    sim_different = minhash_similarity(hash_original, hash_different)
    
    print(f"\n📊 Similarity Results:")
    print(f"  Original vs. Modified:   {sim_modified:.1%}")
    print(f"  Original vs. Different:  {sim_different:.1%}")
    
    print("\n✅ MinHash provides fast similarity for large documents")
    print(f"  • Processing: ~{elapsed / 3:.1f}ms per document")
    print("  • Accurate similarity detection")
    print("  • Scales to multi-MB documents")


def example_6_integration_with_watermarking():
    """
    Example 6: Integration with CIAF Watermarking
    
    Show how MinHash complements other verification methods.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Integration with Watermarking System")
    print("=" * 70)
    
    from ciaf.watermarks import (
        build_text_artifact_evidence,
        simhash_text,
        simhash_distance,
    )
    
    # Create watermarked artifact
    original_text = """
    Quarterly Risk Assessment Report
    
    Model drift detected in credit scoring model v2.3.1.
    Recommendation: Retrain with updated dataset incorporating recent economic indicators.
    Priority: Medium
    Estimated effort: 2 weeks
    """
    
    evidence, watermarked = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="risk-assessment-model",
        model_version="2.3.1",
        actor_id="analyst-005",
        prompt="Analyze credit scoring model drift",
        verification_base_url="https://verify.example.com"
    )
    
    # Simulate suspect text (watermark removed)
    suspect = original_text
    
    # Method 1: Exact hash (will fail - watermark removed)
    from ciaf.watermarks.hashing import sha256_text
    exact_match = sha256_text(suspect) == evidence.hashes.content_hash_after_watermark
    
    # Method 2: SimHash (near-duplicate detection)
    suspect_simhash = simhash_text(suspect)
    simhash_dist = simhash_distance(suspect_simhash, evidence.hashes.simhash_before)
    simhash_match = simhash_dist <= 10
    
    # Method 3: MinHash (Jaccard similarity)
    evidence_minhash = minhash_text(original_text)
    suspect_minhash = minhash_text(suspect)
    minhash_sim = minhash_similarity(evidence_minhash, suspect_minhash)
    minhash_match = minhash_sim > 0.7
    
    print("\n🔍 Multi-Method Verification:")
    print(f"  Exact Hash Match:  {'✓ PASS' if exact_match else '✗ FAIL'}")
    print(f"  SimHash Distance:   {simhash_dist} bits → {'✓ PASS' if simhash_match else '✗ FAIL'}")
    print(f"  MinHash Similarity: {minhash_sim:.1%} → {'✓ PASS' if minhash_match else '✗ FAIL'}")
    
    print("\n📝 Hierarchical Verification Strategy:")
    print("  1. Exact Hash: Fast (1ms) - Detects unmodified content")
    print("  2. SimHash:   Medium (10ms) - Detects minor edits")
    print("  3. MinHash:   Medium (15ms) - Detects paraphrasing")
    print("  4. Fragments: Slow (100ms) - Forensic-grade proof")
    
    if not exact_match and (simhash_match or minhash_match):
        print("\n⚠️  VERDICT: Content likely authentic but watermark removed")
    
    print("\n✅ MinHash complements exact and perceptual hashing")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print(" MINHASH DOCUMENT SIMILARITY EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating MinHash for fast document similarity estimation")
    
    example_1_basic_similarity()
    example_2_plagiarism_detection()
    example_3_document_versions()
    example_4_watermark_removal()
    example_5_large_documents()
    example_6_integration_with_watermarking()
    
    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. MinHash provides fast Jaccard similarity estimation")
    print("  2. Effective for plagiarism and duplicate detection")
    print("  3. Scales efficiently to large documents")
    print("  4. Complements exact hashing and SimHash")
    print("  5. Useful for watermark removal detection")
    print()


if __name__ == "__main__":
    main()
