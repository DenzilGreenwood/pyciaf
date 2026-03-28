"""
Hierarchical Verification Examples - Three-Tier Cost/Accuracy Optimization

This module demonstrates the hierarchical verification strategy in action:
- Tier 1: Fast exact matching (~5 ms)
- Tier 2: DNA fragment sampling (~100 ms)
- Tier 3: Perceptual/similarity matching (~300 ms)

Run this script to see the performance comparison.

Usage:
    python hierarchical_verification_examples.py
"""

from ciaf.watermarks import (
    build_text_artifact_evidence,
    verify_text_artifact_hierarchical,
    format_hierarchical_verification_report,
    VerificationStatistics,
    VerificationTier,
)


def example_1_exact_match():
    """Example 1: Exact match detection (Tier 1 - fastest)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Exact Match Detection (Tier 1 - ~5 ms)")
    print("=" * 70)

    # Create original evidence
    original_text = """
    Artificial intelligence continues to revolutionize how we process information.
    Machine learning models can now analyze complex patterns in large datasets
    with unprecedented accuracy. The applications range from healthcare diagnostics
    to financial forecasting. This represents a transformative shift in technology.
    """

    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-1",
        prompt="Summarize AI trends",
        verification_base_url="https://vault.example.com",
        enable_forensic_fragments=True,
    )

    print(f"\n✓ Created evidence for original artifact")
    print(f"  Original hash: {evidence.hashes.content_hash_before_watermark[:16]}...")
    print(f"  Watermarked hash: {evidence.hashes.content_hash_after_watermark[:16]}...")

    # Case A: Exact copy detected
    print(f"\n[Case A] Exact redistribution:")
    suspect_text_exact = original_text  # Exact copy
    result = verify_text_artifact_hierarchical(suspect_text_exact, evidence)

    print(f"  Result: {result.final_tier.value}")
    print(f"  Confidence: {result.overall_confidence:.1%}")
    print(f"  Time: {result.total_execution_time_ms:.2f} ms")
    print(f"  ✓ Detected in Tier 1 (instant)")

    # Case B: Watermark removed (content unchanged)
    print(f"\n[Case B] Watermark removal:")
    suspect_text_no_watermark = original_text
    result = verify_text_artifact_hierarchical(suspect_text_no_watermark, evidence)

    print(f"  Result: {result.final_tier.value}")
    print(f"  Confidence: {result.overall_confidence:.1%}")
    print(f"  Time: {result.total_execution_time_ms:.2f} ms")
    print(f"  ✓ Detected in Tier 1 (instant)")


def example_2_spliced_content():
    """Example 2: Spliced content detection (Tier 2 - DNA fragments)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Spliced Content Detection (Tier 2 - ~100 ms)")
    print("=" * 70)

    # Create original AI-generated evidence
    ai_text = """
    Quantum computing represents a fundamental shift in computational paradigm.
    Unlike classical computers that use bits, quantum computers harness the power
    of quantum bits (qubits) to explore multiple states simultaneously. This
    enables them to solve certain problems exponentially faster than classical
    approaches. The implications for cryptography, drug discovery, and optimization
    are profound and far-reaching.
    """

    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text=ai_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-2",
        prompt="Explain quantum computing",
        verification_base_url="https://vault.example.com",
        enable_forensic_fragments=True,
    )

    print(f"\n✓ Created evidence for original AI text")

    # Case: Mix 50% AI + 50% human
    human_text = """
    I've been studying quantum computing for years now. My personal experience
    is that the real challenge isn't the physics - it's engineering stable qubits
    at scale. The companies making actual progress right now are focused on error
    correction and maintaining coherence times.
    """

    spliced_text = ai_text[:len(ai_text) // 2] + human_text[len(human_text) // 2 :]

    print(f"\n[Case] Spliced content (50% AI + 50% human):")
    result = verify_text_artifact_hierarchical(spliced_text, evidence)

    print(f"  Result: {result.final_tier.value}")
    print(f"  Confidence: {result.overall_confidence:.1%}")
    print(f"  Time: {result.total_execution_time_ms:.2f} ms")

    if result.final_tier == VerificationTier.TIER2_FRAGMENTS:
        print(f"  ✓ Detected in Tier 2 (DNA fragments)")
        if result.tier2_fragment_results:
            print(
                f"    Fragments matched: {result.tier2_fragment_results.fragments_matched}/{result.tier2_fragment_results.total_fragments_checked}"
            )


def example_3_paraphrased_content():
    """Example 3: Paraphrased content detection (Tier 3 - similarity matching)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Paraphrased Content Detection (Tier 3 - ~300 ms)")
    print("=" * 70)

    # Create original AI-generated evidence
    original_text = """
    Blockchain technology provides a decentralized ledger system that ensures
    data integrity through cryptographic hashing. Each block contains a reference
    to the previous block, creating an immutable chain. The consensus mechanisms
    employed by different blockchain networks determine their security properties
    and transaction throughput.
    """

    evidence, watermarked_text = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:analyst-3",
        prompt="Explain blockchain",
        verification_base_url="https://vault.example.com",
        enable_forensic_fragments=True,
    )

    print(f"\n✓ Created evidence for original text")

    # Paraphrased version (same meaning, different wording)
    paraphrased_text = """
    A decentralized ledger system like blockchain uses cryptographic hashing
    to guarantee that your data hasn't been tampered with. In a blockchain,
    every block points back to the previous one, forming a chain that can't
    be modified without detection. The specific protocols used by various
    blockchain systems determine how secure they are and how many transactions
    they can process per second.
    """

    print(f"\n[Case] Paraphrased content:")
    result = verify_text_artifact_hierarchical(paraphrased_text, evidence)

    print(f"  Result: {result.final_tier.value}")
    print(f"  Confidence: {result.overall_confidence:.1%}")
    print(f"  Time: {result.total_execution_time_ms:.2f} ms")

    if result.final_tier == VerificationTier.TIER3_SIMILARITY:
        print(f"  ✓ Detected in Tier 3 (similarity matching)")


def example_4_batch_performance():
    """Example 4: Performance comparison on batch of mixed verifications."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Performance Analysis")
    print("=" * 70)

    # Create multiple artifacts
    test_cases = [
        ("exact_match", "Exact copy of original", False),
        ("minor_edit", "With a few words changed", False),
        ("paraphrased", "Completely reworded but same meaning", True),
        ("unrelated", "Completely different content", True),
    ]

    # Create original evidence
    original_text = """
    This is the original AI-generated content about machine learning
    and its applications in modern software development. It covers
    various aspects of how machine learning transforms industries.
    """

    evidence, _ = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:batch-test",
        prompt="Test prompt",
        verification_base_url="https://vault.example.com",
        enable_forensic_fragments=True,
    )

    stats = VerificationStatistics()

    print(f"\n✓ Running 4 verification tests:")

    for test_name, description, is_different in test_cases:
        if not is_different:
            suspect = original_text
        else:
            suspect = "Completely different content that has nothing to do with the original"

        result = verify_text_artifact_hierarchical(suspect, evidence)
        stats.add_result(result)

        status = "✓" if result.is_authentic else "✗"
        print(
            f"\n  {status} {test_name:20} ({description})"
            f"\n     → Tier: {result.final_tier.value}, Confidence: {result.overall_confidence:.1%}, Time: {result.total_execution_time_ms:.2f} ms"
        )

    print(f"\n" + "-" * 70)
    print(stats.get_summary())


def example_5_detailed_report():
    """Example 5: Detailed verification report."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Detailed Verification Report")
    print("=" * 70)

    original_text = """
    Deep learning has revolutionized computer vision by enabling machines
    to automatically learn visual features from raw pixel data. Convolutional
    neural networks form the backbone of modern image recognition systems.
    They achieve superhuman performance on tasks like image classification
    and object detection.
    """

    evidence, _ = build_text_artifact_evidence(
        raw_text=original_text,
        model_id="gpt-4",
        model_version="2026-03",
        actor_id="user:detailed-test",
        prompt="Discuss deep learning",
        verification_base_url="https://vault.example.com",
        enable_forensic_fragments=True,
    )

    # Minor edits - should match in Tier 2
    suspect = original_text.replace("superhuman", "exceptional")

    result = verify_text_artifact_hierarchical(suspect, evidence)

    print(f"\nGenerating detailed report:")
    print(format_hierarchical_verification_report(result))


def main():
    """Run all examples."""
    print("\n" + "█" * 70)
    print("HIERARCHICAL VERIFICATION EXAMPLES")
    print("Three-Tier Cost/Accuracy Optimization Strategy")
    print("█" * 70)

    try:
        example_1_exact_match()
        example_2_spliced_content()
        example_3_paraphrased_content()
        example_4_batch_performance()
        example_5_detailed_report()

        print("\n" + "█" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("█" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
