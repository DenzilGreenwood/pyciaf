#!/usr/bin/env python3
"""
CIAF Watermarks - Performance Benchmark Suite

Benchmarks for critical watermarking operations:
- Text watermarking performance
- Fragment selection/verification
- Perceptual hashing (all algorithms)
- Signature envelope creation/serialization
- Complete workflow throughput

Created: 2026-03-30
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path
import time
import statistics
from typing import List, Dict, Any
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks.text import build_text_artifact_evidence
from ciaf.watermarks.fragment_selection import select_text_forensic_fragments
from ciaf.watermarks.fragment_verification import verify_text_fragments
from ciaf.watermarks.hashing import (
    sha256_text,
    perceptual_hash_image,
    normalized_text_hash,
    simhash_text,
)
from ciaf.watermarks.signature_envelope import (
    create_signature_envelope,
    KeyBackend,
)
from ciaf.watermarks.models import ArtifactEvidence

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - image benchmarks skipped")


class PerformanceBenchmark:
    """Performance benchmark utility."""

    def __init__(self, name: str, iterations: int = 100):
        self.name = name
        self.iterations = iterations
        self.times: List[float] = []

    def run(self, func, *args, **kwargs):
        """Run benchmark and collect timing data."""
        print(f"\n[BENCHMARK] {self.name}")
        print(f"  Iterations: {self.iterations}")

        # Warmup (not counted)
        for _ in range(min(10, self.iterations // 10)):
            func(*args, **kwargs)

        # Actual benchmark
        for i in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            self.times.append(end - start)

        self.report()
        return self.get_stats()

    def report(self):
        """Print benchmark results."""
        if not self.times:
            print("  ❌ No timing data collected")
            return

        mean = statistics.mean(self.times) * 1000  # Convert to ms
        median = statistics.median(self.times) * 1000
        stdev = statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0
        min_time = min(self.times) * 1000
        max_time = max(self.times) * 1000

        print(f"  ✅ Mean:   {mean:.3f} ms")
        print(f"  ✅ Median: {median:.3f} ms")
        print(f"  ✅ StdDev: {stdev:.3f} ms")
        print(f"  ✅ Min:    {min_time:.3f} ms")
        print(f"  ✅ Max:    {max_time:.3f} ms")
        print(f"  ✅ Throughput: {1000/mean:.1f} ops/sec")

    def get_stats(self) -> Dict[str, float]:
        """Get benchmark statistics."""
        if not self.times:
            return {}

        return {
            "mean_ms": statistics.mean(self.times) * 1000,
            "median_ms": statistics.median(self.times) * 1000,
            "stdev_ms": (
                statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0
            ),
            "min_ms": min(self.times) * 1000,
            "max_ms": max(self.times) * 1000,
            "throughput_ops_per_sec": 1000 / (statistics.mean(self.times) * 1000),
        }


def generate_text(size_chars: int) -> str:
    """Generate test text of specific size."""
    base = """The enterprise risk management framework requires comprehensive 
    tracking of AI model deployments. Our governance controls enforce strict 
    audit requirements for all generative AI interactions. Model outputs must 
    include provenance tracking to maintain regulatory compliance. The machine 
    learning pipeline incorporates automated watermarking at multiple stages. 
    """

    # Repeat to reach desired size
    repetitions = (size_chars // len(base)) + 1
    text = (base * repetitions)[:size_chars]
    return text


def benchmark_text_watermarking():
    """Benchmark text watermarking performance across different sizes."""
    print("\n" + "=" * 60)
    print("TEXT WATERMARKING BENCHMARKS")
    print("=" * 60)

    sizes = [500, 1000, 5000, 10000]
    results = {}

    for size in sizes:
        text = generate_text(size)

        def watermark_text():
            evidence, watermarked = build_text_artifact_evidence(
                raw_text=text,
                model_id="gpt-4",
                model_version="2026-03",
                actor_id="user:benchmark",
                prompt="Test prompt",
                verification_base_url="https://vault.example.com",
                include_simhash=True,
            )
            return evidence, watermarked

        benchmark = PerformanceBenchmark(
            f"Text Watermarking ({size} chars)", iterations=50
        )
        stats = benchmark.run(watermark_text)
        results[f"watermark_{size}"] = stats

    return results


def benchmark_fragment_operations():
    """Benchmark fragment selection and verification."""
    print("\n" + "=" * 60)
    print("FRAGMENT OPERATION BENCHMARKS")
    print("=" * 60)

    text = generate_text(5000)
    hash_before = sha256_text(text)
    hash_after = sha256_text(text + " [WATERMARK]")

    results = {}

    # Benchmark fragment selection
    def select_fragments():
        return select_text_forensic_fragments(
            raw_text=text,
            fragment_hash_before=hash_before,
            fragment_hash_after=hash_after,
            min_entropy=0.0,
        )

    benchmark = PerformanceBenchmark("Fragment Selection (5000 chars)", iterations=100)
    stats = benchmark.run(select_fragments)
    results["fragment_selection"] = stats

    # Pre-select fragments for verification benchmark
    fragments = select_fragments()

    # Benchmark fragment verification
    def verify_fragments():
        return verify_text_fragments(text, fragments)

    benchmark = PerformanceBenchmark(
        f"Fragment Verification ({len(fragments)} fragments)", iterations=100
    )
    stats = benchmark.run(verify_fragments)
    results["fragment_verification"] = stats

    return results


def benchmark_hashing_algorithms():
    """Benchmark all hashing algorithms."""
    print("\n" + "=" * 60)
    print("HASHING ALGORITHM BENCHMARKS")
    print("=" * 60)

    text = generate_text(5000)
    results = {}

    # SHA-256 text hashing
    benchmark = PerformanceBenchmark("SHA-256 Text Hash", iterations=500)
    stats = benchmark.run(lambda: sha256_text(text))
    results["sha256"] = stats

    # Normalized text hashing
    benchmark = PerformanceBenchmark("Normalized Text Hash", iterations=500)
    stats = benchmark.run(lambda: normalized_text_hash(text))
    results["normalized_hash"] = stats

    # SimHash
    benchmark = PerformanceBenchmark("SimHash", iterations=100)
    stats = benchmark.run(lambda: simhash_text(text))
    results["simhash"] = stats

    # Perceptual hashing (if PIL available)
    if PIL_AVAILABLE:
        img = Image.new("RGB", (400, 400), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        for algo in ["phash", "ahash", "dhash", "whash"]:
            benchmark = PerformanceBenchmark(
                f"Perceptual Hash ({algo})", iterations=100
            )
            stats = benchmark.run(
                lambda a=algo: perceptual_hash_image(img_bytes, algorithm=a)
            )
            results[f"perceptual_{algo}"] = stats

    return results


def benchmark_signature_envelope():
    """Benchmark signature envelope operations."""
    print("\n" + "=" * 60)
    print("SIGNATURE ENVELOPE BENCHMARKS")
    print("=" * 60)

    results = {}

    payload_hash = "a" * 64
    signature_value = "U2lnbmVkRGF0YQ=="

    # Benchmark envelope creation
    def create_envelope():
        return create_signature_envelope(
            payload_hash=payload_hash,
            signature_value=signature_value,
            key_id="benchmark-key",
            key_backend=KeyBackend.LOCAL,
        )

    benchmark = PerformanceBenchmark("Signature Envelope Creation", iterations=1000)
    stats = benchmark.run(create_envelope)
    results["envelope_creation"] = stats

    # Pre-create envelope for serialization benchmark
    envelope = create_envelope()

    # Benchmark serialization
    benchmark = PerformanceBenchmark(
        "Envelope Serialization (to_dict)", iterations=1000
    )
    stats = benchmark.run(lambda: envelope.to_dict())
    results["envelope_serialization"] = stats

    # Benchmark deserialization
    envelope_dict = envelope.to_dict()

    from ciaf.watermarks.signature_envelope import SignatureEnvelope

    benchmark = PerformanceBenchmark(
        "Envelope Deserialization (from_dict)", iterations=1000
    )
    stats = benchmark.run(lambda: SignatureEnvelope.from_dict(envelope_dict))
    results["envelope_deserialization"] = stats

    return results


def benchmark_complete_workflow():
    """Benchmark complete end-to-end workflow."""
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW BENCHMARK")
    print("=" * 60)

    text = generate_text(2000)

    def complete_workflow():
        # 1. Build artifact with watermark
        evidence, watermarked = build_text_artifact_evidence(
            raw_text=text,
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:benchmark",
            prompt="Test prompt",
            verification_base_url="https://vault.example.com",
            include_simhash=True,
        )

        # 2. Add signature
        envelope = create_signature_envelope(
            payload_hash=evidence.compute_receipt_hash(),
            signature_value="U2lnbmVkQXJ0aWZhY3Q=",
            key_id="benchmark-key",
            key_backend=KeyBackend.LOCAL,
        )
        evidence.signature = envelope

        # 3. Serialize to dict
        evidence_dict = evidence.to_dict()

        # 4. Select fragments
        hash_before = sha256_text(text)
        hash_after = sha256_text(watermarked)
        fragments = select_text_forensic_fragments(
            raw_text=text,
            fragment_hash_before=hash_before,
            fragment_hash_after=hash_after,
            min_entropy=0.0,
        )

        # 5. Verify fragments (if any selected)
        if fragments:
            result = verify_text_fragments(text, fragments)

        return evidence_dict

    benchmark = PerformanceBenchmark("Complete Workflow (2000 chars)", iterations=50)
    stats = benchmark.run(complete_workflow)

    return {"complete_workflow": stats}


def benchmark_signature_overhead():
    """Compare signature envelope vs flat string overhead."""
    print("\n" + "=" * 60)
    print("SIGNATURE OVERHEAD COMPARISON")
    print("=" * 60)

    from ciaf.watermarks.models import (
        ArtifactType,
        ArtifactHashSet,
        WatermarkDescriptor,
        WatermarkType,
    )

    # Create base evidence
    def create_base_evidence():
        return ArtifactEvidence(
            artifact_id="benchmark-artifact",
            artifact_type=ArtifactType.TEXT,
            mime_type="text/plain",
            created_at="2026-03-30T18:00:00Z",
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:benchmark",
            prompt_hash="a" * 64,
            output_hash_raw="b" * 64,
            output_hash_distributed="c" * 64,
            watermark=WatermarkDescriptor(
                watermark_id="wmk-benchmark",
                watermark_type=WatermarkType.VISIBLE,
                verification_url="https://test.example.com",
            ),
            hashes=ArtifactHashSet(
                content_hash_before_watermark="b" * 64,
                content_hash_after_watermark="c" * 64,
            ),
        )

    # Benchmark without signature
    evidence_no_sig = create_base_evidence()

    benchmark = PerformanceBenchmark(
        "Evidence Serialization (no signature)", iterations=1000
    )
    stats_no_sig = benchmark.run(lambda: evidence_no_sig.to_dict())

    # Benchmark with signature envelope
    evidence_with_sig = create_base_evidence()
    envelope = create_signature_envelope(
        payload_hash=evidence_with_sig.compute_receipt_hash(),
        signature_value="U2lnbmVkQXJ0aWZhY3Q=",
        key_id="benchmark-key",
        key_backend=KeyBackend.LOCAL,
    )
    evidence_with_sig.signature = envelope

    benchmark = PerformanceBenchmark(
        "Evidence Serialization (with signature)", iterations=1000
    )
    stats_with_sig = benchmark.run(lambda: evidence_with_sig.to_dict())

    # Calculate overhead
    overhead_ms = stats_with_sig["mean_ms"] - stats_no_sig["mean_ms"]
    overhead_pct = (overhead_ms / stats_no_sig["mean_ms"]) * 100

    print("\n  📊 Signature Envelope Overhead:")
    print(f"     Absolute: {overhead_ms:.3f} ms")
    print(f"     Relative: {overhead_pct:.1f}%")

    return {
        "no_signature": stats_no_sig,
        "with_signature": stats_with_sig,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    }


def generate_performance_report(all_results: Dict[str, Any]):
    """Generate comprehensive performance report."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY REPORT")
    print("=" * 60)

    print("\n📊 TEXT WATERMARKING")
    print("-" * 60)
    for size in [500, 1000, 5000, 10000]:
        key = f"watermark_{size}"
        if key in all_results["watermarking"]:
            stats = all_results["watermarking"][key]
            print(
                f"  {size:5} chars: {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:6.1f} ops/sec)"
            )

    print("\n📊 FRAGMENT OPERATIONS")
    print("-" * 60)
    for op in ["fragment_selection", "fragment_verification"]:
        if op in all_results["fragments"]:
            stats = all_results["fragments"][op]
            op_name = op.replace("_", " ").title()
            print(
                f"  {op_name:25}: {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:6.1f} ops/sec)"
            )

    print("\n📊 HASHING ALGORITHMS")
    print("-" * 60)
    for algo in ["sha256", "normalized_hash", "simhash"]:
        if algo in all_results["hashing"]:
            stats = all_results["hashing"][algo]
            print(
                f"  {algo.upper():15}: {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:7.1f} ops/sec)"
            )

    if PIL_AVAILABLE:
        print("\n  Perceptual Hashing:")
        for algo in ["phash", "ahash", "dhash", "whash"]:
            key = f"perceptual_{algo}"
            if key in all_results["hashing"]:
                stats = all_results["hashing"][key]
                print(
                    f"    {algo:6}: {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:6.1f} ops/sec)"
                )

    print("\n📊 SIGNATURE ENVELOPE")
    print("-" * 60)
    for op in [
        "envelope_creation",
        "envelope_serialization",
        "envelope_deserialization",
    ]:
        if op in all_results["signature"]:
            stats = all_results["signature"][op]
            op_name = op.replace("envelope_", "").replace("_", " ").title()
            print(
                f"  {op_name:20}: {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:7.1f} ops/sec)"
            )

    print("\n📊 SIGNATURE OVERHEAD")
    print("-" * 60)
    overhead = all_results["overhead"]
    print(f"  Without signature:  {overhead['no_signature']['mean_ms']:7.2f} ms")
    print(f"  With signature:     {overhead['with_signature']['mean_ms']:7.2f} ms")
    print(
        f"  Overhead:           {overhead['overhead_ms']:7.2f} ms ({overhead['overhead_pct']:5.1f}%)"
    )

    print("\n📊 COMPLETE WORKFLOW")
    print("-" * 60)
    stats = all_results["workflow"]["complete_workflow"]
    print(
        f"  End-to-End (2000 chars): {stats['mean_ms']:7.2f} ms  ({stats['throughput_ops_per_sec']:6.1f} ops/sec)"
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    # Calculate key metrics
    watermark_1k = all_results["watermarking"]["watermark_1000"]["mean_ms"]
    workflow_2k = all_results["workflow"]["complete_workflow"]["mean_ms"]
    sig_overhead = all_results["overhead"]["overhead_pct"]

    print(f"\n  ✅ Standard watermarking (1000 chars): {watermark_1k:.2f} ms")
    print(f"  ✅ Complete workflow (2000 chars):     {workflow_2k:.2f} ms")
    print(f"  ✅ Signature envelope overhead:        {sig_overhead:.1f}%")

    # Performance targets
    print("\n  🎯 PERFORMANCE TARGETS:")
    if watermark_1k < 50:
        print(f"     ✅ Watermarking < 50ms (target met: {watermark_1k:.2f} ms)")
    else:
        print(
            f"     ⚠️  Watermarking > 50ms (needs optimization: {watermark_1k:.2f} ms)"
        )

    if workflow_2k < 100:
        print(f"     ✅ Complete workflow < 100ms (target met: {workflow_2k:.2f} ms)")
    else:
        print(
            f"     ⚠️  Complete workflow > 100ms (needs optimization: {workflow_2k:.2f} ms)"
        )

    if sig_overhead < 20:
        print(f"     ✅ Signature overhead < 20% (target met: {sig_overhead:.1f}%)")
    else:
        print(
            f"     ⚠️  Signature overhead > 20% (consider optimization: {sig_overhead:.1f}%)"
        )


def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("\n" + "=" * 60)
    print("CIAF WATERMARKS - PERFORMANCE BENCHMARK SUITE")
    print("Testing performance of critical watermarking operations")
    print("=" * 60)

    all_results = {}

    try:
        # Text watermarking benchmarks
        all_results["watermarking"] = benchmark_text_watermarking()

        # Fragment operation benchmarks
        all_results["fragments"] = benchmark_fragment_operations()

        # Hashing algorithm benchmarks
        all_results["hashing"] = benchmark_hashing_algorithms()

        # Signature envelope benchmarks
        all_results["signature"] = benchmark_signature_envelope()

        # Signature overhead comparison
        all_results["overhead"] = benchmark_signature_overhead()

        # Complete workflow benchmark
        all_results["workflow"] = benchmark_complete_workflow()

        # Generate comprehensive report
        generate_performance_report(all_results)

        print("\n" + "=" * 60)
        print("✅ ALL PERFORMANCE BENCHMARKS COMPLETE")
        print("=" * 60)
        print("\n✅ Week 3 Task 3 COMPLETE: Performance Benchmarks")
        print("\n")

        return all_results

    except Exception as e:
        print(f"\n❌ BENCHMARK ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_all_benchmarks()
    sys.exit(0 if results else 1)
